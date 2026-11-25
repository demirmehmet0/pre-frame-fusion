"""
Multi-frame fusion ("lucky imaging") for a short list of frames of the same object.

The frames are scored for sharpness, the sharpest ones are kept, each kept frame is aligned to the sharpest one and
the aligned frames are merged into one image. Merging averages out sensor/compression noise; picking only the
sharpest frames keeps motion blur and defocus out of the result. Pure numpy/OpenCV, no SDK dependency.
"""

import hashlib
import math
from collections import OrderedDict

import cv2
import numpy as np

ALIGN_NONE = "alignNone"
ALIGN_TRANSLATION = "alignTranslation"
ALIGN_HOMOGRAPHY = "alignHomography"
MERGE_MEAN = "mergeMean"
MERGE_MEDIAN = "mergeMedian"

MIN_ALIGNMENT_SCORE = 0.5  # normalised correlation with the reference; below it the frame shows something else
GOOD_ALIGNMENT_SCORE = 0.75  # good enough: no other starting point is tried
MIN_SIDE = 16  # smaller crops cannot be aligned reliably; they are fused without alignment
CACHE_SIZE = 512
MAX_ECC_SIDE = 400  # longest side ECC works on; the transform is scaled back to full size


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.shape[2] == 1:
        return image[:, :, 0].astype(np.float32)
    code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, code).astype(np.float32)


def sharpness(gray: np.ndarray) -> float:
    """
    Variance of the Laplacian after a light blur, measured on the inner 80 % of the frame.
    The blur keeps pixel noise from being counted as detail; the margin keeps crop borders out of the score.
    """
    h, w = gray.shape
    my, mx = int(h * 0.1), int(w * 0.1)
    inner = gray[my : h - my or h, mx : w - mx or w]
    return float(cv2.Laplacian(cv2.GaussianBlur(inner, (0, 0), 1.0), cv2.CV_32F).var())


def fingerprint(image: np.ndarray) -> str:
    """Cheap content key for caching per-frame work (image uIDs are not unique per frame)."""
    step = max(1, min(image.shape[:2]) // 32)
    digest = hashlib.blake2b(np.ascontiguousarray(image[::step, ::step]).tobytes(), digest_size=12)
    digest.update(str(image.shape).encode())
    return digest.hexdigest()


class LRU:
    """Small bounded cache kept in the node state between calls."""

    def __init__(self, size: int = CACHE_SIZE):
        self.size = size
        self.data = OrderedDict()

    def get(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        return None

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        while len(self.data) > self.size:
            self.data.popitem(last=False)


def _fit_to(gray: np.ndarray, shape) -> tuple:
    """Centres a frame on a canvas of the reference size; returns the canvas and the frame->canvas offset."""
    h, w = shape
    fh, fw = gray.shape
    dy, dx = (h - fh) / 2.0, (w - fw) / 2.0
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    canvas = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return canvas, dx, dy


def _correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 0.25 * mask.size:
        return 0.0
    av, bv = a[mask], b[mask]
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = math.sqrt(float((av * av).sum()) * float((bv * bv).sum())) + 1e-9
    return float((av * bv).sum() / denom)


def _ecc_pyramid(ref: np.ndarray, frame: np.ndarray, warp: np.ndarray):
    """
    ECC homography (reference -> frame) refined coarse to fine: 1/4, 1/2 and full working size. Starting small widens the
    range of scale/shift ECC can recover from a rough start; the full-size pass then only polishes, so it gets few
    iterations. Returns None when ECC does not converge at any level.
    """
    top = min(1.0, MAX_ECC_SIDE / max(ref.shape))  # large crops are aligned at reduced size (sub-pixel anyway)
    levels = [f for f in (top / 4, top / 2) if min(ref.shape) * f >= 32] + [top]
    done = False
    for f in levels:
        full = f >= top
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15 if full else 50, 1e-5)
        s = np.diag([f, f, 1.0])
        w = (s @ warp @ np.linalg.inv(s)).astype(np.float32)
        r = ref if f >= 1.0 else cv2.resize(ref, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        m = frame if f >= 1.0 else cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        try:
            _, w = cv2.findTransformECC(r, m, w, cv2.MOTION_HOMOGRAPHY, criteria, None, 5 if full else 3)
            warp = np.linalg.inv(s) @ w.astype(np.float64) @ s
            done = True
        except cv2.error:
            continue
    return warp if done else None


def estimate_transform(frame_gray: np.ndarray, ref_gray: np.ndarray, method: str):
    """
    3x3 matrix that maps frame pixel coordinates onto reference coordinates, and the correlation reached.
    Translation: centre the frame on the reference, then phase correlation for the remaining shift.
    Homography: start from that translation and refine with ECC (handles small scale/rotation/perspective).
    """
    h, w = ref_gray.shape
    canvas, dx, dy = _fit_to(frame_gray, (h, w))
    centred = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
    starts = [_phase_shift(ref_gray, canvas, 1.0) @ centred]
    if method == ALIGN_HOMOGRAPHY:
        # a scale change makes the full-size phase peak unreliable; the coarse one and plain centring are tried too
        starts += [_phase_shift(ref_gray, canvas, 0.25) @ centred, centred]

    scored = sorted(((_score(frame_gray, ref_gray, s), i) for i, s in enumerate(starts)), reverse=True)
    best = (starts[scored[0][1]], scored[0][0])
    if method == ALIGN_HOMOGRAPHY:
        ref_s = cv2.GaussianBlur(ref_gray, (0, 0), 1.0)
        frm_s = cv2.GaussianBlur(frame_gray, (0, 0), 1.0)
        # pad bottom/right only, so frame coordinates stay the same
        frm_s = cv2.copyMakeBorder(
            frm_s, 0, max(0, h - frm_s.shape[0]), 0, max(0, w - frm_s.shape[1]), cv2.BORDER_REPLICATE
        )
        for _, i in scored:  # most promising start first; the others only if ECC ends up poor
            refined = _ecc_pyramid(ref_s, frm_s, np.linalg.inv(starts[i]))
            if refined is None:
                continue
            matrix = np.linalg.inv(refined)
            score = _score(frame_gray, ref_gray, matrix)
            if score > best[1]:
                best = (matrix, score)
            if best[1] >= GOOD_ALIGNMENT_SCORE:
                break
    return best


def _phase_shift(ref: np.ndarray, canvas: np.ndarray, f: float) -> np.ndarray:
    """Remaining shift between two same-size images by phase correlation, measured at scale f (canvas -> ref)."""
    if f < 1.0 and min(ref.shape) * f >= 16:
        ref = cv2.resize(ref, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        canvas = cv2.resize(canvas, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        f = 1.0
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (sx, sy), _ = cv2.phaseCorrelate(ref, canvas, win)
    return np.array([[1, 0, -sx / f], [0, 1, -sy / f], [0, 0, 1]], np.float64)


def _score(frame_gray: np.ndarray, ref_gray: np.ndarray, matrix: np.ndarray) -> float:
    h, w = ref_gray.shape
    warped = cv2.warpPerspective(frame_gray, matrix, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones_like(frame_gray), matrix, (w, h), flags=cv2.INTER_NEAREST) > 0.5
    return _correlation(warped, ref_gray, mask)


def _score_frames(images, cache):
    """Grey copy, sharpness and content key of every frame (sharpness cached by content)."""
    grays, scores, keys = [], [], []
    for img in images:
        key = fingerprint(img)
        gray = to_gray(img)
        score = cache.get(("sharp", key))
        if score is None:
            score = sharpness(gray)
            cache.put(("sharp", key), score)
        grays.append(gray)
        scores.append(score)
        keys.append(key)
    return grays, scores, keys


def _align_selected(selected, grays, keys, alignment, cache):
    """(index, frame->reference matrix) for every selected frame that lines up with the reference."""
    ref = selected[0]
    ref_gray = grays[ref]
    h, w = ref_gray.shape
    can_align = alignment != ALIGN_NONE and min(h, w) >= MIN_SIDE
    transforms = [(ref, np.eye(3))]
    for i in selected[1:]:
        if not can_align:
            _, dx, dy = _fit_to(grays[i], (h, w))
            transforms.append((i, np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)))
            continue
        ck = ("tf", alignment, keys[i], keys[ref])
        hit = cache.get(ck)
        if hit is None:
            hit = estimate_transform(grays[i], ref_gray, alignment)
            cache.put(ck, hit)
        matrix, score = hit
        if score >= MIN_ALIGNMENT_SCORE:
            transforms.append((i, matrix))
    return transforms


def _match_channels(img: np.ndarray, channels: int) -> np.ndarray:
    if img.ndim == 2:
        return img if channels == 1 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == channels:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0] if channels == 1 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels == 1 else img[:, :, :channels]


def _merge(images, transforms, out_size, upscale, merge):
    """Warps the aligned frames onto the (upscaled) reference grid and combines them; uncovered areas keep the
    reference frame."""
    out_w, out_h = out_size
    ref_img = images[transforms[0][0]]
    channels = 1 if ref_img.ndim == 2 else ref_img.shape[2]
    scale = np.diag([upscale, upscale, 1.0])
    layers, masks = [], []
    for i, matrix in transforms:
        img = _match_channels(images[i], channels)
        m = scale @ matrix
        layers.append(
            cv2.warpPerspective(
                img.astype(np.float32), m, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
        )
        valid = np.ones(img.shape[:2], np.float32)
        masks.append(cv2.warpPerspective(valid, m, (out_w, out_h), flags=cv2.INTER_LINEAR) > 0.99)
    stack, valid = np.stack(layers), np.stack(masks)
    valid_c = valid[..., None] if stack.ndim == 4 else valid
    if merge == MERGE_MEDIAN and len(layers) > 2:
        fused = np.nanmedian(np.where(valid_c, stack, np.nan), axis=0)
    else:
        weight = valid_c.astype(np.float32)
        fused = (stack * weight).sum(0) / np.maximum(weight.sum(0), 1e-6)
    holes = ~valid.any(0)
    if holes.any():
        fused[holes] = layers[0][holes]
    fused = np.nan_to_num(fused, nan=0.0)
    return np.clip(np.rint(fused), 0, 255).astype(np.uint8)


def fuse(images, alignment=ALIGN_HOMOGRAPHY, keep_percent=100, merge=MERGE_MEAN, upscale=1, cache=None):
    """
    Fuses a list of frames of the same object into one image.

    images: list of HxW or HxWxC uint8 arrays (sizes may differ slightly, e.g. crops of a moving object).
    Returns (fused uint8 image, index of the reference frame in `images`, info dict).
    """
    if not images:
        raise ValueError("No frames to fuse.")
    cache = cache if cache is not None else LRU()
    grays, scores, keys = _score_frames(images, cache)
    order = sorted(range(len(images)), key=lambda i: -scores[i])
    keep = max(1, math.ceil(len(images) * min(max(keep_percent, 1), 100) / 100.0))
    selected = order[:keep]
    ref = selected[0]
    transforms = _align_selected(selected, grays, keys, alignment, cache)
    h, w = grays[ref].shape
    fused = _merge(images, transforms, (w * upscale, h * upscale), upscale, merge)
    info = {
        "inputFrames": len(images),
        "selectedFrames": len(selected),
        "usedFrames": len(transforms),
        "referenceSharpness": round(scores[ref], 2),
        "meanSharpness": round(float(np.mean([scores[i] for i, _ in transforms])), 2),
    }
    return fused, ref, info
