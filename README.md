# Frame Fusion

> Merges the sharpest of several recent frames of the same object into one cleaner image, so barcode and text readers see less noise and less blur.

**Category:** Preprocessing

**Status:** Beta

**Last Updated:** 25.11.2026

---

## 1. Overview

Frame Fusion takes a list of recent frames of one object, usually the output of Stack Image placed after Crop, and returns a single image. It scores every frame for sharpness, keeps the sharpest share, aligns each kept frame to the sharpest one and merges them. Merging removes sensor and compression noise; leaving blurred frames out keeps fine lines sharp. This approach is known as "lucky imaging".

The node keeps no frames itself. Collecting frames is Stack Image's job; this node only merges what it receives. Each call works on the list it is given, and per-frame alignment results are cached per node, so a frame is aligned only once while it stays in the list.

**Typical use cases:**

- Reading a barcode or label on a moving pallet or conveyor when no single frame is sharp and clean enough.
- OCR on plates, labels or displays filmed in low light, where single frames are noisy.
- Reading a gauge or a small printed code from a camera that is far away.

---

## 2. Inputs

| Field Name | Kind/Type | Required | Default | Description |
|---|---|---|---|---|
| `inputImages` | list of `Image` or `Image` | Yes | — | Recent frames of the same object (e.g. Stack Image `outputImages`). Sizes may differ slightly. A single image is passed through. |

> Derived from the package `Inputs` / `Input` classes. A field that accepts a single object or a list is shown explicitly.

---

## 3. Configuration Parameters

| Parameter | Type | Field Type (UI control) | Allowed Values / Range | Default | Description |
|---|---|---|---|---|---|
| `Alignment` | string | dropdownlist | Shift, Scale and Perspective (`alignHomography`) · Shift (`alignTranslation`) · None (`alignNone`) | `alignHomography` | How frames are aligned to the sharpest one |
| `Selection` | string | dependentDropdownlist | Sharpest Frames (`selectSharpest`) · All Frames (`selectAll`) | `selectSharpest` | Which frames are merged |
| `KeepPercent` | number | textInput | 1–100 | 30 | Share of frames kept, sharpest first (Sharpest Frames only) |
| `MergeMethod` | string | dropdownlist | Mean (`mergeMean`) · Median (`mergeMedian`) | `mergeMean` | How aligned frames are combined |
| `Upscale` | number | dropdownlist | 1x (`1`) · 2x (`2`) | `1` | Output size relative to the sharpest frame |

Task: `Fusion` (the only task).

> Derived from the package `Configs` classes, including validation bounds and dropdown option labels where present. `ConfigExecutor` is not included.

---

## 4. Outputs

| Field Name | Kind/Type | Description |
|---|---|---|
| `outputImage` | `Image` | The fused frame, same size as the sharpest input frame (twice that with 2x). Stored in Redis; `value` is empty. |
| `outputData` | object | `inputFrames`, `selectedFrames`, `usedFrames`, `referenceSharpness`, `meanSharpness` |

> Derived from the package `Outputs` classes.

---

## 5. Use Case Examples

**Use case 1: Barcode on a moving pallet**

`VideoFeed → Yolo (barcode model) → Crop (Dynamic, padding 10–20 %) → Stack Image (Stack Size 20) → Frame Fusion (Shift, Scale and Perspective; Sharpest Frames 30 %; Mean) → Filtering (Sharpen) → Scaling → Barcode Reader`

Stack Image keeps the last 20 crops; Frame Fusion merges the sharpest 6 of them. Put Stack Image after Crop: it shrinks frames larger than 1920x1080, which would cost detail on a full 4K frame.

**Use case 2: Read only when enough frames were fused**

Connect `outputData` to If (`usedFrames >= 4`) and pass `outputImage` to the reader only then, so the reader does not run on a single noisy frame right after the object appears.

---

## 6. Limitations and Notes

- One object per list. Stack Image stacks one image per call; when Crop outputs several boxes, only the first is stacked.
- Frames that do not line up with the sharpest one (correlation below 0.5) are left out. When the object changes (next pallet), old frames drop out by themselves until Stack Image has refilled; `usedFrames` shows it.
- Alignment cost: Shift ≈ 25 ms, Shift, Scale and Perspective ≈ 110 ms per call for 20 crops of 730x480 on an Apple M-series CPU (only new frames are aligned when the sharpest frame stays the same). Large crops are aligned at a longest side of 400 px; the result is applied at full size.
- Fusion removes noise and avoids blurred frames; it cannot recover detail that is missing from every frame.
- Median needs at least 3 used frames; with fewer, Mean is used.

---

## 7. Development

**Setup (once per clone):**

```bash
pip install pre-commit
pre-commit install
```

From then on, each commit runs the same checks as CI:

- **Spec check:** `PackageModel.py` and the executors follow the package development specification.
- **Lint and format:** ruff fixes and formats `src/` automatically. Stage the changes and commit again.
- **Commit message:** must follow [Conventional Commits](https://www.conventionalcommits.org).

To run all checks at any time, use `pre-commit run --all-files`.

**Workflow:**

1. Branch off `develop` and open a PR back into `develop`. CI must pass.
2. Write commit messages as `<type>[(scope)]: <description>`, for example `feat: add threshold config` or `fix(executor): handle empty input`. The type sets the next version:

   | Type | Version bump |
   |---|---|
   | `fix`, `perf` | patch |
   | `feat` | minor |
   | `feat!` or a `BREAKING CHANGE:` footer | major |
   | `refactor`, `docs`, `style`, `test`, `build`, `ci`, `chore` | none |

3. Merge `develop` into `main` to release. A release PR is then opened that bumps the version in `setup.py` and updates `CHANGELOG.md`. Merging it creates the tag and the GitHub Release. Don't edit the version yourself.

- Schema: `python apps/export.py` writes `data.json`. Try against a running host: `python apps/client.py`.
- The fusion itself is in `src/utils/fusion.py` (numpy/OpenCV only, no SDK), so it can be tried on saved frames directly.
