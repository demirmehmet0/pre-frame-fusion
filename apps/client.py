import os
import sys
import json
import numpy as np
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from components.FrameFusion.src.models.PackageModel import (
    PackageModel, PackageConfigs, ConfigExecutor, FusionExecutor, FusionRequest, FusionConfigs, FusionInputs,
    InputImages, Alignment, AlignTranslation, Selection, SelectSharpest, KeepPercent, MergeMethod, MergeMean,
    Upscale, Upscale1x,
)
from sdks.novavision.src.base.model import Image
from sdks.novavision.src.media.image import Image as image

ENDPOINT_URL = "http://127.0.0.1:8000/api"


def demo_frames(count=10, seed=0):
    """Noisy, slightly shifted copies of a synthetic barcode-like pattern."""
    rng = np.random.default_rng(seed)
    bars = (rng.random(120) > 0.5).astype(np.float32).repeat(4)
    base = np.tile(bars, (160, 1)) * 180 + 40
    frames = []
    for _ in range(count):
        shifted = np.roll(base, int(rng.integers(-6, 7)), axis=1)
        noisy = np.clip(shifted + rng.normal(0, 25, shifted.shape), 0, 255).astype(np.uint8)
        frames.append(np.dstack([noisy] * 3))
    return frames


def infer():
    images = [image.encode64(Image(name="DemoImage", uID=f"{i:03d}", mimeType="image/jpg", encoding="bytes",
                                   value=frame, type="Image")) for i, frame in enumerate(demo_frames())]
    configs = FusionConfigs(
        alignment=Alignment(value=AlignTranslation()),
        selection=Selection(value=SelectSharpest(keepPercent=KeepPercent(value=50))),
        mergeMethod=MergeMethod(value=MergeMean()),
        upscale=Upscale(value=Upscale1x()),
    )
    req = FusionRequest(inputs=FusionInputs(inputImages=InputImages(value=images)), configs=configs)
    executor = ConfigExecutor(value=FusionExecutor(value=req))
    request = PackageModel(configs=PackageConfigs(executor=executor), name="FrameFusion")
    response = requests.post(ENDPOINT_URL, json=json.loads(request.json()))
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    infer()
