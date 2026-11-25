"""
Fuses a list of recent frames of the same object into one sharper, less noisy image (lucky imaging).
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.FrameFusion.src.utils.response import build_response
from components.FrameFusion.src.utils.fusion import LRU, fuse
from components.FrameFusion.src.models.PackageModel import PackageModel


class Fusion(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.images = self.request.get_param("inputImages")
        self.alignment = self.request.get_param("Alignment")
        self.selection = self.request.get_param("Selection")
        self.keep_percent = self.request.get_param("KeepPercent")
        self.merge = self.request.get_param("MergeMethod")
        self.upscale = self.request.get_param("Upscale")
        self.image = None
        self.data = {}

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {"caches": {}}

    def _cache(self) -> LRU:
        # bootstrap is shared by every node of this executor in the container: keep one cache per node
        caches = self.bootstrap.setdefault("caches", {})
        key = self.uID or "default"
        if key not in caches:
            caches[key] = LRU()
        return caches[key]

    def run(self):
        items = self.images if isinstance(self.images, list) else [self.images]
        items = [item for item in items if item]
        if not items:
            raise ValueError("inputImages is empty.")
        frames = [Image.get_frame(img=item, redis_db=self.redis_db) for item in items]
        keep = int(self.keep_percent or 100) if self.selection == "selectSharpest" else 100
        fused, ref, self.data = fuse(
            [np.asarray(frame.value) for frame in frames],
            alignment=self.alignment or "alignHomography",
            keep_percent=keep,
            merge=self.merge or "mergeMean",
            upscale=int(self.upscale or 1),
            cache=self._cache(),
        )
        out = frames[ref]
        out.value = fused
        self.image = Image.set_frame(img=out, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self, executor_name="Fusion")


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
