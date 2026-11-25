from sdks.novavision.src.helper.package import PackageHelper
from components.FrameFusion.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    FusionOutputs,
    OutputImage,
    OutputData,
    FusionResponse,
    FusionExecutor,
)

_EXECUTORS = {
    "Fusion": (FusionResponse, FusionExecutor),
}


def build_response(context, executor_name):
    response_class, executor_class = _EXECUTORS[executor_name]
    outputs = FusionOutputs(outputImage=OutputImage(value=context.image), outputData=OutputData(value=context.data))
    executor = ConfigExecutor(value=executor_class(value=response_class(outputs=outputs)))
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    return package.build_model(context)
