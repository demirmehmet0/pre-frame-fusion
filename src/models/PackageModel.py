from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package,
    Image,
    Inputs,
    Configs,
    Outputs,
    Response,
    Request,
    Output,
    Input,
    Config,
)


class InputImages(Input):
    name: Literal["inputImages"] = "inputImages"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Images"
        json_schema_extra = {
            "inputNotes": [
                "Recent frames of the same object, e.g. the output of Stack Image placed after Crop.",
                "Frame sizes may differ slightly; every frame is aligned to the sharpest one.",
            ]
        }


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"
        json_schema_extra = {
            "outputStructure": {
                "name": "string",
                "uID": "string",
                "mimeType": "string",
                "encoding": "string",
                "value": "string",
                "r_key": "string",
                "shape_key": "string",
            },
            "outputNotes": [
                "Image frame is stored in Redis (r_key/shape_key); value is empty. Connect only to image inputs.",
                "Same size as the sharpest input frame (twice that with Upscale 2x).",
            ],
        }


class OutputData(Output):
    name: Literal["outputData"] = "outputData"
    value: dict = Field(default_factory=dict)
    type: Literal["object"] = "object"

    class Config:
        title = "Fusion Info"
        json_schema_extra = {
            "outputStructure": {
                "inputFrames": "number",
                "selectedFrames": "number",
                "usedFrames": "number",
                "referenceSharpness": "number",
                "meanSharpness": "number",
            },
            "outputNotes": [
                "usedFrames can be lower than selectedFrames: frames that could not be aligned are left out.",
                "Use it with If to read only when enough frames were fused.",
            ],
        }


class AlignNone(Config):
    name: Literal["alignNone"] = "alignNone"
    value: Literal["alignNone"] = "alignNone"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "None (Fixed Camera and Object)"


class AlignTranslation(Config):
    name: Literal["alignTranslation"] = "alignTranslation"
    value: Literal["alignTranslation"] = "alignTranslation"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Shift"


class AlignHomography(Config):
    name: Literal["alignHomography"] = "alignHomography"
    value: Literal["alignHomography"] = "alignHomography"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Shift, Scale and Perspective"


class Alignment(Config):
    """
    How frames are aligned to the sharpest one before merging.
    'Shift, Scale and Perspective' follows the object moving, getting closer, tilting or turning.
    'Shift' is several times faster but only follows sideways movement; frames that also changed scale are left out.
    """

    name: Literal["Alignment"] = "Alignment"
    value: Union[AlignHomography, AlignTranslation, AlignNone]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Alignment"
        json_schema_extra = {"shortDescription": "Frame Alignment"}


class KeepPercent(Config):
    """
    Share of the frames, sharpest first, that are merged. Blurred frames are left out.
    """

    name: Literal["KeepPercent"] = "KeepPercent"
    value: int = Field(ge=1, le=100, default=30)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[1-100]"] = "[1-100]"

    class Config:
        title = "Keep Sharpest (%)"
        json_schema_extra = {"shortDescription": "Sharpest Frames Kept"}


class SelectAll(Config):
    name: Literal["selectAll"] = "selectAll"
    value: Literal["selectAll"] = "selectAll"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "All Frames"


class SelectSharpest(Config):
    name: Literal["selectSharpest"] = "selectSharpest"
    value: Literal["selectSharpest"] = "selectSharpest"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    keepPercent: KeepPercent

    class Config:
        title = "Sharpest Frames"


class Selection(Config):
    """
    Which frames are merged. 'Sharpest Frames' leaves motion-blurred and out-of-focus frames out, which keeps fine
    lines (barcodes, text) sharp. 'All Frames' removes the most noise but averages the blur in as well.
    """

    name: Literal["Selection"] = "Selection"
    value: Union[SelectSharpest, SelectAll]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Frame Selection"
        json_schema_extra = {"shortDescription": "Frame Selection"}


class MergeMean(Config):
    name: Literal["mergeMean"] = "mergeMean"
    value: Literal["mergeMean"] = "mergeMean"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Mean"


class MergeMedian(Config):
    name: Literal["mergeMedian"] = "mergeMedian"
    value: Literal["mergeMedian"] = "mergeMedian"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Median"


class MergeMethod(Config):
    """
    How the aligned frames are combined. 'Mean' removes the most noise. 'Median' ignores a passing hand, glare or
    a badly aligned frame, at the cost of a little more noise.
    """

    name: Literal["MergeMethod"] = "MergeMethod"
    value: Union[MergeMean, MergeMedian]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Merge Method"
        json_schema_extra = {"shortDescription": "Merge Method"}


class Upscale1x(Config):
    name: Literal["upscale1x"] = "upscale1x"
    value: Literal[1] = 1
    type: Literal["number"] = "number"
    field: Literal["option"] = "option"

    class Config:
        title = "1x"


class Upscale2x(Config):
    name: Literal["upscale2x"] = "upscale2x"
    value: Literal[2] = 2
    type: Literal["number"] = "number"
    field: Literal["option"] = "option"

    class Config:
        title = "2x"


class Upscale(Config):
    """
    Output size. '2x' merges the aligned frames on a twice finer grid, which keeps more of the detail that falls
    between pixels when the object moves; the output is twice the reference frame size.
    """

    name: Literal["Upscale"] = "Upscale"
    value: Union[Upscale1x, Upscale2x]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Upscale"
        json_schema_extra = {"shortDescription": "Output Upscale"}


class FusionInputs(Inputs):
    inputImages: InputImages


class FusionConfigs(Configs):
    alignment: Alignment
    selection: Selection
    mergeMethod: MergeMethod
    upscale: Upscale


class FusionOutputs(Outputs):
    outputImage: OutputImage
    outputData: OutputData


class FusionRequest(Request):
    inputs: Optional[FusionInputs]
    configs: FusionConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class FusionResponse(Response):
    outputs: FusionOutputs


class FusionExecutor(Config):
    """
    Picks the sharpest of the incoming frames, aligns them to the sharpest one and merges them into one cleaner
    image (lucky imaging). Place it after Stack Image to give barcode or text readers a sharper, less noisy view.
    """

    name: Literal["Fusion"] = "Fusion"
    value: Union[FusionRequest, FusionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Fusion"
        json_schema_extra = {"target": {"value": 0}}


class ConfigExecutor(Config):
    """
    Multi-frame fusion task.
    """

    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[FusionExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"shortDescription": "Fusion Task", "target": "value"}


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["FrameFusion"] = "FrameFusion"
