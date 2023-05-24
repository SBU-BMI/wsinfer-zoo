"""API to get various models.

How should we have a list of available models? Should they be hardcoded here?
Maybe. If not hardcoded here, they should be fetched from the web. Or they can
be specified in a JSON sidecar.
"""

import dataclasses
import json
from typing import Dict, List, Optional, Sequence, Tuple

from huggingface_hub import hf_hub_download

# TODO: we might consider fetching available models from the web.
# from huggingface_hub import HfApi
# hf_api = HfApi()
# models = hf_api.list_models(author="kaczmarj")
# print("Found these models...")
# print(models)

HF_CONFIG_NAME = "config.json"
# HF_WEIGHTS_NAME = "pytorch_model.bin"
HF_TORCHSCRIPT_NAME = "torchscript_model_frozen.bin"

# TODO: consider adding the training set to the model name.
# TODO: where should this state be stored? Here should be fine...
NAME_TO_HF_ID = {
    "Breast tumor": "kaczmarj/breast-tumor-resnet34",
    "Pancancer lymphocytes": "kaczmarj/pancancer-lymphocytes-inceptionv4",
}


@dataclasses.dataclass(frozen=True)
class TransformConfiguration:
    resize_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class ModelConfiguration:
    # FIXME: add fields like author, license, training data, publications, etc.
    num_classes: int
    class_names: Sequence[str]
    patch_size_pixels: int
    spacing_um_px: float
    transform: TransformConfiguration

    def get_transform(self):
        raise NotImplementedError()

    def get_model(self):
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, config: Dict):
        # TODO: add validation here...
        num_classes = config["num_classes"]
        patch_size_pixels = config["patch_size_pixels"]
        spacing_um_px = config["spacing_um_px"]
        class_names = config["class_names"]
        tdict = config["transform"]
        transform = TransformConfiguration(
            resize_size=tdict["resize_size"], mean=tdict["mean"], std=tdict["std"]
        )
        return cls(
            num_classes=num_classes,
            patch_size_pixels=patch_size_pixels,
            spacing_um_px=spacing_um_px,
            class_names=class_names,
            transform=transform,
        )

    def get_slide_patch_size(self, slide_spacing_um_px: float) -> int:
        """Get the size of the patches to extract from the slide to be compatible
        with the patch size and spacing the model expects.

        The model expects images of a particular physical size. This can be calculated with
        spacing_um_px * patch_size_pixels, and the results units are in micrometers (um).

        The native spacing of a slide can be different than what the model expects, so
        patches should be extracted at a different size and rescaled to the pixel size
        expected by the model.
        """
        return round(self.patch_size_pixels * self.spacing_um_px / slide_spacing_um_px)


@dataclasses.dataclass(frozen=True)
class HFInfo:
    repo_id: str
    revision: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Model:
    config: ModelConfiguration
    model_path: str
    hf_info: HFInfo


def load_frozen_model_from_hf(repo_id: str, revision: Optional[str] = None) -> Model:
    """Load a frozen TorchScript model from HuggingFace."""
    model_path = hf_hub_download(repo_id, HF_TORCHSCRIPT_NAME, revision=revision)

    config_path = hf_hub_download(repo_id, HF_CONFIG_NAME, revision=revision)
    with open(config_path) as f:
        config_dict = json.load(f)
    if not isinstance(config_dict, dict):
        raise TypeError(
            f"Expected configuration to be a dict but got {type(config_dict)}"
        )
    config = ModelConfiguration.from_dict(config_dict)
    del config_dict
    # FIXME: should we always load on cpu?
    hf_info = HFInfo(repo_id=repo_id, revision=revision)
    model = Model(config=config, model_path=model_path, hf_info=hf_info)
    return model


def list_model_names() -> List[str]:
    return sorted(NAME_TO_HF_ID.keys())


def get_models_and_hf_repos() -> Dict[str, str]:
    return NAME_TO_HF_ID.copy()


def load_model_from_name(name: str, revision: Optional[str] = None) -> Model:
    try:
        repo_id = NAME_TO_HF_ID[name]
    except KeyError:
        raise KeyError(f"Unknown name '{name}'")
    return load_frozen_model_from_hf(repo_id=repo_id, revision=revision)
