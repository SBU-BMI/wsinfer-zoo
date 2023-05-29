"""API to interact with WSInfer model zoo, hosted on HuggingFace."""

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import urllib.error
import urllib.request

import requests
from huggingface_hub import hf_hub_download

# TODO: we might consider fetching available models from the web.
# from huggingface_hub import HfApi
# hf_api = HfApi()
# models = hf_api.list_models(author="kaczmarj")
# print("Found these models...")
# print(models)

# FIXME: consider changing the name of this file because perhaps there will
# be multiple configs? Or add a key inside the json map 'wsinfer_config'.
HF_CONFIG_NAME = "config.json"
# HF_WEIGHTS_NAME = "pytorch_model.bin"
HF_TORCHSCRIPT_NAME = "torchscript_model_frozen.bin"

# URL to the latest model registry.
WSINFER_ZOO_REGISTRY_URL = "https://raw.githubusercontent.com/SBU-BMI/wsinfer-zoo/main/wsinfer-zoo-registry.json"
# The path to the registry file.
WSINFER_ZOO_REGISTRY_DEFAULT_PATH = Path.home() / ".wsinfer-zoo-registry.json"


@dataclasses.dataclass
class TransformConfiguration:
    """Container for the transform configuration for a model.

    This is stored in te 'transform' key of 'config.json'.
    """

    resize_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


@dataclasses.dataclass
class ModelConfiguration:
    """Container for the configuration of a single model.

    This is from the contents of 'config.json'.
    """

    # FIXME: add fields like author, license, training data, publications, etc.
    num_classes: int
    class_names: Sequence[str]
    patch_size_pixels: int
    spacing_um_px: float
    transform: TransformConfiguration

    @classmethod
    def from_dict(cls, config: Dict) -> "ModelConfiguration":
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


@dataclasses.dataclass
class HFInfo:
    repo_id: str
    revision: Optional[str] = None


@dataclasses.dataclass
class Model:
    """Container for the downloaded model path and config."""

    config: ModelConfiguration
    model_path: str
    hf_info: HFInfo


def load_torchscript_model_from_hf(
    repo_id: str, revision: Optional[str] = None
) -> Model:
    """Load a TorchScript model from HuggingFace."""
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


@dataclasses.dataclass
class RegisteredModel:
    """Container with information about where to find a single model."""

    name: str
    hf_repo_id: str
    hf_revision: Optional[str]


@dataclasses.dataclass
class ModelRegistry:
    """Registry of models that can be used with WSInfer."""

    models: List[RegisteredModel]

    @classmethod
    def from_dict(cls, config: Dict) -> "ModelRegistry":
        assert isinstance(config, dict)
        assert "models" in config.keys()
        assert isinstance(config["models"], list)
        assert config["models"]

        # Test that all model items have required keys.
        for cm in config["models"]:
            for key in ["name", "hf_repo_id"]:
                if key not in cm.keys():
                    raise KeyError(f"required key '{key}' not found in model info")

        # Test if there are any duplicate model names.
        uniq_names = set(cm["name"] for cm in config["models"])
        if len(uniq_names) != len(config["models"]):
            raise ValueError("there are non-unique 'name' values in the model registry")

        models = [
            RegisteredModel(
                name=cm["name"],
                hf_repo_id=cm["hf_repo_id"],
                hf_revision=cm.get("hf_revision"),
            )
            for cm in config["models"]
        ]

        return cls(models=models)

    @property
    def model_names(self) -> List[str]:
        return [m.name for m in self.models]

    @property
    def model_names_to_info(self) -> Dict[str, RegisteredModel]:
        return {m.name: m for m in self.models}

    def load_model_from_name(self, name: str) -> Model:
        try:
            model_info = self.model_names_to_info[name]
        except KeyError:
            raise KeyError(
                f"Unknown model name '{name}', available models are {self.model_names}."
            )
        return load_torchscript_model_from_hf(
            repo_id=model_info.hf_repo_id, revision=model_info.hf_revision
        )


def _remote_registry_is_newer() -> bool:
    """Return `True` if the remote registry is newer than local, `False` otherwise."""
    # Local file does not exist, so we should download it.
    if not WSINFER_ZOO_REGISTRY_DEFAULT_PATH.exists():
        return True

    url = "https://api.github.com/repos/SBU-BMI/wsinfer-zoo/commits?path=wsinfer-zoo-registry.json&page=1&per_page=1"
    resp = requests.get(url)
    if not resp.ok:
        raise RuntimeError(
            "could not get the last updated time of the remote model registry file"
        )
    remote_commit_date = resp.json()[0]["commit"]["committer"]["date"]
    if remote_commit_date.endswith("Z"):
        remote_commit_date = remote_commit_date[:-1] + "+00:00"
    remote_mtime = datetime.fromisoformat(remote_commit_date)

    # Defaults to local time zone.
    local_mtime = datetime.fromtimestamp(
        WSINFER_ZOO_REGISTRY_DEFAULT_PATH.stat().st_mtime
    ).astimezone()

    remote_is_newer = remote_mtime > local_mtime
    return remote_is_newer


def _download_registry_if_necessary():
    """Download the wsinfer zoo registry from the web if
    1. a local version is not found,
    2. the web version is newer than local.
    """
    try:
        if _remote_registry_is_newer():
            urllib.request.urlretrieve(
                WSINFER_ZOO_REGISTRY_URL, WSINFER_ZOO_REGISTRY_DEFAULT_PATH
            )
    except (requests.RequestException, urllib.error.URLError) as e:
        print(f"Could not download most recent registry, error: {e}")
