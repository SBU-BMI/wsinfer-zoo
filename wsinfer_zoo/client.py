"""API to interact with WSInfer model zoo, hosted on HuggingFace."""

import dataclasses
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

import jsonschema
import requests
from huggingface_hub import hf_hub_download

# The name of the configuration JSON file.
HF_CONFIG_NAME = "config.json"
# The name of the torchscript saved file.
HF_TORCHSCRIPT_NAME = "torchscript_model.pt"
# The name of the safetensors file with weights.
HF_WEIGHTS_SAFETENSORS_NAME = "model.safetensors"
# The name of the pytorch (pickle) file with weights.
HF_WEIGHTS_PICKLE_NAME = "pytorch_model.bin"

# URL to the latest model registry.
WSINFER_ZOO_REGISTRY_URL = "https://raw.githubusercontent.com/SBU-BMI/wsinfer-zoo/main/wsinfer-zoo-registry.json"  # noqa
# The path to the registry file.
WSINFER_ZOO_REGISTRY_DEFAULT_PATH = Path.home() / ".wsinfer-zoo-registry.json"

# In pyinstaller runtime for one-file executables, the root path
# is the path to the executable.
if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", False):
    _here = Path(sys._MEIPASS).resolve()  # type: ignore
else:
    _here = Path(__file__).parent.resolve()


class WSInferZooException(Exception):
    ...


class InvalidRegistryConfiguration(WSInferZooException):
    ...


class InvalidModelConfiguration(WSInferZooException):
    ...


def validate_config_json(instance: object):
    """Raise an error if the model configuration JSON is invalid. Otherwise return True."""
    schema_path = _here / "schemas" / "model-config.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"JSON schema for model configurations not found: {schema_path}"
        )
    with open(schema_path) as f:
        schema = json.load(f)
    try:
        jsonschema.validate(instance, schema=schema)
    except jsonschema.ValidationError as e:
        raise InvalidModelConfiguration(
            "Invalid model configuration. See traceback above for details."
        ) from e

    return True


def validate_model_zoo_json(instance: object):
    """Raise an error if the model zoo registry JSON is invalid. Otherwise return True."""
    schema_path = _here / "schemas" / "wsinfer-zoo-registry.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"JSON schema for wsinfer zoo not found: {schema_path}")
    with open(schema_path) as f:
        schema = json.load(f)
    try:
        jsonschema.validate(instance, schema=schema)
    except jsonschema.ValidationError as e:
        raise InvalidRegistryConfiguration(
            "Invalid model zoo registry configuration. See traceback above for details."
        ) from e
    return True


@dataclasses.dataclass
class TransformConfigurationItem:
    """Container for one item in the 'transform' property of the model configuration."""

    name: str
    arguments: Optional[Dict[str, Any]]


@dataclasses.dataclass
class ModelConfiguration:
    """Container for the configuration of a single model.

    This is from the contents of 'config.json'.
    """

    # FIXME: add fields like author, license, training data, publications, etc.
    architecture: str
    num_classes: int
    class_names: Sequence[str]
    patch_size_pixels: int
    spacing_um_px: float
    transform: List[TransformConfigurationItem]

    def __post_init__(self):
        if len(self.class_names) != self.num_classes:
            raise InvalidModelConfiguration()

    @classmethod
    def from_dict(cls, config: Dict) -> "ModelConfiguration":
        validate_config_json(config)
        architecture = config["architecture"]
        num_classes = config["num_classes"]
        patch_size_pixels = config["patch_size_pixels"]
        spacing_um_px = config["spacing_um_px"]
        class_names = config["class_names"]
        transform_list: List[Dict[str, Any]] = config["transform"]
        transform = [
            TransformConfigurationItem(name=t["name"], arguments=t.get("arguments"))
            for t in transform_list
        ]
        return cls(
            architecture=architecture,
            num_classes=num_classes,
            patch_size_pixels=patch_size_pixels,
            spacing_um_px=spacing_um_px,
            class_names=class_names,
            transform=transform,
        )

    def get_slide_patch_size(self, slide_spacing_um_px: float) -> int:
        """Get the size of the patches to extract from the slide to be compatible
        with the patch size and spacing the model expects.

        The model expects images of a particular physical size. This can be calculated
        with spacing_um_px * patch_size_pixels, and the results units are in
        micrometers (um).

        The native spacing of a slide can be different than what the model expects, so
        patches should be extracted at a different size and rescaled to the pixel size
        expected by the model.
        """
        return round(self.patch_size_pixels * self.spacing_um_px / slide_spacing_um_px)


@dataclasses.dataclass
class HFInfo:
    """Container for information on model's location on HuggingFace Hub."""

    repo_id: str
    revision: Optional[str] = None


@dataclasses.dataclass
class Model:
    config: ModelConfiguration
    model_path: str


@dataclasses.dataclass
class HFModel(Model):
    """Container for a model hosted on HuggingFace."""

    hf_info: HFInfo


@dataclasses.dataclass
class HFModelTorchScript(HFModel):
    """Container for the downloaded model path and config."""


# This is here to avoid confusion. We could have used Model directly with
# weights files, but then downstream it would not be clear whether the
# model has torchscript files or weights files.
@dataclasses.dataclass
class HFModelWeightsOnly(HFModel):
    """Container for a model with weights only (not a TorchScript model)."""


def load_torchscript_model_from_hf(
    repo_id: str, revision: Optional[str] = None
) -> HFModelTorchScript:
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
    # FIXME: should we always load on cpu?
    hf_info = HFInfo(repo_id=repo_id, revision=revision)
    model = HFModelTorchScript(config=config, model_path=model_path, hf_info=hf_info)
    return model


def load_weights_from_hf(
    repo_id: str, revision: Optional[str] = None, safetensors: bool = False
) -> HFModelWeightsOnly:
    """Load model weights from HuggingFace (this is not TorchScript)."""
    if safetensors:
        model_path = hf_hub_download(
            repo_id, HF_WEIGHTS_SAFETENSORS_NAME, revision=revision
        )
    else:
        model_path = hf_hub_download(repo_id, HF_WEIGHTS_PICKLE_NAME, revision=revision)

    config_path = hf_hub_download(repo_id, HF_CONFIG_NAME, revision=revision)
    with open(config_path) as f:
        config_dict = json.load(f)
    if not isinstance(config_dict, dict):
        raise TypeError(
            f"Expected configuration to be a dict but got {type(config_dict)}"
        )
    config = ModelConfiguration.from_dict(config_dict)
    hf_info = HFInfo(repo_id=repo_id, revision=revision)
    model = HFModelWeightsOnly(config=config, model_path=model_path, hf_info=hf_info)
    return model


@dataclasses.dataclass
class RegisteredModel:
    """Container with information about where to find a single model."""

    name: str
    description: str
    hf_repo_id: str
    hf_revision: str

    def load_model_torchscript(self) -> HFModelTorchScript:
        return load_torchscript_model_from_hf(
            repo_id=self.hf_repo_id, revision=self.hf_revision
        )

    def load_model_weights(self, safetensors: bool = False) -> HFModelWeightsOnly:
        return load_weights_from_hf(
            repo_id=self.hf_repo_id, revision=self.hf_revision, safetensors=safetensors
        )

    def __str__(self) -> str:
        return (
            f"{self.name} -> {self.description} ({self.hf_repo_id}"
            f" @ {self.hf_revision})"
        )


@dataclasses.dataclass
class ModelRegistry:
    """Registry of models that can be used with WSInfer."""

    models: Dict[str, RegisteredModel]

    def get_model_by_name(self, name: str) -> RegisteredModel:
        try:
            return self.models[name]
        except KeyError:
            raise KeyError(f"model not found with name '{name}'.")

    @classmethod
    def from_dict(cls, config: Dict) -> "ModelRegistry":
        """Create a new ModelRegistry instance from a dictionary."""
        validate_model_zoo_json(config)
        models = {
            name: RegisteredModel(
                name=name,
                description=kwds["description"],
                hf_repo_id=kwds["hf_repo_id"],
                hf_revision=kwds["hf_revision"],
            )
            for name, kwds in config["models"].items()
        }

        return cls(models=models)


def _remote_registry_is_newer() -> bool:
    """Return `True` if the remote registry is newer than local, `False` otherwise."""
    # Local file does not exist, so we should download it.
    if not WSINFER_ZOO_REGISTRY_DEFAULT_PATH.exists():
        return True

    url = "https://api.github.com/repos/SBU-BMI/wsinfer-zoo/commits?path=wsinfer-zoo-registry.json&page=1&per_page=1"  # noqa
    resp = requests.get(url)
    if not resp.ok:
        raise requests.RequestException(
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
            r = requests.get(WSINFER_ZOO_REGISTRY_URL)
            WSINFER_ZOO_REGISTRY_DEFAULT_PATH.write_bytes(r.content)
    except requests.RequestException as e:
        print(f"Could not download most recent registry, error: {e}")
