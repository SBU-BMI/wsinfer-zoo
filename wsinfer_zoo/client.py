"""API to interact with WSInfer model zoo, hosted on HuggingFace."""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
from pathlib import Path
from typing import Any
from typing import Sequence

import jsonschema
from huggingface_hub import hf_hub_download

# The name of the configuration JSON file.
HF_CONFIG_NAME = "config.json"
# The name of the torchscript saved file.
HF_TORCHSCRIPT_NAME = "torchscript_model.pt"
# The name of the safetensors file with weights.
HF_WEIGHTS_SAFETENSORS_NAME = "model.safetensors"
# The name of the pytorch (pickle) file with weights.
HF_WEIGHTS_PICKLE_NAME = "pytorch_model.bin"

# The path to the registry file.
WSINFER_ZOO_REGISTRY_DEFAULT_PATH = (
    Path.home() / ".wsinfer-zoo" / "wsinfer-zoo-registry.json"
)
WSINFER_ZOO_REGISTRY_DEFAULT_PATH.parent.mkdir(exist_ok=True)

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
    """Raise an error if the model configuration JSON is invalid. Otherwise return
    True.
    """
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
    """Raise an error if the model zoo registry JSON is invalid. Otherwise return
    True.
    """
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
    arguments: dict[str, Any] | None


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
    transform: list[TransformConfigurationItem]

    def __post_init__(self):
        if len(self.class_names) != self.num_classes:
            raise InvalidModelConfiguration()

    @classmethod
    def from_dict(cls, config: dict) -> ModelConfiguration:
        validate_config_json(config)
        architecture = config["architecture"]
        num_classes = config["num_classes"]
        patch_size_pixels = config["patch_size_pixels"]
        spacing_um_px = config["spacing_um_px"]
        class_names = config["class_names"]
        transform_list: list[dict[str, Any]] = config["transform"]
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
    revision: str | None = None


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
    repo_id: str, revision: str | None = None
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
    repo_id: str, revision: str | None = None, safetensors: bool = False
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

    models: dict[str, RegisteredModel]

    def get_model_by_name(self, name: str) -> RegisteredModel:
        try:
            return self.models[name]
        except KeyError:
            raise KeyError(f"model not found with name '{name}'.")

    @classmethod
    def from_dict(cls, config: dict) -> ModelRegistry:
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


@functools.lru_cache()
def load_registry(registry_file: str | Path | None = None) -> ModelRegistry:
    """Load model registry.

    This downloads the registry JSON file to a cache and reuses it if
    the remote file is the same as the cached file.

    If registry_file is not None, it should be a path to a JSON file. This will be
    preferred over the remote registry file on HuggingFace.
    """
    if registry_file is None:
        path = hf_hub_download(
            repo_id="kaczmarj/wsinfer-model-zoo-json",
            filename="wsinfer-zoo-registry.json",
            revision="main",
            repo_type="dataset",
            local_dir=WSINFER_ZOO_REGISTRY_DEFAULT_PATH.parent,
        )
        if not Path(WSINFER_ZOO_REGISTRY_DEFAULT_PATH).exists():
            raise FileNotFoundError(
                "Expected registry to be saved to"
                f" {WSINFER_ZOO_REGISTRY_DEFAULT_PATH} but was saved instead to {path}"
            )
    else:
        if not Path(registry_file).exists():
            raise FileNotFoundError(f"registry file not found at {registry_file}")
        path = registry_file

    with open(path) as f:
        registry = json.load(f)
    try:
        validate_model_zoo_json(registry)
    except InvalidRegistryConfiguration as e:
        raise InvalidRegistryConfiguration(
            "Registry schema is invalid. Please contact the developer by"
            " creating a new issue on our GitHub page:"
            " https://github.com/SBU-BMI/wsinfer-zoo/issues/new."
        ) from e

    return ModelRegistry.from_dict(registry)
