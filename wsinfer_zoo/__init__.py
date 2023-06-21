import os as _os

import json
import jsonschema
from wsinfer_zoo import _version
from wsinfer_zoo.client import (
    _download_registry_if_necessary,
    ModelRegistry,
    WSINFER_ZOO_REGISTRY_DEFAULT_PATH,
    InvalidRegistryConfiguration,
    validate_model_zoo_json,
)

__version__ = _version.get_versions()["version"]

if _os.getenv("WSINFER_ZOO_NO_UPDATE_REGISTRY") is None:
    _download_registry_if_necessary()


if not WSINFER_ZOO_REGISTRY_DEFAULT_PATH.exists():
    raise FileNotFoundError(
        f"registry file not found: {WSINFER_ZOO_REGISTRY_DEFAULT_PATH}"
    )
with open(WSINFER_ZOO_REGISTRY_DEFAULT_PATH) as f:
    d = json.load(f)
try:
    validate_model_zoo_json(d)
except InvalidRegistryConfiguration as e:
    raise InvalidRegistryConfiguration(
        "Registry schema is invalid. Please contact the developer by"
        " creating a new issue on our GitHub page:"
        " https://github.com/SBU-BMI/wsinfer-zoo/issues/new."
    ) from e

registry = ModelRegistry.from_dict(d)

del d, json, jsonschema
