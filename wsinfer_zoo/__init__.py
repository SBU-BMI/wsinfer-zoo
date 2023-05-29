import os as _os

from wsinfer_zoo import _version
from wsinfer_zoo.client import _download_registry_if_necessary

__version__ = _version.get_versions()["version"]

if _os.getenv("WSINFER_ZOO_NO_UPDATE_REGISTRY") is None:
    _download_registry_if_necessary()
