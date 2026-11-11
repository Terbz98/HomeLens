"""The provider used when no licensed imagery source is configured."""
from typing import List

from core.imagery.base import ExteriorImage, ImageryError, ImageryProvider


class NoImageryProvider(ImageryProvider):
    """Always reports that no exterior photo is available."""

    name = "none"
    license = ""

    @property
    def configured(self) -> bool:
        return False

    def search(self, lat: float, lon: float, radius_m: float) -> List[ExteriorImage]:
        return []

    def download(self, image: ExteriorImage) -> bytes:
        raise ImageryError("no imagery provider is configured")
