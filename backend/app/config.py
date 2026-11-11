"""Backend settings, read from environment variables (and the local .env file)."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional

from core.paths import DEMO_MODEL_DIR, FIXTURES_DIR, PROCESSED_DIR, REAL_MODEL_DIR

EXTENSION_ID_PATTERN = re.compile(r"^[a-p]{32}$")


@dataclass
class Settings:
    demo: bool = False
    extension_id: str = ""
    extra_origins: List[str] = field(default_factory=list)
    mapillary_token: str = ""
    model_dir: Path = REAL_MODEL_DIR
    zip_table_path: Path = PROCESSED_DIR / "zip_values.csv"
    parcels_path: Optional[Path] = PROCESSED_DIR / "seattle_parcels.csv"
    requests_per_minute: int = 30
    geocoders: List[str] = field(default_factory=lambda: ["nominatim", "census"])
    geocode_timeout_s: float = 6.0
    estimate_timeout_s: float = 25.0
    max_body_bytes: int = 10000

    @property
    def cors_origin_regex(self) -> str:
        """Only Chrome extension pages may call the API from a browser."""
        if self.extension_id:
            return r"^chrome-extension://%s$" % re.escape(self.extension_id)
        return r"^chrome-extension://[a-p]{32}$"


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    env = os.environ if env is None else env
    demo = _truthy(env.get("HOMELENS_DEMO"))
    extension_id = (env.get("HOMELENS_EXTENSION_ID") or "").strip()
    if extension_id and not EXTENSION_ID_PATTERN.match(extension_id):
        raise ValueError("HOMELENS_EXTENSION_ID must be the 32 letter id shown on chrome://extensions")
    extra = [o.strip() for o in (env.get("HOMELENS_EXTRA_ORIGINS") or "").split(",") if o.strip()]
    for origin in extra:
        if origin == "*" or not origin.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("HOMELENS_EXTRA_ORIGINS may only list local preview origins, got %r" % origin)
    geocoders = [g.strip().lower() for g in (env.get("HOMELENS_GEOCODERS") or "nominatim,census").split(",") if g.strip()]
    for name in geocoders:
        if name not in ("nominatim", "census"):
            raise ValueError("HOMELENS_GEOCODERS may only list nominatim and census, got %r" % name)
    settings = Settings(
        demo=demo,
        geocoders=geocoders,
        extension_id=extension_id,
        extra_origins=extra,
        mapillary_token=(env.get("MAPILLARY_TOKEN") or "").strip(),
    )
    if demo:
        settings.model_dir = DEMO_MODEL_DIR
        settings.zip_table_path = FIXTURES_DIR / "synthetic_zip_values.csv"
        settings.parcels_path = None
    return settings
