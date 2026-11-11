"""The typed API contract. FastAPI turns these classes into the OpenAPI document at /docs."""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

PropertyType = Literal["detached_single_family", "other", "unknown"]
ResolveSource = Literal["place", "street_view", "map_center", "manual"]
EstimateStatus = Literal[
    "estimate_ready",         # estimate made with location and an exterior photo
    "location_only",          # estimate made from location alone
    "unsupported_location",
    "unsupported_property",
    "insufficient_data",
]
ImageStatus = Literal[
    "used",                   # a licensed photo was found and used by the model
    "unavailable",            # no usable licensed photo near this house
    "not_requested",          # the user asked for location only
    "not_configured",         # the server has no imagery provider token
    "no_photo_model",         # no trained photo model is served for this city
    "error",                  # the provider failed; location only was used instead
]
EstimateKind = Literal["trained", "scenario", "demo"]
EstimateBasis = Literal["per_house_model", "area_level_model", "assumption_added"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ------------------------------------------------------------------ errors

class FieldProblem(BaseModel):
    field: str
    problem: str


class ErrorBody(BaseModel):
    code: str
    message: str
    fields: List[FieldProblem] = []


class ErrorResponse(BaseModel):
    error: ErrorBody


# ------------------------------------------------------------------ health and cities

class HealthResponse(BaseModel):
    status: Literal["ok"]
    mode: Literal["real", "demo"]
    models_loaded: bool
    model_version: Optional[str]
    imagery_provider: str
    imagery_configured: bool
    photo_model_served: bool


class CityInfo(BaseModel):
    id: str
    name: str
    state: str
    sale_basis: EstimateBasis
    rent_basis: EstimateBasis
    photo_model_served: bool


class CitiesResponse(BaseModel):
    cities: List[CityInfo]
    supported_property_type: str
    note: str


# ------------------------------------------------------------------ resolve

class ResolveRequest(StrictModel):
    address: Optional[str] = Field(default=None, min_length=3, max_length=200)
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lng: Optional[float] = Field(default=None, ge=-180, le=180)
    source: ResolveSource = "manual"

    @model_validator(mode="after")
    def need_something(self):
        if (self.lat is None) != (self.lng is None):
            raise ValueError("lat and lng must be given together")
        if self.address is None and self.lat is None:
            raise ValueError("give an address, or lat and lng")
        return self


class Candidate(BaseModel):
    candidate_id: str
    address: Optional[str]
    lat: float
    lng: float
    zip: Optional[str]
    city_id: Optional[str]
    city_name: Optional[str]
    supported: bool
    match: Literal["parcel", "place", "geocoder", "pin"]
    property_id: Optional[str] = None
    distance_m: Optional[float] = None
    note: str = ""


class ResolveResponse(BaseModel):
    status: Literal["ok", "needs_confirmation", "unsupported_location", "not_found"]
    candidates: List[Candidate]
    message: str


# ------------------------------------------------------------------ estimate

class EstimateRequest(StrictModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    address: Optional[str] = Field(default=None, max_length=200)
    zip: Optional[str] = Field(default=None, pattern=r"^\d{5}$")
    property_id: Optional[str] = Field(default=None, max_length=40)
    property_type: PropertyType
    use_image: bool = True
    client_request_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")


class Interval(BaseModel):
    low_usd: int
    high_usd: int
    target_coverage: float
    measured_coverage: float
    method: str


class Estimate(BaseModel):
    key: Literal["buy_unfurnished", "buy_furnished", "rent_unfurnished", "rent_furnished"]
    label: str
    status: Literal["ok", "unavailable"]
    amount_usd: Optional[int]
    unit: Literal["USD", "USD/month"]
    kind: EstimateKind
    basis: EstimateBasis
    interval: Optional[Interval] = None
    interval_note: str
    explanation: str


class ImageInfo(BaseModel):
    status: ImageStatus
    message: str
    provider: Optional[str] = None
    image_id: Optional[str] = None
    thumb_url: Optional[str] = None
    page_url: Optional[str] = None
    creator: Optional[str] = None
    captured_at: Optional[str] = None
    license: Optional[str] = None
    distance_m: Optional[float] = None


class FurnitureItem(BaseModel):
    name: str
    quantity: int


class FurnitureInfo(BaseModel):
    package_name: str
    package_version: str
    condition: str
    valuation_basis: str
    fixed_inventory_note: str
    items: List[FurnitureItem]
    new_cost_usd: float
    resale_fraction: float
    resale_value_usd: float
    rent_spread_months: int
    monthly_amount_usd: float
    rent_basis: str
    source: Literal["assumption"]


class PropertyInfo(BaseModel):
    address: Optional[str]
    lat: float
    lng: float
    zip: Optional[str]
    city_id: Optional[str]
    city_name: Optional[str]
    property_type: PropertyType


class DataPeriod(BaseModel):
    sale: Optional[str]
    rent: Optional[str]


class EstimateResponse(BaseModel):
    status: EstimateStatus
    message: str
    client_request_id: str
    property: PropertyInfo
    prediction_mode: Literal["location_plus_image", "location_only", "none"]
    result_kind: Literal["trained", "demo", "none"]
    is_demo: bool
    estimates: List[Estimate]
    image: ImageInfo
    furniture: Optional[FurnitureInfo]
    model_version: Optional[str]
    data_period: DataPeriod
    estimate_date: str
    limitations: List[str]
    attributions: List[str]
