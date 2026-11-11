"""The HomeLens API.

Start it from the project folder:
    uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health     is the server up, which mode, which model
    GET  /cities     supported cities and what kind of estimate each gets
    POST /resolve    address or map position -> candidate properties to confirm
    POST /estimate   confirmed property -> the four estimates
"""
import asyncio
import hashlib
import logging
import time
from datetime import date
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app import schemas
from backend.app.cities import Locator, zip_from_address
from backend.app.config import Settings, load_settings
from backend.app.estimator import Estimator
from backend.app.geocode import GeocodeError, GeocoderChain, build_geocoder
from backend.app.ratelimit import RateLimiter
from core import furniture
from core.geo import load_cities
from core.imagery import get_provider
from core.paths import ROOT

log = logging.getLogger("homelens")
SUPPORTED_TYPE = "detached_single_family"


def error_response(status_code: int, code: str, message: str, fields: Optional[List[Dict]] = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "fields": fields or []}}
    return JSONResponse(status_code=status_code, content=body)


def _candidate_id(lat: float, lng: float, address: Optional[str]) -> str:
    text = "%.6f,%.6f,%s" % (lat, lng, address or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def create_app(settings: Optional[Settings] = None, estimator: Optional[Estimator] = None,
               locator: Optional[Locator] = None, geocoder: Optional[GeocoderChain] = None) -> FastAPI:
    """Build the app. Tests pass in their own parts; production uses the defaults."""
    if settings is None:
        load_dotenv(ROOT / ".env")
        settings = load_settings()
    cities = load_cities()
    if locator is None:
        locator = Locator.from_files(settings.zip_table_path, cities, settings.parcels_path)
    if estimator is None:
        estimator = Estimator.from_dir(settings.model_dir, get_provider(settings.mapillary_token),
                                       furniture.load_package(), settings.demo)
    if geocoder is None:
        geocoder = build_geocoder(settings.geocoders, settings.geocode_timeout_s)
    limiter = RateLimiter(settings.requests_per_minute)

    app = FastAPI(title="HomeLens API", version="1.0.0",
                  description="Estimates for detached single-family homes in Seattle, San Francisco and Portland.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.extra_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
        max_age=600,
    )

    # ---------------------------------------------------------------- guards

    @app.middleware("http")
    async def guard(request: Request, call_next):
        started = time.monotonic()
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is None or not length.isdigit() or int(length) > settings.max_body_bytes:
                return error_response(413, "body_too_large", "The request body is missing a length or is too large.")
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(client):
                return error_response(429, "rate_limited", "Too many requests. Wait a minute and try again.")
        response = await call_next(request)
        # Only the path, status and time are logged. Never addresses, coordinates or bodies.
        log.info("%s %s -> %d in %d ms", request.method, request.url.path, response.status_code,
                 int((time.monotonic() - started) * 1000))
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        fields = []
        for err in exc.errors():
            where = ".".join(str(p) for p in err.get("loc", []) if p != "body") or "body"
            fields.append({"field": where, "problem": str(err.get("msg", "invalid"))})
        return error_response(422, "invalid_request", "The request is missing something or has a bad value.", fields)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        code = "not_found" if exc.status_code == 404 else "http_error"
        return error_response(exc.status_code, code, "Not found." if exc.status_code == 404 else "Request failed.")

    @app.exception_handler(Exception)
    async def on_unexpected_error(request: Request, exc: Exception):
        log.error("unexpected error on %s: %s", request.url.path, type(exc).__name__)
        return error_response(500, "internal_error", "Something went wrong on the server.")

    # ---------------------------------------------------------------- simple endpoints

    @app.get("/health", response_model=schemas.HealthResponse)
    def health():
        return {
            "status": "ok",
            "mode": "demo" if settings.demo else "real",
            "models_loaded": estimator.loaded,
            "model_version": estimator.model_version,
            "imagery_provider": estimator.provider.name,
            "imagery_configured": estimator.provider.configured,
            "photo_model_served": any(estimator.photo_model_served(c) for c in cities),
        }

    @app.get("/cities", response_model=schemas.CitiesResponse)
    def list_cities():
        out = []
        for c in cities:
            out.append({
                "id": c["id"], "name": c["name"], "state": c["state"],
                "sale_basis": "per_house_model" if estimator.has_per_house_model(c) else "area_level_model",
                "rent_basis": "area_level_model",
                "photo_model_served": estimator.photo_model_served(c),
            })
        return {"cities": out, "supported_property_type": SUPPORTED_TYPE,
                "note": "Only detached single-family homes inside these cities are supported."}

    # ---------------------------------------------------------------- resolve

    def _make_candidate(lat: float, lng: float, address: Optional[str], match: str, zip_hint: Optional[str],
                        property_id: Optional[str] = None, distance_m: Optional[float] = None, note: str = "") -> Dict:
        place = locator.locate(lat, lng, zip_hint)
        city = place["city"]
        return {
            "candidate_id": _candidate_id(lat, lng, address),
            "address": address, "lat": lat, "lng": lng, "zip": place["zip"],
            "city_id": city["id"] if city else None,
            "city_name": city["name"] if city else place["place_name"],
            "supported": city is not None, "match": match, "property_id": property_id,
            "distance_m": distance_m, "note": note,
        }

    def _resolve(req: schemas.ResolveRequest) -> Dict:
        lat, lng, address = req.lat, req.lng, req.address
        zip_hint = zip_from_address(address)
        geocoded = False
        geocode_source = ""
        if lat is None:
            # Address only: ask an address lookup service where it is.
            try:
                found = geocoder.lookup(address)
            except GeocodeError as exc:
                return {"status": "not_found", "candidates": [],
                        "message": "The address lookup service did not answer (%s). Try again, or paste a map pin." % exc}
            if found is None:
                return {"status": "not_found", "candidates": [],
                        "message": "That address was not found. Check the spelling and include the city and state."}
            lat, lng, address, geocoded, geocode_source = found.lat, found.lon, found.address, True, found.source
            # A ZIP the user typed is trusted over the lookup service's ZIP.
            zip_hint = zip_hint or found.zip
            if zip_hint and zip_hint not in address:
                address = "%s %s" % (address, zip_hint)

        candidates: List[Dict] = []
        if address:
            parcel = locator.parcel_for_address(lat, lng, address) if locator.has_parcels else None
            if parcel:
                candidates.append(_make_candidate(parcel["lat"], parcel["lon"], parcel["address"], "parcel", parcel["zip"],
                                                  parcel["property_id"], parcel["distance_m"],
                                                  "Matched to a single-family parcel in the county records."))
            else:
                note = ("Position from %s. It may be a few houses off." % geocode_source
                        if geocoded else "Position comes from the selected place on the map.")
                candidates.append(_make_candidate(lat, lng, address, "geocoder" if geocoded else "place", zip_hint, note=note))
            status = "ok"
            message = "Check that this is the house you mean."
        else:
            # Position only (for example a Street View camera position): never assume which house it is.
            for parcel in locator.nearest_parcels(lat, lng, count=3):
                candidates.append(_make_candidate(parcel["lat"], parcel["lon"], parcel["address"], "parcel", parcel["zip"],
                                                  parcel["property_id"], parcel["distance_m"]))
            if candidates:
                message = "This map position is not tied to one house. Pick the house you mean, or type its address."
            else:
                candidates.append(_make_candidate(lat, lng, None, "pin", None,
                                                  note="A bare map position. It may be the camera or the street, not the house."))
                message = "This map position is not tied to one house. Type the address, or confirm the pin is on the house."
            status = "needs_confirmation"

        if not any(c["supported"] for c in candidates):
            status = "unsupported_location"
            where = candidates[0]["city_name"] or "This location"
            message = "%s is outside the supported cities (Seattle, San Francisco, Portland)." % where
        return {"status": status, "candidates": candidates, "message": message}

    @app.post("/resolve", response_model=schemas.ResolveResponse,
              responses={422: {"model": schemas.ErrorResponse}, 429: {"model": schemas.ErrorResponse}})
    def resolve(req: schemas.ResolveRequest):
        return _resolve(req)

    # ---------------------------------------------------------------- estimate

    def _empty_estimate(req: schemas.EstimateRequest, status: str, message: str, place: Dict) -> Dict:
        city = place["city"]
        return {
            "status": status, "message": message, "client_request_id": req.client_request_id,
            "property": {"address": req.address, "lat": req.lat, "lng": req.lng, "zip": place["zip"],
                         "city_id": city["id"] if city else None,
                         "city_name": city["name"] if city else place["place_name"],
                         "property_type": req.property_type},
            "prediction_mode": "none", "result_kind": "none", "is_demo": settings.demo, "estimates": [],
            "image": {"status": "not_requested", "message": "No estimate was made."},
            "furniture": None, "model_version": estimator.model_version,
            "data_period": {"sale": None, "rent": None}, "estimate_date": date.today().isoformat(),
            "limitations": [], "attributions": [],
        }

    def _estimate(req: schemas.EstimateRequest) -> Dict:
        place = locator.locate(req.lat, req.lng, req.zip or zip_from_address(req.address))
        city = place["city"]
        if city is None:
            where = place["place_name"] or "This location"
            return _empty_estimate(req, "unsupported_location",
                                   "%s is outside the supported cities (Seattle, San Francisco, Portland)." % where, place)
        if req.property_type != SUPPORTED_TYPE:
            return _empty_estimate(req, "unsupported_property",
                                   "HomeLens only supports detached single-family homes.", place)
        if not estimator.loaded:
            return _empty_estimate(req, "insufficient_data",
                                   "No trained models were found on the server. Train them or start the server in demo mode.", place)
        result = estimator.estimate(req.lat, req.lng, city, place["metro_id"], use_image=req.use_image)
        out = _empty_estimate(req, "location_only", "", place)
        if not result["any_ok"]:
            out["status"] = "insufficient_data"
            out["message"] = "The models could not produce a usable estimate for this location."
        elif result["prediction_mode"] == "location_plus_image":
            out["status"] = "estimate_ready"
            out["message"] = "Estimate made from location and a licensed street photo."
        else:
            out["message"] = "Location-only estimate. The visible exterior was not analyzed."
        out.update({
            "prediction_mode": result["prediction_mode"],
            "result_kind": "demo" if settings.demo else ("trained" if result["any_ok"] else "none"),
            "estimates": result["estimates"], "image": result["image"], "furniture": result["furniture"],
            "data_period": result["data_period"], "estimate_date": result["estimate_date"],
            "limitations": result["limitations"], "attributions": result["attributions"],
        })
        log.info("estimate city=%s status=%s mode=%s", city["id"], out["status"], out["prediction_mode"])
        return out

    @app.post("/estimate", response_model=schemas.EstimateResponse,
              responses={422: {"model": schemas.ErrorResponse}, 429: {"model": schemas.ErrorResponse},
                         504: {"model": schemas.ErrorResponse}})
    async def estimate(req: schemas.EstimateRequest):
        try:
            return await asyncio.wait_for(run_in_threadpool(_estimate, req), timeout=settings.estimate_timeout_s)
        except asyncio.TimeoutError:
            return error_response(504, "timeout", "The estimate took too long. Try again.")

    return app


class _LazyApp:
    """Builds the real app on first use, so importing this module in tests stays cheap."""

    def __init__(self):
        self._app = None

    async def __call__(self, scope, receive, send):
        if self._app is None:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
            self._app = create_app()
        await self._app(scope, receive, send)


app = _LazyApp()
