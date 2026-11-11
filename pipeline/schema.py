"""The import schema for housing records. Documented in docs/data_schema.md.

One row is one priced event for one property: a completed sale, an asking
price, or a rental listing. Area averages do not belong in this table.
"""

# Columns every row must have.
REQUIRED_COLUMNS = [
    "property_id",       # stable ID for the house (for example a parcel number)
    "address",           # street address
    "zip",               # 5 digit ZIP code
    "city_id",           # one of the ids in config/cities.json
    "lat",               # latitude of the house
    "lon",               # longitude of the house
    "property_type",     # detached_single_family is the only supported value
    "record_date",       # YYYY-MM-DD: sale date or listing date
    "target_type",       # sale or rent
    "price_usd",         # sale price, or rent for one month
    "currency",          # must be USD
    "price_unit",        # total for sales, per_month for rents
    "price_basis",       # completed_transaction or asking
    "source",            # where the row came from
    "synthetic",         # true only for made-up fixture rows
]

# Columns that may be empty or missing.
OPTIONAL_COLUMNS = [
    "furnished",            # yes, no or unknown (verified status only)
    "lease_months",         # rentals: lease length
    "utilities_included",   # rentals: yes, no or unknown
    "image_ref",            # ID of the matched exterior photo
    "image_provider",       # for example mapillary
    "image_license",        # for example CC BY-SA 4.0
    "image_captured_at",    # YYYY-MM-DD
    "image_distance_m",     # meters between the photo position and the house
]

ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

SUPPORTED_PROPERTY_TYPE = "detached_single_family"
TARGET_TYPES = ("sale", "rent")
PRICE_BASES = ("completed_transaction", "asking")
FURNISHED_VALUES = ("yes", "no", "unknown", "")
UNIT_FOR_TARGET = {"sale": "total", "rent": "per_month"}

# Plausible price ranges in USD. Rows outside are treated as data errors.
PLAUSIBLE_PRICE = {"sale": (50000.0, 30000000.0), "rent": (300.0, 50000.0)}

# A matched photo farther than this from the house probably shows another house.
MAX_IMAGE_DISTANCE_M = 40.0

# Area table (ZIP level typical values). Kept apart from the records table on purpose.
AREA_COLUMNS = [
    "zip", "city", "state", "metro", "metro_id", "supported_city_id",
    "lat", "lon", "zhvi_usd", "zhvi_month", "zori_usd_month", "zori_month", "synthetic",
]
