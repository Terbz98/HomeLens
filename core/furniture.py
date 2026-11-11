"""The standard furniture scenario.

Exterior photos cannot show furniture, so furnished numbers are never
predictions about a real interior. They are the unfurnished estimate plus a
fixed, documented package from config/furniture_package.json.
"""
import json
from pathlib import Path
from typing import Dict, Optional

from core.paths import CONFIG_DIR


def load_package(config_dir: Optional[Path] = None) -> Dict:
    path = (config_dir or CONFIG_DIR) / "furniture_package.json"
    with open(path, "r", encoding="utf-8") as f:
        package = json.load(f)
    validate_package(package)
    return package


def validate_package(package: Dict) -> None:
    """Raise ValueError if the package config makes no sense."""
    items = package.get("items")
    if not items:
        raise ValueError("furniture package has no items")
    for item in items:
        if item["quantity"] <= 0 or item["assumed_new_price_usd"] < 0:
            raise ValueError("bad furniture item: %s" % item.get("name"))
    if not 0.0 <= package["resale_fraction"] <= 1.0:
        raise ValueError("resale_fraction must be between 0 and 1")
    if package["rent_spread_months"] <= 0:
        raise ValueError("rent_spread_months must be positive")


def new_cost_usd(package: Dict) -> float:
    """Total assumed price of the whole package when new."""
    return float(sum(i["quantity"] * i["assumed_new_price_usd"] for i in package["items"]))


def resale_value_usd(package: Dict) -> float:
    """Assumed resale value of the used package. Added to the purchase price."""
    return round(new_cost_usd(package) * package["resale_fraction"], 2)


def monthly_amount_usd(package: Dict) -> float:
    """Scenario amount added to monthly rent: new cost spread over N months."""
    return round(new_cost_usd(package) / package["rent_spread_months"], 2)


def furnished_purchase(unfurnished_usd: float, package: Dict) -> float:
    return unfurnished_usd + resale_value_usd(package)


def furnished_rent(unfurnished_usd_month: float, package: Dict) -> float:
    return unfurnished_usd_month + monthly_amount_usd(package)


def summary(package: Dict) -> Dict:
    """Everything the interface needs to explain the furniture assumption."""
    return {
        "package_name": package["package_name"],
        "package_version": package["package_version"],
        "condition": package["condition"],
        "valuation_basis": package["valuation_basis"],
        "fixed_inventory_note": package["fixed_inventory_note"],
        "items": [{"name": i["name"], "quantity": i["quantity"]} for i in package["items"]],
        "new_cost_usd": new_cost_usd(package),
        "resale_fraction": package["resale_fraction"],
        "resale_value_usd": resale_value_usd(package),
        "rent_spread_months": package["rent_spread_months"],
        "monthly_amount_usd": monthly_amount_usd(package),
        "rent_basis": package["rent_basis"],
        "source": "assumption",
    }
