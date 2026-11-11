// Read a Google Maps page address (URL) and work out what the user has selected.
//
// Only the URL text is used. Nothing is read from the map picture, the page
// contents or Street View imagery.
//
// Two different positions can appear in a Google Maps URL:
//   - after "@":       where the camera (or the Street View car) is
//   - "!3d...!4d...":  where the selected place itself is
// The camera position is NOT the house. Only the place position, or an
// address, identifies a house.

const MAPS_HOSTS = new Set(["www.google.com", "google.com"]);

function toNumber(text) {
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
}

function validLatLng(lat, lng) {
  return lat !== null && lng !== null && Math.abs(lat) <= 90 && Math.abs(lng) <= 180;
}

function safeDecode(text) {
  try {
    return decodeURIComponent(text.replace(/\+/g, " "));
  } catch (error) {
    return text.replace(/\+/g, " ");
  }
}

// "209 N 41st St, Seattle, WA 98103" looks like a street address.
// "Space Needle" or "Fremont, Seattle, WA" does not.
export function looksLikeStreetAddress(text) {
  return /^\s*\d+[A-Za-z]?(-\d+)?\s+\S+/.test(text || "");
}

// Google Maps adds the country after the ZIP code, in the browser's language
// and sometimes with no comma (for example "... WA 98103, USA" or "... WA 98103美國").
// Everything after a US state and ZIP code is dropped.
export function cleanAddress(text) {
  const cut = text.match(/^(.*\b[A-Z]{2},? \d{5}(?:-\d{4})?)(?!\d)/);
  if (cut) {
    return cut[1].trim();
  }
  return text.replace(/,\s*(USA|United States)$/i, "").trim();
}

function round(value, places) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function emptySelection(kind, reason) {
  return {
    kind,                 // not_maps | none | place | street_view
    reason,               // short plain explanation of what was found
    address: null,        // street address text, when the URL has one
    placeName: null,      // place text that is not a street address
    lat: null,            // position to send to the backend, when one is usable
    lng: null,
    positionIsCamera: false, // true when lat/lng is a camera position, not a house
    needsAddress: true,   // true when the user must type or confirm an address
    source: "map_center", // value for the backend "source" field
    key: null,            // identity of this selection, used to detect stale results
  };
}

export function parseMapsUrl(urlText) {
  let url;
  try {
    url = new URL(urlText);
  } catch (error) {
    return emptySelection("not_maps", "This tab is not Google Maps.");
  }
  if (url.protocol !== "https:" || !MAPS_HOSTS.has(url.hostname) || !/^\/maps(\/|$)/.test(url.pathname)) {
    return emptySelection("not_maps", "This tab is not Google Maps.");
  }

  const path = url.pathname;
  const camera = path.match(/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),([^/]*)/);
  let cameraLat = null;
  let cameraLng = null;
  let isStreetView = false;
  if (camera) {
    cameraLat = toNumber(camera[1]);
    cameraLng = toNumber(camera[2]);
    // Street View URLs carry "3a" right after the coordinates.
    isStreetView = /(^|,)3a(,|$)/.test(camera[3]);
    if (!validLatLng(cameraLat, cameraLng)) {
      cameraLat = null;
      cameraLng = null;
    }
  }

  // The selected place's own position. If it appears more than once, the last one is the place.
  let placeLat = null;
  let placeLng = null;
  const placeMatches = [...path.matchAll(/!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)/g)];
  if (placeMatches.length > 0) {
    const last = placeMatches[placeMatches.length - 1];
    placeLat = toNumber(last[1]);
    placeLng = toNumber(last[2]);
    if (!validLatLng(placeLat, placeLng)) {
      placeLat = null;
      placeLng = null;
    }
  }

  const place = path.match(/^\/maps\/place\/([^/@]+)/);
  if (place) {
    const text = safeDecode(place[1]).replace(/\s+/g, " ").trim();
    const selection = emptySelection("place", "");
    selection.source = "place";
    if (looksLikeStreetAddress(text)) {
      selection.address = cleanAddress(text);
      selection.lat = placeLat;
      selection.lng = placeLng;
      selection.needsAddress = false;
      selection.reason = "A street address is selected on the map.";
      selection.key = "place:" + selection.address.toLowerCase();
    } else {
      selection.placeName = text;
      selection.reason = "The selected place is not a street address. Type the address of the house.";
      selection.key = "name:" + text.toLowerCase();
    }
    return selection;
  }

  if (isStreetView && cameraLat !== null) {
    const selection = emptySelection("street_view", "Street View is open. The camera position is not the house itself.");
    selection.source = "street_view";
    selection.lat = cameraLat;
    selection.lng = cameraLng;
    selection.positionIsCamera = true;
    // Turning the camera keeps the same spot. Moving along the street is a new spot.
    selection.key = "sv:" + round(cameraLat, 5) + "," + round(cameraLng, 5);
    return selection;
  }

  if (/^\/maps\/dir\//.test(path)) {
    return emptySelection("none", "Directions are open. Select a single house instead.");
  }
  if (/^\/maps\/search\//.test(path)) {
    return emptySelection("none", "A search is open. Click one house in the results or on the map.");
  }
  return emptySelection("none", "No house is selected yet.");
}

// Parse a "47.6561, -122.3560" pin typed or pasted by the user.
export function parseLatLngText(text) {
  const match = (text || "").trim().match(/^(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)$/);
  if (!match) {
    return null;
  }
  const lat = toNumber(match[1]);
  const lng = toNumber(match[2]);
  return validLatLng(lat, lng) ? { lat, lng } : null;
}
