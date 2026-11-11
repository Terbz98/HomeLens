// Google Maps URL parsing with representative URLs.
import assert from "node:assert/strict";
import { test } from "node:test";

import { SAMPLE_URLS } from "../dev/sample_urls.js";
import { cleanAddress, looksLikeStreetAddress, parseLatLngText, parseMapsUrl } from "../lib/maps_url.js";

test("an address place gives the address and the place position, not the camera position", () => {
  const s = parseMapsUrl(SAMPLE_URLS["Seattle address selected"]);
  assert.equal(s.kind, "place");
  assert.equal(s.address, "209 N 41st St, Seattle, WA 98103");
  assert.equal(s.lat, 47.6561157);
  assert.equal(s.lng, -122.3559997);          // from !3d!4d, not the -122.3585746 camera value after "@"
  assert.equal(s.needsAddress, false);
  assert.equal(s.positionIsCamera, false);
  assert.equal(s.source, "place");
});

test("Street View alone gives a camera position that is flagged as not the house", () => {
  const s = parseMapsUrl(SAMPLE_URLS["Seattle Street View, no address"]);
  assert.equal(s.kind, "street_view");
  assert.equal(s.address, null);
  assert.equal(s.positionIsCamera, true);
  assert.equal(s.needsAddress, true);
  assert.equal(s.lat, 47.6563012);
  assert.equal(s.source, "street_view");
});

test("Street View opened from an address keeps the address", () => {
  const url = "https://www.google.com/maps/place/209+N+41st+St,+Seattle,+WA+98103/@47.6563,-122.3562,3a,75y,180h,90t/"
    + "data=!3m7!1e1!3m5!1sAbCdEf!2e0!6shttps:%2F%2Fexample!7i16384!8i8192!4m6!3m5!1s0x1:0x2!8m2!3d47.6561157!4d-122.3559997!16s%2Fg%2F1";
  const s = parseMapsUrl(url);
  assert.equal(s.kind, "place");
  assert.equal(s.address, "209 N 41st St, Seattle, WA 98103");
  assert.equal(s.lat, 47.6561157);
  assert.equal(s.positionIsCamera, false);
});

test("the trailing country is removed and encoded characters are decoded", () => {
  const s = parseMapsUrl(SAMPLE_URLS["Portland address selected"]);
  assert.equal(s.address, "3415 SE Clinton St, Portland, OR 97202");
  const encoded = parseMapsUrl("https://www.google.com/maps/place/12%20Main%20St%2C%20Portland%2C%20OR%2097214/@45.5,-122.6,17z");
  assert.equal(encoded.address, "12 Main St, Portland, OR 97214");
  assert.equal(encoded.lat, null);            // no place position in this URL: the backend looks the address up
});

test("a country name glued on by Google Maps in another language is removed", () => {
  // Seen in a real Chrome set to Traditional Chinese: the country follows the ZIP with no comma.
  const url = "https://www.google.com/maps/place/209+N+41st+St,+Seattle,+WA+98103%E7%BE%8E%E5%9C%8B/@47.6561733,-122.3560291,17z/"
    + "data=!3m1!4b1!4m6!3m5!1s0x1:0x2!8m2!3d47.6561733!4d-122.3560291!16s%2Fg%2F1";
  const s = parseMapsUrl(url);
  assert.equal(s.address, "209 N 41st St, Seattle, WA 98103");
  assert.equal(s.lat, 47.6561733);
  assert.equal(cleanAddress("12 Main St, Portland, OR 97214-1234, Estados Unidos"), "12 Main St, Portland, OR 97214-1234");
  assert.equal(cleanAddress("12 Main St, Portland"), "12 Main St, Portland");
});

test("a landmark is not treated as a house address", () => {
  const s = parseMapsUrl(SAMPLE_URLS["Landmark, not an address"]);
  assert.equal(s.kind, "place");
  assert.equal(s.address, null);
  assert.equal(s.placeName, "Space Needle");
  assert.equal(s.needsAddress, true);
  assert.equal(s.lat, null);                  // its position is not offered as a house
});

test("map, search and directions views select nothing", () => {
  for (const url of [
    SAMPLE_URLS["Plain map, nothing selected"],
    "https://www.google.com/maps",
    "https://www.google.com/maps/search/houses+for+sale/@47.6,-122.3,13z",
    "https://www.google.com/maps/dir/Seattle,+WA/Portland,+OR/@46.5,-123.4,8z",
    "https://www.google.com/maps/@47.65,-122.35,157m/data=!3m1!1e3",   // satellite view, not Street View
  ]) {
    const s = parseMapsUrl(url);
    assert.equal(s.kind, "none", url);
    assert.equal(s.lat, null);
    assert.equal(s.key, null);
  }
});

test("pages that are not Google Maps are refused", () => {
  for (const url of [
    SAMPLE_URLS["Not Google Maps"],
    "http://www.google.com/maps/place/1+A+St/",            // not https
    "https://www.google.com.evil.example/maps/place/1+A+St/",
    "https://evil.example/www.google.com/maps/place/1+A+St/",
    "https://www.google.com/mapsfake/place/1+A+St/",
    "not a url",
    "",
  ]) {
    assert.equal(parseMapsUrl(url).kind, "not_maps", url);
  }
});

test("broken numbers and broken encoding do not crash or produce a position", () => {
  const s = parseMapsUrl("https://www.google.com/maps/place/5+%E0%A4%A+St/@999,999,17z/data=!3d999!4d999");
  assert.equal(s.kind, "place");
  assert.equal(s.lat, null);
  assert.equal(s.lng, null);
});

test("the selection key ignores panning and camera turns but changes with the house", () => {
  const a = parseMapsUrl(SAMPLE_URLS["Seattle address selected"]);
  const panned = parseMapsUrl(SAMPLE_URLS["Seattle address selected"].replace("@47.6561157,-122.3585746,17z", "@47.66,-122.36,15z"));
  assert.equal(a.key, panned.key);
  assert.notEqual(a.key, parseMapsUrl(SAMPLE_URLS["Portland address selected"]).key);

  const sv = SAMPLE_URLS["Seattle Street View, no address"];
  const turned = parseMapsUrl(sv.replace("182.4h", "10h"));
  const moved = parseMapsUrl(sv.replace("47.6563012", "47.6569012"));
  assert.equal(parseMapsUrl(sv).key, turned.key);
  assert.notEqual(parseMapsUrl(sv).key, moved.key);
});

test("street address detection", () => {
  assert.equal(looksLikeStreetAddress("209 N 41st St, Seattle, WA"), true);
  assert.equal(looksLikeStreetAddress("12B Baker St"), true);
  assert.equal(looksLikeStreetAddress("Space Needle"), false);
  assert.equal(looksLikeStreetAddress("Fremont, Seattle, WA"), false);
  assert.equal(looksLikeStreetAddress("98103"), false);
});

test("typed map pins", () => {
  assert.deepEqual(parseLatLngText(" 47.6561, -122.3560 "), { lat: 47.6561, lng: -122.356 });
  assert.equal(parseLatLngText("209 N 41st St"), null);
  assert.equal(parseLatLngText("91, 0"), null);
  assert.equal(parseLatLngText(""), null);
});
