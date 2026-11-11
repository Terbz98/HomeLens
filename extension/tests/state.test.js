// The side panel rules: confirmation, stale answers, kept results, demo and image labels.
import assert from "node:assert/strict";
import { test } from "node:test";

import { SAMPLE_URLS } from "../dev/sample_urls.js";
import { parseMapsUrl } from "../lib/maps_url.js";
import { BADGE_LABELS, HOUSE_TYPE, STATUS_LABELS, badges, canEstimate, chosenCandidate, initialState, reduce,
         visibleResult } from "../lib/state.js";

const SEATTLE = parseMapsUrl(SAMPLE_URLS["Seattle address selected"]);
const PORTLAND = parseMapsUrl(SAMPLE_URLS["Portland address selected"]);
const STREET_VIEW = parseMapsUrl(SAMPLE_URLS["Seattle Street View, no address"]);

function candidate(id, address, supported = true) {
  return { candidate_id: id, address, lat: 47.656, lng: -122.356, zip: "98103", city_id: "seattle",
           city_name: "Seattle", supported, match: "parcel", property_id: "P-" + id, distance_m: 5, note: "" };
}

function estimateResponse(requestId, extra = {}) {
  return {
    status: "location_only", message: "Location-only estimate.", client_request_id: requestId, is_demo: false,
    prediction_mode: "location_only", image: { status: "no_photo_model", message: "" },
    estimates: [{ key: "buy_unfurnished", amount_usd: 1000000 }], ...extra,
  };
}

function run(state, ...events) {
  return events.reduce(reduce, state);
}

// Select an address on the map, get one sure candidate back, confirm the type.
function confirmedSeattle() {
  let s = reduce(initialState(), { type: "selection", selection: SEATTLE });
  s = reduce(s, { type: "resolved", requestId: s.pending.id,
                  response: { status: "ok", candidates: [candidate("A", "209 N 41st St, Seattle, WA 98103")], message: "Check." } });
  return reduce(s, { type: "property_type", value: HOUSE_TYPE });
}

function withEstimate(state, extra) {
  const s = reduce(state, { type: "estimate", useImage: true });
  return reduce(s, { type: "estimated", requestId: s.pending.id, response: estimateResponse(s.pending.id, extra) });
}

test("every status and badge from the specification has a label", () => {
  assert.deepEqual(Object.values(STATUS_LABELS).sort(), [
    "Backend unavailable", "Confirm property", "Estimate ready", "Insufficient data", "Loading",
    "Location-only estimate", "Request failed", "Select a house", "Unsupported location", "Unsupported property",
  ]);
  assert.deepEqual(Object.values(BADGE_LABELS).sort(), ["Demo mode", "Exterior image unavailable"]);
});

test("starts by asking the user to select a house", () => {
  const s = initialState();
  assert.equal(s.status, "select_house");
  assert.equal(visibleResult(s), null);
  assert.equal(canEstimate(s), false);
});

test("selecting an address asks the backend to resolve it, with the place position", () => {
  const s = reduce(initialState(), { type: "selection", selection: SEATTLE });
  assert.equal(s.status, "loading");
  assert.equal(s.pending.kind, "resolve");
  assert.deepEqual(s.pending.body, { address: "209 N 41st St, Seattle, WA 98103", source: "place", lat: 47.6561157, lng: -122.3559997 });
});

test("a sure match is preselected but still needs the property type before estimating", () => {
  let s = reduce(initialState(), { type: "selection", selection: SEATTLE });
  s = reduce(s, { type: "resolved", requestId: s.pending.id, response: { status: "ok", candidates: [candidate("A", "209 N 41st St")], message: "" } });
  assert.equal(s.status, "confirm_property");
  assert.equal(chosenCandidate(s).candidate_id, "A");
  assert.equal(canEstimate(s), false);                       // property type not answered yet
  assert.equal(reduce(s, { type: "estimate" }), s);          // the estimate request is refused
  s = reduce(s, { type: "property_type", value: HOUSE_TYPE });
  assert.equal(canEstimate(s), true);
});

test("a Street View position never picks a house silently", () => {
  let s = reduce(initialState(), { type: "selection", selection: STREET_VIEW });
  assert.deepEqual(s.pending.body, { lat: 47.6563012, lng: -122.3562034, source: "street_view" });
  s = reduce(s, { type: "resolved", requestId: s.pending.id, response: {
    status: "needs_confirmation", message: "Pick the house you mean.",
    candidates: [candidate("N1", "114 N 41st St"), candidate("N2", "209 N 41st St"), candidate("N3", "118 N 41st St")] } });
  assert.equal(s.status, "confirm_property");
  assert.equal(s.chosenId, null);                            // nothing preselected, not even the nearest
  s = reduce(s, { type: "property_type", value: HOUSE_TYPE });
  assert.equal(canEstimate(s), false);                       // still no house chosen
  s = reduce(s, { type: "choose", candidateId: "N2" });
  assert.equal(canEstimate(s), true);
  s = reduce(s, { type: "estimate", useImage: true });
  assert.equal(s.pending.body.address, "209 N 41st St");     // the chosen house, not the neighbor
  assert.equal(s.pending.body.property_id, "P-N2");
});

test("a single unconfirmed map pin is not preselected either", () => {
  let s = reduce(initialState(), { type: "selection", selection: STREET_VIEW });
  s = reduce(s, { type: "resolved", requestId: s.pending.id, response: {
    status: "needs_confirmation", message: "", candidates: [{ ...candidate("PIN", null), match: "pin" }] } });
  assert.equal(s.chosenId, null);
});

test("choosing an unknown candidate does nothing", () => {
  const s = confirmedSeattle();
  assert.equal(reduce(s, { type: "choose", candidateId: "nope" }), s);
});

test("a map view with nothing selected goes back to Select a house", () => {
  const s = run(confirmedSeattle(), { type: "selection", selection: parseMapsUrl(SAMPLE_URLS["Plain map, nothing selected"]) });
  assert.equal(s.status, "select_house");
  assert.equal(s.candidates.length, 0);
  assert.equal(s.pending, null);
});

test("manual address and manual pin fallbacks need no property details", () => {
  let s = reduce(initialState(), { type: "manual", text: "209 N 41st St, Seattle, WA" });
  assert.deepEqual(s.pending.body, { address: "209 N 41st St, Seattle, WA", source: "manual" });
  s = reduce(initialState(), { type: "manual", text: "47.6561, -122.356", pin: { lat: 47.6561, lng: -122.356 } });
  assert.deepEqual(s.pending.body, { lat: 47.6561, lng: -122.356, source: "manual" });
  s = reduce(initialState(), { type: "manual", text: " a " });
  assert.equal(s.pending, null);
  assert.equal(s.status, "select_house");
});

test("unsupported location and address not found", () => {
  let s = reduce(initialState(), { type: "selection", selection: SEATTLE });
  const unsupported = reduce(s, { type: "resolved", requestId: s.pending.id, response: {
    status: "unsupported_location", message: "Bellevue is outside the supported cities.", candidates: [candidate("B", "1 A St", false)] } });
  assert.equal(unsupported.status, "unsupported_location");
  assert.equal(canEstimate(reduce(unsupported, { type: "property_type", value: HOUSE_TYPE })), false);
  const missing = reduce(s, { type: "resolved", requestId: s.pending.id, response: { status: "not_found", message: "Not found.", candidates: [] } });
  assert.equal(missing.status, "select_house");
  assert.equal(missing.message, "Not found.");
});

test("answering 'something else' gives Unsupported property and no request", () => {
  let s = confirmedSeattle();
  s = reduce(s, { type: "property_type", value: "other" });
  assert.equal(s.status, "unsupported_property");
  assert.equal(canEstimate(s), false);
  assert.equal(reduce(s, { type: "estimate" }).pending, null);
});

test("a successful estimate is shown for the confirmed property", () => {
  const s = withEstimate(confirmedSeattle());
  assert.equal(s.status, "location_only");
  assert.equal(visibleResult(s).estimates[0].amount_usd, 1000000);
  const ready = withEstimate(confirmedSeattle(), { status: "estimate_ready", prediction_mode: "location_plus_image", image: { status: "used" } });
  assert.equal(ready.status, "estimate_ready");
});

test("the estimate request carries its own id and the user's photo choice", () => {
  const s = reduce(confirmedSeattle(), { type: "estimate", useImage: false });
  assert.equal(s.status, "loading");
  assert.equal(s.pending.body.client_request_id, s.pending.id);
  assert.equal(s.pending.body.use_image, false);
  assert.equal(s.pending.body.property_type, HOUSE_TYPE);
  assert.ok(!("bedrooms" in s.pending.body) && !("sqft" in s.pending.body));
});

test("STALE: an estimate that arrives after the user selected another house is ignored", () => {
  let s = reduce(confirmedSeattle(), { type: "estimate", useImage: true });
  const oldId = s.pending.id;
  s = reduce(s, { type: "selection", selection: PORTLAND });           // user moved on while it was loading
  const newResolveId = s.pending.id;
  const after = reduce(s, { type: "estimated", requestId: oldId, response: estimateResponse(oldId) });
  assert.equal(after, s);                                              // the late answer changed nothing
  assert.equal(visibleResult(after), null);
  assert.equal(after.pending.id, newResolveId);
});

test("STALE: an old resolve answer cannot replace a newer selection", () => {
  let s = reduce(initialState(), { type: "selection", selection: SEATTLE });
  const oldId = s.pending.id;
  s = reduce(s, { type: "selection", selection: PORTLAND });
  const late = reduce(s, { type: "resolved", requestId: oldId, response: { status: "ok", candidates: [candidate("A", "209 N 41st St")], message: "" } });
  assert.equal(late, s);
  assert.equal(late.candidates.length, 0);
});

test("STALE: a late failure for an old request is ignored", () => {
  let s = reduce(confirmedSeattle(), { type: "estimate", useImage: true });
  const oldId = s.pending.id;
  s = reduce(s, { type: "selection", selection: PORTLAND });
  assert.equal(reduce(s, { type: "failed", requestId: oldId, errorKind: "network" }), s);
});

test("an answer carrying the wrong request id is refused", () => {
  const s = reduce(confirmedSeattle(), { type: "estimate", useImage: true });
  const after = reduce(s, { type: "estimated", requestId: s.pending.id, response: estimateResponse("someone-else") });
  assert.equal(after.status, "request_failed");
  assert.equal(visibleResult(after), null);
});

test("results for a previous property are never shown under a newly selected address", () => {
  const done = withEstimate(confirmedSeattle());
  assert.ok(visibleResult(done));
  let s = reduce(done, { type: "selection", selection: PORTLAND });
  assert.equal(visibleResult(s), null);                                // gone as soon as the selection changes
  s = reduce(s, { type: "resolved", requestId: s.pending.id, response: { status: "ok", candidates: [candidate("P", "3415 SE Clinton St")], message: "" } });
  assert.equal(visibleResult(s), null);
  s = reduce(s, { type: "property_type", value: HOUSE_TYPE });
  assert.equal(visibleResult(s), null);
});

test("switching between candidates clears the result of the other candidate", () => {
  let s = reduce(initialState(), { type: "selection", selection: STREET_VIEW });
  s = reduce(s, { type: "resolved", requestId: s.pending.id, response: {
    status: "needs_confirmation", message: "", candidates: [candidate("N1", "114 N 41st St"), candidate("N2", "209 N 41st St")] } });
  s = run(s, { type: "choose", candidateId: "N1" }, { type: "property_type", value: HOUSE_TYPE });
  s = withEstimate(s);
  assert.ok(visibleResult(s));
  s = reduce(s, { type: "choose", candidateId: "N2" });
  assert.equal(visibleResult(s), null);
  assert.equal(s.status, "confirm_property");
});

test("panning or zooming the same house keeps everything", () => {
  const done = withEstimate(confirmedSeattle());
  const panned = parseMapsUrl(SAMPLE_URLS["Seattle address selected"].replace(",17z", ",15z"));
  assert.equal(reduce(done, { type: "selection", selection: panned }), done);
});

test("backend unavailable keeps the last good result for the same property", () => {
  const done = withEstimate(confirmedSeattle());
  let s = reduce(done, { type: "estimate", useImage: true });
  assert.ok(visibleResult(s));                                          // still visible while loading again
  s = reduce(s, { type: "failed", requestId: s.pending.id, errorKind: "network", message: "The backend is not running." });
  assert.equal(s.status, "backend_unavailable");
  assert.equal(visibleResult(s).estimates[0].amount_usd, 1000000);
  assert.match(s.message, /still shown/);
  assert.equal(s.pending, null);
});

test("timeouts and error answers give Request failed, and Try again repeats the request with a new id", () => {
  let s = reduce(confirmedSeattle(), { type: "estimate", useImage: true });
  const firstId = s.pending.id;
  s = reduce(s, { type: "failed", requestId: firstId, errorKind: "timeout", message: "The backend did not answer in time." });
  assert.equal(s.status, "request_failed");
  assert.equal(visibleResult(s), null);
  s = reduce(s, { type: "retry" });
  assert.equal(s.status, "loading");
  assert.notEqual(s.pending.id, firstId);
  assert.equal(s.pending.body.client_request_id, s.pending.id);
  const late = reduce(s, { type: "estimated", requestId: firstId, response: estimateResponse(firstId) });
  assert.equal(late, s);                                                // the first attempt's answer is stale now
});

test("a failed address lookup can be retried too", () => {
  let s = reduce(initialState(), { type: "selection", selection: SEATTLE });
  s = reduce(s, { type: "failed", requestId: s.pending.id, errorKind: "network" });
  assert.equal(s.status, "backend_unavailable");
  s = reduce(s, { type: "retry" });
  assert.equal(s.pending.kind, "resolve");
});

test("backend refusals clear any numbers", () => {
  for (const status of ["insufficient_data", "unsupported_property", "unsupported_location"]) {
    const s = withEstimate(confirmedSeattle(), { status, estimates: [] });
    assert.equal(s.status, status);
    assert.equal(visibleResult(s), null);
  }
  const odd = withEstimate(confirmedSeattle(), { status: "something_new" });
  assert.equal(odd.status, "request_failed");
});

test("demo mode is labeled from the health check and from the result", () => {
  assert.deepEqual(badges(reduce(initialState(), { type: "health", demo: true })), ["demo"]);
  assert.deepEqual(badges(initialState()), []);
  const demo = withEstimate(confirmedSeattle(), { is_demo: true });
  assert.ok(badges(demo).includes("demo"));
  assert.deepEqual(badges(withEstimate(confirmedSeattle())), []);
});

test("exterior image unavailable is labeled when a photo was wanted but not found", () => {
  for (const status of ["unavailable", "error"]) {
    const s = withEstimate(confirmedSeattle(), { image: { status, message: "No photo." } });
    assert.deepEqual(badges(s), ["image_unavailable"]);
    assert.equal(s.status, "location_only");
  }
  assert.deepEqual(badges(withEstimate(confirmedSeattle(), { image: { status: "not_requested" } })), []);
});

test("nothing can be changed while a request is in flight", () => {
  const s = reduce(confirmedSeattle(), { type: "estimate", useImage: true });
  assert.equal(reduce(s, { type: "estimate", useImage: true }), s);
  assert.equal(reduce(s, { type: "property_type", value: "other" }), s);
  assert.equal(reduce(s, { type: "choose", candidateId: "A" }), s);
  assert.equal(reduce(s, { type: "retry" }), s);
});
