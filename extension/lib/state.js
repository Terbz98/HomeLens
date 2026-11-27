// The side panel's state, as plain data, and the rules for changing it.
//
// reduce(state, event) returns a new state and never touches the network or
// the page. That keeps the important rules easy to test:
//   - a result is only ever shown for the property it was computed for
//   - an answer that arrives after the user moved on is ignored (stale)
//   - a recoverable error keeps the last good result for the same property
//   - the estimate starts by itself once one house is known (a sure match, or
//     a house the user picked), assuming a detached house; "Not a house?" undoes that

export const STATUS_LABELS = {
  select_house: "Select a house",
  confirm_property: "Confirm property",
  loading: "Loading",
  estimate_ready: "Estimate ready",
  location_only: "Location-only estimate",
  unsupported_location: "Unsupported location",
  unsupported_property: "Unsupported property",
  insufficient_data: "Insufficient data",
  backend_unavailable: "Backend unavailable",
  request_failed: "Request failed",
};

export const BADGE_LABELS = {
  demo: "Demo mode",
  image_unavailable: "Exterior image unavailable",
};

export const HOUSE_TYPE = "detached_single_family";
const RESULT_STATUSES = new Set(["estimate_ready", "location_only"]);

export function initialState() {
  return {
    status: "select_house",
    message: "Open Google Maps and click a house, or type its address below.",
    selectionKey: null,     // identity of what is selected on the map
    selection: null,        // output of parseMapsUrl
    candidates: [],         // properties the backend offered for confirmation
    chosenId: null,         // candidate_id the user confirmed
    propertyType: null,     // HOUSE_TYPE (assumed once a house is known), "other", or null
    useImage: true,         // false after "This photo is not my house"
    result: null,           // { candidateId, response } of the last good estimate
    pending: null,          // { id, kind, body } request that is in flight
    lastFailed: null,       // { kind, body } request to repeat on "Try again"
    backendDemo: false,     // the backend said it is running in demo mode
    seq: 0,                 // counter used to make request ids
  };
}

function clearedForNewProperty(state) {
  return {
    ...state,
    candidates: [],
    chosenId: null,
    propertyType: null,
    useImage: true,
    result: null,
    pending: null,
    lastFailed: null,
  };
}

// Start the estimate for the chosen house, assuming a detached single-family house.
function startEstimate(state) {
  const c = chosenCandidate(state);
  if (!c || !c.supported) {
    return state;
  }
  const body = { lat: c.lat, lng: c.lng, property_type: HOUSE_TYPE, use_image: state.useImage };
  if (c.address) body.address = c.address;
  if (c.zip) body.zip = c.zip;
  if (c.property_id) body.property_id = c.property_id;
  return withRequest({ ...state, propertyType: HOUSE_TYPE }, "estimate", body);
}

function withRequest(state, kind, body) {
  const seq = state.seq + 1;
  const id = "r" + seq;
  const fullBody = kind === "estimate" ? { ...body, client_request_id: id } : body;
  return { ...state, seq, pending: { id, kind, body: fullBody }, lastFailed: null, status: "loading",
           message: kind === "resolve" ? "Finding the property." : "Estimating." };
}

export function chosenCandidate(state) {
  return state.candidates.find((c) => c.candidate_id === state.chosenId) || null;
}

// The only way the interface reads a result. It returns null unless the
// result belongs to the property that is confirmed right now.
export function visibleResult(state) {
  if (!state.result || !state.chosenId || state.result.candidateId !== state.chosenId) {
    return null;
  }
  return state.result.response;
}

export function badges(state) {
  const out = [];
  const result = visibleResult(state);
  if (state.backendDemo || (result && result.is_demo)) {
    out.push("demo");
  }
  if (result && result.image && (result.image.status === "unavailable" || result.image.status === "error")) {
    out.push("image_unavailable");
  }
  return out;
}

export function canEstimate(state) {
  const chosen = chosenCandidate(state);
  return Boolean(chosen && chosen.supported && state.propertyType === HOUSE_TYPE && !state.pending);
}

function isCurrent(state, requestId, kind) {
  return Boolean(state.pending && state.pending.id === requestId && state.pending.kind === kind);
}

export function reduce(state, event) {
  switch (event.type) {
    case "health":
      return { ...state, backendDemo: Boolean(event.demo) };

    // The Google Maps URL changed (or another tab was opened).
    case "selection": {
      const selection = event.selection;
      if (selection.key === state.selectionKey && state.selection && selection.kind === state.selection.kind) {
        return state; // same house, for example the user only panned or zoomed
      }
      const base = { ...clearedForNewProperty(state), selection, selectionKey: selection.key };
      if (selection.kind === "place" && selection.address) {
        const body = { address: selection.address, source: "place" };
        if (selection.lat !== null && selection.lng !== null) {
          body.lat = selection.lat;
          body.lng = selection.lng;
        }
        return withRequest(base, "resolve", body);
      }
      if (selection.kind === "street_view" && selection.lat !== null) {
        return withRequest(base, "resolve", { lat: selection.lat, lng: selection.lng, source: "street_view" });
      }
      return { ...base, status: "select_house", message: selection.reason };
    }

    // The user typed an address or pasted "lat, lng".
    case "manual": {
      const base = clearedForNewProperty(state);
      if (event.pin) {
        return withRequest(base, "resolve", { lat: event.pin.lat, lng: event.pin.lng, source: "manual" });
      }
      const text = (event.text || "").trim();
      if (text.length < 3) {
        return { ...base, status: "select_house", message: "Type a street address with city and state." };
      }
      return withRequest(base, "resolve", { address: text, source: "manual" });
    }

    case "resolved": {
      if (!isCurrent(state, event.requestId, "resolve")) {
        return state; // stale: the user has already selected something else
      }
      const response = event.response;
      const candidates = Array.isArray(response.candidates) ? response.candidates : [];
      const next = { ...state, pending: null, candidates, message: response.message || "" };
      if (response.status === "unsupported_location") {
        return { ...next, status: "unsupported_location", chosenId: null };
      }
      if (response.status === "not_found" || candidates.length === 0) {
        return { ...next, status: "select_house", candidates: [], chosenId: null };
      }
      // Estimate at once only when the backend is sure which house it is. A bare
      // map position or an unmatched address is never preselected: the user picks.
      const sure = response.status === "ok" && candidates.length === 1 && candidates[0].supported;
      if (sure) {
        return startEstimate({ ...next, chosenId: candidates[0].candidate_id });
      }
      return { ...next, status: "confirm_property", chosenId: null };
    }

    case "choose": {
      if (state.pending || !state.candidates.some((c) => c.candidate_id === event.candidateId)) {
        return state;
      }
      if (event.candidateId === state.chosenId && visibleResult(state)) {
        return state;
      }
      return startEstimate({ ...state, chosenId: event.candidateId, result: null });
    }

    case "property_type": {
      if (state.pending) {
        return state;
      }
      if (event.value !== HOUSE_TYPE) {
        return { ...state, propertyType: "other", result: null, status: "unsupported_property",
                 message: "HomeLens only supports detached single-family homes." };
      }
      return startEstimate(state);  // "It is a detached house": estimate again
    }

    case "estimate": {
      if (!canEstimate(state)) {
        return state;
      }
      return startEstimate({ ...state, useImage: event.useImage !== false });
    }

    case "estimated": {
      if (!isCurrent(state, event.requestId, "estimate")) {
        return state; // stale: an answer for a request the user has moved on from
      }
      const response = event.response;
      if (!response || response.client_request_id !== event.requestId) {
        return { ...state, pending: null, status: "request_failed", lastFailed: { kind: "estimate", body: state.pending.body },
                 message: "The backend answered for a different request. Try again." };
      }
      const next = { ...state, pending: null, message: response.message || "" };
      if (RESULT_STATUSES.has(response.status)) {
        return { ...next, status: response.status, result: { candidateId: state.chosenId, response } };
      }
      const known = Object.prototype.hasOwnProperty.call(STATUS_LABELS, response.status);
      return { ...next, status: known ? response.status : "request_failed", result: null };
    }

    case "failed": {
      if (!state.pending || state.pending.id !== event.requestId) {
        return state; // stale failure
      }
      const status = event.errorKind === "network" ? "backend_unavailable" : "request_failed";
      const kept = visibleResult(state) ? " The last estimate for this property is still shown." : "";
      return { ...state, pending: null, status, lastFailed: { kind: state.pending.kind, body: state.pending.body },
               message: (event.message || "The request failed.") + kept };
    }

    case "retry": {
      if (state.pending || !state.lastFailed) {
        return state;
      }
      return withRequest(state, state.lastFailed.kind, state.lastFailed.body);
    }

    default:
      return state;
  }
}
