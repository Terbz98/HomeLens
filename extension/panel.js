// The side panel: draws the state and sends user actions to the state rules in lib/state.js.
//
// startPanel() is given three things so the same code runs in Chrome and in the
// local preview page:
//   root  the page element to draw into
//   api   the backend client from lib/api.js
//   maps  { getActiveUrl(): Promise<string|null>, onUrlChange(callback) }

import { parseLatLngText, parseMapsUrl } from "./lib/maps_url.js";
import { formatAmount, formatDate, formatPercent, formatRange, formatUsd, unitLabel } from "./lib/format.js";
import { BADGE_LABELS, HOUSE_TYPE, STATUS_LABELS, badges, canEstimate, chosenCandidate, initialState, reduce,
         visibleResult } from "./lib/state.js";

const STATUS_TONE = {
  select_house: "neutral", confirm_property: "info", loading: "info", estimate_ready: "good",
  location_only: "good", unsupported_location: "warn", unsupported_property: "warn",
  insufficient_data: "warn", backend_unavailable: "bad", request_failed: "bad",
};
const BASIS_TAG = {
  per_house_model: "Per-house model", area_level_model: "Area-level figure", assumption_added: "Assumption added",
};
const KIND_TAG = { trained: "Trained model", scenario: "Scenario", demo: "Demo, not real" };
const MODE_TEXT = {
  location_plus_image: "Mode: location plus a licensed street photo",
  location_only: "Mode: location only. The visible exterior was not analyzed.",
  none: "Mode: no estimate",
};

// Build a page element. Text is always set with textContent, never as HTML.
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (name === "class") node.className = value;
    else if (name === "text") node.textContent = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else if (value === true) node.setAttribute(name, "");
    else node.setAttribute(name, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function logo() {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 32 32");
  svg.setAttribute("class", "logo");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = '<circle cx="14" cy="14" r="11" fill="none" stroke="currentColor" stroke-width="2.5"/>'
    + '<path d="M8.5 14.5 14 9.5l5.5 5v5h-11z" fill="currentColor"/>'
    + '<path d="m22 22 7 7" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>';
  return svg;
}

export function startPanel({ root, api, maps }) {
  let state = initialState();
  let launchedId = null;
  const ui = { manualText: "", useImage: true, manualOpen: false };

  function dispatch(event) {
    const before = state;
    state = reduce(state, event);
    if (state.pending && state.pending.id !== launchedId) {
      launch(state.pending);
    }
    if (state !== before) {
      render();
    }
  }

  function launch(pending) {
    launchedId = pending.id;
    const call = pending.kind === "resolve" ? api.resolve(pending.body) : api.estimate(pending.body);
    call.then(
      (response) => dispatch({ type: pending.kind === "resolve" ? "resolved" : "estimated", requestId: pending.id, response }),
      (error) => dispatch({ type: "failed", requestId: pending.id, errorKind: error.kind || "http",
                            message: error.kind === "network"
                              ? "The HomeLens backend is not running or cannot be reached."
                              : (error.message || "The request failed.") }),
    );
  }

  function onUrl(url) {
    dispatch({ type: "selection", selection: parseMapsUrl(url || "") });
  }

  function checkBackend() {
    api.health().then(
      (health) => dispatch({ type: "health", demo: health.mode === "demo" }),
      () => { /* shown when the first real request fails */ },
    );
  }

  function submitManual(event) {
    event.preventDefault();
    const text = ui.manualText.trim();
    ui.manualOpen = false;
    dispatch({ type: "manual", text, pin: parseLatLngText(text) });
  }

  // ---------------------------------------------------------------- drawing

  function header() {
    const pills = [el("span", { class: "pill tone-" + STATUS_TONE[state.status], text: STATUS_LABELS[state.status] })];
    for (const badge of badges(state)) {
      pills.push(el("span", { class: "pill tone-" + (badge === "demo" ? "demo" : "warn"), text: BADGE_LABELS[badge] }));
    }
    return el("header", { class: "top" },
      el("div", { class: "brand" }, logo(), el("h1", { text: "HomeLens" })),
      el("div", { class: "pills", role: "status", "aria-live": "polite" }, pills),
    );
  }

  function messageLine() {
    if (!state.message) return null;
    const failed = state.status === "backend_unavailable" || state.status === "request_failed";
    return el("div", { class: "message" + (failed ? " is-error" : "") },
      el("p", { text: state.message }),
      state.status === "backend_unavailable"
        ? el("p", { class: "hint", text: "Start it from the project folder with: uvicorn backend.app.main:app --port 8000" })
        : null,
      failed && state.lastFailed
        ? el("button", { type: "button", id: "retry", class: "button", onclick: () => dispatch({ type: "retry" }) }, "Try again")
        : null,
    );
  }

  function candidateRow(candidate) {
    const id = "cand-" + candidate.candidate_id;
    const title = candidate.address || ("Map pin at " + candidate.lat.toFixed(5) + ", " + candidate.lng.toFixed(5));
    const place = candidate.city_name || "Unknown area";
    const details = [];
    if (typeof candidate.distance_m === "number" && candidate.match === "parcel" && candidate.distance_m > 0) {
      details.push(Math.round(candidate.distance_m) + " m from the map position");
    }
    if (candidate.note) details.push(candidate.note);
    return el("label", { class: "choice", for: id },
      el("input", { type: "radio", name: "candidate", id, value: candidate.candidate_id,
                    checked: candidate.candidate_id === state.chosenId, disabled: Boolean(state.pending),
                    onchange: () => dispatch({ type: "choose", candidateId: candidate.candidate_id }) }),
      el("span", { class: "choice-body" },
        el("span", { class: "choice-title", text: title }),
        el("span", { class: "city " + (candidate.supported ? "is-supported" : "is-unsupported"),
                     text: place + (candidate.supported ? ": supported city" : ": not supported") }),
        details.length ? el("span", { class: "choice-note", text: details.join(". ") }) : null,
      ),
    );
  }

  function typeChoice(value, label) {
    const id = "type-" + value;
    return el("label", { class: "choice", for: id },
      el("input", { type: "radio", name: "ptype", id, value, checked: state.propertyType === value,
                    disabled: Boolean(state.pending),
                    onchange: () => dispatch({ type: "property_type", value }) }),
      el("span", { class: "choice-body" }, el("span", { class: "choice-title", text: label })),
    );
  }

  function manualForm(open) {
    const form = el("form", { class: "manual", onsubmit: submitManual },
      el("label", { for: "manual-input", text: "Street address with city and state, or a map pin as latitude, longitude" }),
      el("div", { class: "manual-row" },
        el("input", { type: "text", id: "manual-input", value: ui.manualText, autocomplete: "off", maxlength: "200",
                      placeholder: "209 N 41st St, Seattle, WA",
                      oninput: (event) => { ui.manualText = event.target.value; } }),
        el("button", { type: "submit", id: "manual-submit", class: "button", disabled: Boolean(state.pending) }, "Find"),
      ),
      el("p", { class: "hint", text: "You never need to enter bedrooms, bathrooms or floor area." }),
    );
    if (open) return form;
    return el("details", { class: "manual-details", open: ui.manualOpen,
                           ontoggle: (event) => { ui.manualOpen = event.target.open; } },
      el("summary", { id: "manual-summary", text: "Not the right house? Enter an address" }), form);
  }

  function propertySection() {
    const section = el("section", { class: "card", "aria-labelledby": "property-heading" },
      el("h2", { id: "property-heading", text: "Property" }));
    if (state.candidates.length === 0) {
      section.append(manualForm(true));
      return section;
    }
    section.append(el("fieldset", {},
      el("legend", { text: state.candidates.length > 1 ? "Which house do you mean?" : "Is this the house?" }),
      state.candidates.map(candidateRow)));
    const chosen = chosenCandidate(state);
    if (chosen && chosen.supported) {
      section.append(el("fieldset", {},
        el("legend", { text: "What kind of property is it?" }),
        typeChoice(HOUSE_TYPE, "Detached single-family house"),
        typeChoice("other", "Something else (condo, apartment, townhouse, duplex, land)")));
      section.append(el("label", { class: "check", for: "use-image" },
        el("input", { type: "checkbox", id: "use-image", checked: ui.useImage, disabled: Boolean(state.pending),
                      onchange: (event) => { ui.useImage = event.target.checked; } }),
        el("span", { text: "Use a licensed street photo of the house when one exists" })));
      section.append(el("button", { type: "button", id: "estimate", class: "button primary",
                                    disabled: !canEstimate(state),
                                    onclick: () => dispatch({ type: "estimate", useImage: ui.useImage }) },
        state.pending && state.pending.kind === "estimate" ? "Estimating..." : (visibleResult(state) ? "Estimate again" : "Estimate")));
      if (!state.propertyType) {
        section.append(el("p", { class: "hint", text: "Choose the property type to enable the estimate." }));
      }
    }
    section.append(manualForm(false));
    return section;
  }

  function priceCard(card) {
    const ok = card.status === "ok";
    const range = ok ? formatRange(card.interval, card.unit) : null;
    return el("div", { class: "price" + (ok ? "" : " is-missing") },
      el("h4", { text: card.key.endsWith("unfurnished") ? "Without furniture" : "With furniture" }),
      el("p", { class: "amount", "aria-label": ok ? formatAmount(card.amount_usd, card.unit) + ", " + unitLabel(card.unit) : "Unavailable" },
        el("span", { class: "figure", text: ok ? formatUsd(card.amount_usd) : "Unavailable" }),
        ok && card.unit === "USD/month" ? el("span", { class: "per", text: "/month" }) : null),
      el("p", { class: "unit", text: ok ? unitLabel(card.unit) : "" }),
      range
        ? el("p", { class: "range", text: "Range " + range + ". " + formatPercent(card.interval.measured_coverage)
            + " of test houses fell inside a range this wide (target " + formatPercent(card.interval.target_coverage) + ")." })
        : el("p", { class: "range is-none", text: card.interval_note }),
      el("p", { class: "tags" },
        el("span", { class: "tag tag-" + card.kind, text: KIND_TAG[card.kind] || card.kind }),
        el("span", { class: "tag", text: BASIS_TAG[card.basis] || card.basis })),
      el("p", { class: "explain", text: card.explanation }),
    );
  }

  function photoBlock(result) {
    const image = result.image;
    if (image.status !== "used") {
      return el("p", { class: "note", text: image.message });
    }
    const credit = "Photo: " + (image.provider === "mapillary" ? "Mapillary" : image.provider)
      + (image.creator ? ", by " + image.creator : "") + ", " + (image.license || "")
      + (image.captured_at ? ", taken " + formatDate(image.captured_at) : "")
      + (typeof image.distance_m === "number" ? ", about " + Math.round(image.distance_m) + " m from the house" : "") + ".";
    return el("figure", { class: "photo" },
      image.thumb_url ? el("img", { src: image.thumb_url, alt: "Street photo that was analyzed for this estimate", loading: "lazy" }) : null,
      el("figcaption", {},
        el("span", { text: credit + " " }),
        image.page_url ? el("a", { href: image.page_url, target: "_blank", rel: "noopener noreferrer", text: "Open on Mapillary" }) : null),
      el("button", { type: "button", id: "not-my-house", class: "button",
                     onclick: () => { ui.useImage = false; dispatch({ type: "estimate", useImage: false }); } },
        "This photo is not my house"),
    );
  }

  function furnitureBlock(pack) {
    if (!pack) return null;
    return el("details", { class: "more" },
      el("summary", { text: "Furniture assumptions" }),
      el("p", { text: "A photo of the outside cannot show furniture. Both furnished numbers add the same fixed package. It is an assumption, not something measured for this house." }),
      el("dl", {},
        el("dt", { text: "Package" }), el("dd", { text: pack.package_name + " (version " + pack.package_version + ")" }),
        el("dt", { text: "Condition" }), el("dd", { text: pack.condition }),
        el("dt", { text: "Assumed cost when new" }), el("dd", { text: formatUsd(pack.new_cost_usd) }),
        el("dt", { text: "Added to the purchase price" }),
        el("dd", { text: formatUsd(pack.resale_value_usd) + " (resale at " + formatPercent(pack.resale_fraction) + " of new)" }),
        el("dt", { text: "Added to the monthly rent" }),
        el("dd", { text: formatUsd(pack.monthly_amount_usd) + "/month (cost when new spread over " + pack.rent_spread_months + " months)" })),
      el("p", { class: "hint", text: pack.fixed_inventory_note }),
      el("ul", { class: "items" }, pack.items.map((item) => el("li", { text: item.quantity + " x " + item.name }))),
    );
  }

  function aboutBlock(result) {
    return el("details", { class: "more" },
      el("summary", { text: "About this estimate" }),
      el("dl", {},
        el("dt", { text: "Estimate date" }), el("dd", { text: formatDate(result.estimate_date) }),
        el("dt", { text: "Result type" }),
        el("dd", { text: result.is_demo ? "Demo: synthetic data, not a real valuation" : "Trained on real data, plus furniture scenario" }),
        el("dt", { text: "Buy price data" }), el("dd", { text: result.data_period.sale || "None" }),
        el("dt", { text: "Rent data" }), el("dd", { text: result.data_period.rent || "None" }),
        el("dt", { text: "Model version" }), el("dd", { class: "mono", text: result.model_version || "None" })),
      el("ul", { class: "sources" }, result.attributions.map((line) => el("li", { text: line }))),
    );
  }

  function resultsSection() {
    const result = visibleResult(state);
    if (!result) return null;
    const byKey = Object.fromEntries(result.estimates.map((card) => [card.key, card]));
    const property = result.property;
    return el("section", { class: "card results" + (state.pending ? " is-busy" : ""), "aria-labelledby": "results-heading",
                           "aria-busy": state.pending ? "true" : "false" },
      el("h2", { id: "results-heading", text: "Estimates" }),
      result.is_demo ? el("p", { class: "demo-banner", text: "Demo mode. These numbers come from synthetic data and are not real valuations." }) : null,
      el("p", { class: "address", text: property.address || ("Map pin at " + property.lat.toFixed(5) + ", " + property.lng.toFixed(5)) }),
      el("p", { class: "city is-supported", text: (property.city_name || "") + ": supported city" }),
      el("p", { class: "mode", text: MODE_TEXT[result.prediction_mode] || "" }),
      el("h3", { text: "Buy" }),
      el("div", { class: "pair" }, priceCard(byKey.buy_unfurnished), priceCard(byKey.buy_furnished)),
      el("h3", { text: "Rent" }),
      el("div", { class: "pair" }, priceCard(byKey.rent_unfurnished), priceCard(byKey.rent_furnished)),
      photoBlock(result),
      furnitureBlock(result.furniture),
      aboutBlock(result),
      el("h3", { text: "Limitations" }),
      el("ul", { class: "limits" }, result.limitations.map((line) => el("li", { text: line }))),
    );
  }

  function render() {
    const focusedId = document.activeElement && document.activeElement.id;
    const scroll = root.scrollTop;
    const parts = [
      header(),
      messageLine(),
      state.status === "loading" ? el("div", { class: "loading", "aria-hidden": "true" }, el("span", { class: "bar" })) : null,
      propertySection(),
      resultsSection(),
      el("footer", { class: "foot" },
        el("p", { text: "Estimates only. Not an appraisal, a forecast or a guarantee. Supported: detached single-family homes in Seattle, San Francisco and Portland." })),
    ];
    root.replaceChildren(...parts.filter((part) => part !== null));
    root.scrollTop = scroll;
    if (focusedId) {
      const again = document.getElementById(focusedId);
      if (again) again.focus();
    }
  }

  maps.onUrlChange(onUrl);
  render();
  checkBackend();
  maps.getActiveUrl().then(onUrl);

  return { dispatch, getState: () => state };
}
