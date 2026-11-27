// The side panel: draws the state and sends user actions to the state rules in lib/state.js.
//
// startPanel() is given three things so the same code runs in Chrome and in the
// local preview page:
//   root  the page element to draw into
//   api   the backend client from lib/api.js
//   maps  { getActiveUrl(): Promise<string|null>, onUrlChange(callback) }

import { parseLatLngText, parseMapsUrl } from "./lib/maps_url.js";
import { formatAmount, formatDate, formatPercent, formatRange, formatUsd } from "./lib/format.js";
import { BADGE_LABELS, HOUSE_TYPE, STATUS_LABELS, badges, chosenCandidate, initialState, reduce,
         visibleResult } from "./lib/state.js";

const STATUS_TONE = {
  select_house: "neutral", confirm_property: "info", loading: "info", estimate_ready: "good",
  location_only: "good", unsupported_location: "warn", unsupported_property: "warn",
  insufficient_data: "warn", backend_unavailable: "bad", request_failed: "bad",
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
  const ui = { manualText: "", manualOpen: false };

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
    if (badges(state).includes("demo")) {
      pills.push(el("span", { class: "pill tone-demo", text: BADGE_LABELS.demo }));
    }
    return el("header", { class: "top" },
      el("div", { class: "brand" }, logo(), el("h1", { text: "HomeLens" })),
      el("div", { class: "pills", role: "status", "aria-live": "polite" }, pills),
    );
  }

  // Short status text. Hidden once numbers are on screen, unless something went wrong.
  function messageLine() {
    const failed = state.status === "backend_unavailable" || state.status === "request_failed";
    if (!state.message || (visibleResult(state) && !failed) || state.status === "unsupported_property") return null;
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

  // Several possible houses (a Street View position or an address with no county match): pick one.
  function pickSection() {
    if (state.status !== "confirm_property" || state.chosenId || state.candidates.length === 0) return null;
    return el("section", { class: "card", "aria-labelledby": "pick-heading" },
      el("h2", { id: "pick-heading", text: "Which house do you mean?" }),
      el("div", { class: "picks" }, state.candidates.map((c) => {
        const title = c.address || ("Map pin at " + c.lat.toFixed(5) + ", " + c.lng.toFixed(5));
        const far = typeof c.distance_m === "number" && c.distance_m > 0 ? Math.round(c.distance_m) + " m away" : "";
        return el("button", { type: "button", class: "pick", id: "pick-" + c.candidate_id,
                              disabled: Boolean(state.pending) || !c.supported,
                              onclick: () => dispatch({ type: "choose", candidateId: c.candidate_id }) },
          el("span", { class: "pick-title", text: title }),
          el("span", { class: "pick-note", text: c.supported ? far : (c.city_name || "This area") + " is not supported" }));
      })),
    );
  }

  function notHouseSection() {
    if (state.status !== "unsupported_property") return null;
    const chosen = chosenCandidate(state);
    return el("section", { class: "card" },
      chosen && chosen.address ? el("p", { class: "address", text: chosen.address }) : null,
      el("p", { text: "HomeLens only prices detached single-family houses, so no estimate is shown." }),
      el("button", { type: "button", id: "is-house", class: "button",
                     onclick: () => dispatch({ type: "property_type", value: HOUSE_TYPE }) }, "It is a detached house"),
    );
  }

  function searchBox(open) {
    const form = el("form", { class: "manual", onsubmit: submitManual },
      el("div", { class: "manual-row" },
        el("input", { type: "text", id: "manual-input", value: ui.manualText, autocomplete: "off", maxlength: "200",
                      "aria-label": "Street address with city and state", placeholder: "209 N 41st St, Seattle, WA",
                      oninput: (event) => { ui.manualText = event.target.value; } }),
        el("button", { type: "submit", id: "manual-submit", class: "button", disabled: Boolean(state.pending) }, "Find"),
      ),
    );
    if (open) return el("section", { class: "card" }, el("h2", { text: "Search an address" }), form);
    return el("details", { class: "manual-details", open: ui.manualOpen,
                           ontoggle: (event) => { ui.manualOpen = event.target.open; } },
      el("summary", { id: "manual-summary", text: "Wrong house? Search an address" }), form);
  }

  const TILE_LABEL = {
    buy_unfurnished: "Buy", buy_furnished: "Buy, furnished", rent_unfurnished: "Rent", rent_furnished: "Rent, furnished",
  };

  function tileNote(card) {
    if (card.status !== "ok") return "Unavailable";
    if (card.basis === "assumption_added") return "Assumed furniture added";
    if (card.interval) return "Likely " + formatRange(card.interval, card.unit);
    if (card.basis === "area_level_model") return "Area average";
    return "";
  }

  function tile(card) {
    const ok = card.status === "ok";
    return el("div", { class: "tile" + (ok ? "" : " is-missing") },
      el("span", { class: "tile-label", text: TILE_LABEL[card.key] }),
      el("span", { class: "tile-figure", "aria-label": ok ? formatAmount(card.amount_usd, card.unit) : "Unavailable" },
        ok ? formatUsd(card.amount_usd) : "n/a",
        ok && card.unit === "USD/month" ? el("span", { class: "per", text: "/mo" }) : null),
      el("span", { class: "tile-note", text: tileNote(card) }),
    );
  }

  function photoPart(result) {
    const image = result.image;
    if (image.status !== "used") {
      return el("p", { class: "hint", text: image.message || "No street photo was used." });
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
      image.message ? el("p", { class: "hint", text: image.message }) : null,
      el("button", { type: "button", id: "not-my-house", class: "button small",
                     onclick: () => dispatch({ type: "estimate", useImage: false }) },
        "This photo is not my house"),
    );
  }

  function detailsPart(result, cards) {
    const pack = result.furniture;
    return el("details", { class: "more", id: "details" },
      el("summary", { text: "Details" }),
      el("h3", { text: "How each number was made" }),
      el("dl", {}, cards.flatMap((card) => [
        el("dt", { text: TILE_LABEL[card.key] }),
        el("dd", {}, el("p", { text: card.explanation }),
          el("p", { class: "hint", text: card.interval
            ? "Range " + formatRange(card.interval, card.unit) + ": " + formatPercent(card.interval.measured_coverage)
              + " of test houses sold inside a range this wide (target " + formatPercent(card.interval.target_coverage) + ")."
            : card.interval_note })),
      ])),
      el("h3", { text: "Street photo" }),
      photoPart(result),
      pack ? el("h3", { text: "Furniture assumption" }) : null,
      pack ? el("p", { class: "hint", text: "One fixed package (" + pack.package_name + "): " + formatUsd(pack.new_cost_usd)
        + " when new. Buy adds its resale value, " + formatUsd(pack.resale_value_usd) + ". Rent adds "
        + formatUsd(pack.monthly_amount_usd) + "/month, the cost spread over " + pack.rent_spread_months
        + " months. An assumption, not measured for this house." }) : null,
      el("h3", { text: "Limitations" }),
      el("ul", { class: "limits" }, result.limitations.map((line) => el("li", { text: line }))),
      el("h3", { text: "Data and model" }),
      el("ul", { class: "sources" },
        el("li", { text: "Buy price data: " + (result.data_period.sale || "none") }),
        el("li", { text: "Rent data: " + (result.data_period.rent || "none") }),
        el("li", { text: "Estimated " + formatDate(result.estimate_date) + ", model " + (result.model_version || "none") }),
        result.attributions.map((line) => el("li", { text: line }))),
    );
  }

  function resultsSection() {
    const result = visibleResult(state);
    if (!result) return null;
    const byKey = Object.fromEntries(result.estimates.map((card) => [card.key, card]));
    const order = ["buy_unfurnished", "buy_furnished", "rent_unfurnished", "rent_furnished"].filter((k) => byKey[k]);
    const property = result.property;
    const buy = byKey.buy_unfurnished;
    const scope = buy && buy.basis === "per_house_model" ? "priced for this house" : "area average for this spot";
    return el("section", { class: "card results" + (state.pending ? " is-busy" : ""), "aria-label": "Estimates",
                           "aria-busy": state.pending ? "true" : "false" },
      result.is_demo ? el("p", { class: "demo-banner", text: "Demo mode: synthetic data, not real valuations." }) : null,
      el("div", { class: "result-head" },
        el("div", {},
          el("p", { class: "address", text: property.address || ("Map pin at " + property.lat.toFixed(5) + ", " + property.lng.toFixed(5)) }),
          el("p", { class: "subtle", text: (property.city_name || "") + " · " + scope })),
        el("button", { type: "button", id: "not-a-house", class: "linkish", disabled: Boolean(state.pending),
                       onclick: () => dispatch({ type: "property_type", value: "other" }) }, "Not a house?")),
      el("div", { class: "tiles" }, order.map((key) => tile(byKey[key]))),
      detailsPart(result, order.map((key) => byKey[key])),
    );
  }

  function render() {
    const focusedId = document.activeElement && document.activeElement.id;
    const scroll = root.scrollTop;
    const noHouseYet = state.candidates.length === 0 && state.status !== "loading";
    const parts = [
      header(),
      messageLine(),
      state.status === "loading" ? el("div", { class: "loading", "aria-hidden": "true" }, el("span", { class: "bar" })) : null,
      pickSection(),
      notHouseSection(),
      resultsSection(),
      searchBox(noHouseYet),
      el("footer", { class: "foot" }, el("p", { text: "Estimates only, not an appraisal." })),
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
