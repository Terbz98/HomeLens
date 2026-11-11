// Developer preview of the side panel in a normal browser tab (no Chrome extension APIs).
// Serve the extension folder with any static file server and open dev/preview.html.
import { createApi } from "../lib/api.js";
import { startPanel } from "../panel.js";
import { SAMPLE_URLS } from "./sample_urls.js";

const select = document.getElementById("sample");
for (const [label, url] of Object.entries(SAMPLE_URLS)) {
  const option = document.createElement("option");
  option.value = url;
  option.textContent = label;
  select.append(option);
}

const params = new URLSearchParams(location.search);
if (params.has("sample")) {
  select.selectedIndex = Number(params.get("sample")) || 0;
}

let listener = () => {};
select.addEventListener("change", () => listener(select.value));

window.homelens = startPanel({
  root: document.getElementById("app"),
  api: createApi({ baseUrl: params.get("api") || "http://127.0.0.1:8000" }),
  maps: {
    getActiveUrl: async () => select.value,
    onUrlChange: (callback) => { listener = callback; },
  },
});
