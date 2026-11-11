// The backend client: unreachable backend, timeouts, error bodies.
import assert from "node:assert/strict";
import { test } from "node:test";

import { ApiError, createApi } from "../lib/api.js";

function jsonResponse(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

test("a successful call posts JSON to the right path", async () => {
  const seen = [];
  const api = createApi({ baseUrl: "http://backend", fetchFn: async (url, options) => {
    seen.push({ url, options });
    return jsonResponse(200, { status: "ok" });
  } });
  assert.deepEqual(await api.resolve({ address: "1 A St" }), { status: "ok" });
  assert.equal(seen[0].url, "http://backend/resolve");
  assert.equal(seen[0].options.method, "POST");
  assert.equal(seen[0].options.body, JSON.stringify({ address: "1 A St" }));
  await api.health();
  assert.equal(seen[1].options.method, "GET");
});

test("backend unavailable is reported as a network error", async () => {
  const api = createApi({ fetchFn: async () => { throw new TypeError("Failed to fetch"); } });
  await assert.rejects(api.estimate({}), (error) => error instanceof ApiError && error.kind === "network");
});

test("a backend that never answers is reported as a timeout", async () => {
  const hanging = (url, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener("abort", () => {
      const error = new Error("aborted");
      error.name = "AbortError";
      reject(error);
    });
  });
  const api = createApi({ fetchFn: hanging, timeoutMs: 20 });
  await assert.rejects(api.estimate({}), (error) => error.kind === "timeout");
});

test("error statuses carry the backend's code and message", async () => {
  const api = createApi({ fetchFn: async () => jsonResponse(429, { error: { code: "rate_limited", message: "Too many requests." } }) });
  await assert.rejects(api.estimate({}), (error) =>
    error.kind === "http" && error.status === 429 && error.code === "rate_limited" && error.message === "Too many requests.");
});

test("an answer that is not JSON is refused", async () => {
  const api = createApi({ fetchFn: async () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError("bad"); } }) });
  await assert.rejects(api.health(), (error) => error.kind === "invalid");
});
