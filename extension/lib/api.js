// Talks to the HomeLens backend. No secrets live in the extension.
//
// Every failure is turned into an ApiError with a "kind":
//   network  the backend could not be reached at all
//   timeout  no answer in time
//   http     the backend answered with an error status
//   invalid  the backend answered with something that is not the expected JSON

export const DEFAULT_BASE_URL = "http://127.0.0.1:8000";
export const DEFAULT_TIMEOUT_MS = 30000;

export class ApiError extends Error {
  constructor(kind, message, status = null, code = null) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.code = code;
  }
}

export function createApi({ baseUrl = DEFAULT_BASE_URL, fetchFn = fetch, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  async function request(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let response;
    try {
      response = await fetchFn(baseUrl + path, {
        method: body === undefined ? "GET" : "POST",
        headers: body === undefined ? {} : { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (error) {
      if (error && error.name === "AbortError") {
        throw new ApiError("timeout", "The backend did not answer in time.");
      }
      throw new ApiError("network", "The backend could not be reached.");
    } finally {
      clearTimeout(timer);
    }
    let data = null;
    try {
      data = await response.json();
    } catch (error) {
      data = null;
    }
    if (!response.ok) {
      const detail = data && data.error ? data.error : {};
      throw new ApiError("http", detail.message || "The backend refused the request.", response.status, detail.code || null);
    }
    if (data === null || typeof data !== "object") {
      throw new ApiError("invalid", "The backend sent an answer that could not be read.");
    }
    return data;
  }

  return {
    health: () => request("/health"),
    cities: () => request("/cities"),
    resolve: (body) => request("/resolve", body),
    estimate: (body) => request("/estimate", body),
  };
}
