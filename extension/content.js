// Runs only on Google Maps pages (see "matches" in manifest.json).
//
// It reads one thing: the page address (URL). It does not read the map
// picture, Street View imagery, search text or anything else on the page,
// and it sends the URL only to the HomeLens side panel of this extension.
// Google Maps changes its URL without reloading the page, so the URL is
// checked on a short timer.
(() => {
  let lastUrl = null;

  function report() {
    if (location.href === lastUrl) {
      return;
    }
    lastUrl = location.href;
    try {
      // Fails harmlessly when the side panel is closed.
      chrome.runtime.sendMessage({ type: "homelens:url", url: lastUrl }).catch(() => {});
    } catch (error) {
      // The extension was reloaded. This old script stops quietly.
      clearInterval(timer);
    }
  }

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message && message.type === "homelens:get-url") {
      sendResponse({ url: location.href });
    }
  });

  const timer = setInterval(report, 700);
  window.addEventListener("popstate", report);
  report();
})();
