// Connects the side panel to Chrome: which tab is in front, and what its Google Maps URL is.
import { createApi } from "./lib/api.js";
import { startPanel } from "./panel.js";

async function activeTabId() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs.length ? tabs[0].id : null;
}

// Ask the content script in the front tab for its URL. Tabs that are not
// Google Maps have no content script, so the answer is null.
async function getActiveUrl() {
  try {
    const tabId = await activeTabId();
    if (tabId === null) return null;
    const answer = await chrome.tabs.sendMessage(tabId, { type: "homelens:get-url" });
    return answer && typeof answer.url === "string" ? answer.url : null;
  } catch (error) {
    return null;
  }
}

function onUrlChange(callback) {
  // A Google Maps tab reports a new URL. It only counts if that tab is the one in front.
  chrome.runtime.onMessage.addListener((message, sender) => {
    if (!message || message.type !== "homelens:url" || !sender.tab) return;
    activeTabId().then((tabId) => {
      if (tabId === sender.tab.id) callback(message.url);
    });
  });
  // The user switched tabs: read the new front tab.
  chrome.tabs.onActivated.addListener(() => {
    getActiveUrl().then(callback);
  });
}

startPanel({
  root: document.getElementById("app"),
  api: createApi(),
  maps: { getActiveUrl, onUrlChange },
});
