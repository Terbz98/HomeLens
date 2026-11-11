// Text formatting for money, units and dates. Every amount is in US dollars.

const USD = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export function formatUsd(amount) {
  if (typeof amount !== "number" || !Number.isFinite(amount)) {
    return "Unavailable";
  }
  return USD.format(Math.round(amount));
}

// unit is "USD" (a total price) or "USD/month" (a monthly rent).
export function formatAmount(amount, unit) {
  const text = formatUsd(amount);
  if (text === "Unavailable") {
    return text;
  }
  return unit === "USD/month" ? text + "/month" : text;
}

export function unitLabel(unit) {
  return unit === "USD/month" ? "US dollars per month" : "US dollars";
}

export function formatRange(interval, unit) {
  if (!interval || typeof interval.low_usd !== "number" || typeof interval.high_usd !== "number") {
    return null;
  }
  return formatUsd(interval.low_usd) + " to " + formatAmount(interval.high_usd, unit);
}

export function formatPercent(fraction) {
  if (typeof fraction !== "number" || !Number.isFinite(fraction)) {
    return "";
  }
  return Math.round(fraction * 100) + "%";
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// "2026-11-11" -> "Nov 11, 2026". Anything else is returned unchanged.
export function formatDate(isoDate) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(isoDate || "");
  if (!match) {
    return isoDate || "";
  }
  const month = MONTHS[Number(match[2]) - 1];
  return month ? month + " " + Number(match[3]) + ", " + match[1] : isoDate;
}
