// Money, unit and date formatting.
import assert from "node:assert/strict";
import { test } from "node:test";

import { formatAmount, formatDate, formatPercent, formatRange, formatUsd, unitLabel } from "../lib/format.js";

test("US dollar formatting has a dollar sign, commas and no cents", () => {
  assert.equal(formatUsd(1166000), "$1,166,000");
  assert.equal(formatUsd(3540), "$3,540");
  assert.equal(formatUsd(2187.5), "$2,188");
  assert.equal(formatUsd(0), "$0");
});

test("missing or impossible amounts are shown as Unavailable, never as a number", () => {
  for (const bad of [null, undefined, NaN, Infinity, "1000", {}]) {
    assert.equal(formatUsd(bad), "Unavailable");
    assert.equal(formatAmount(bad, "USD/month"), "Unavailable");
  }
});

test("rent always carries an explicit monthly unit and purchase never does", () => {
  assert.equal(formatAmount(3540, "USD/month"), "$3,540/month");
  assert.equal(formatAmount(1166000, "USD"), "$1,166,000");
  assert.equal(unitLabel("USD/month"), "US dollars per month");
  assert.equal(unitLabel("USD"), "US dollars");
});

test("ranges, percentages and dates", () => {
  assert.equal(formatRange({ low_usd: 723000, high_usd: 1609000 }, "USD"), "$723,000 to $1,609,000");
  assert.equal(formatRange({ low_usd: 3000, high_usd: 4000 }, "USD/month"), "$3,000 to $4,000/month");
  assert.equal(formatRange(null, "USD"), null);
  assert.equal(formatRange({ low_usd: "a" }, "USD"), null);
  assert.equal(formatPercent(0.8001), "80%");
  assert.equal(formatPercent(undefined), "");
  assert.equal(formatDate("2026-11-11"), "Nov 11, 2026");
  assert.equal(formatDate("garbage"), "garbage");
});
