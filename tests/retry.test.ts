import test from "node:test";
import assert from "node:assert/strict";
import { calculateRetryDelay } from "../src/retry";

test("calculateRetryDelay doubles until the configured cap", () => {
  assert.equal(calculateRetryDelay(1, 5_000, 60_000), 5_000);
  assert.equal(calculateRetryDelay(2, 5_000, 60_000), 10_000);
  assert.equal(calculateRetryDelay(3, 5_000, 60_000), 20_000);
  assert.equal(calculateRetryDelay(6, 5_000, 60_000), 60_000);
});

test("calculateRetryDelay clamps attempts below one and never exceeds the configured max", () => {
  assert.equal(calculateRetryDelay(0, 5_000, 60_000), 5_000);
  assert.equal(calculateRetryDelay(-4, 5_000, 60_000), 5_000);
  assert.equal(calculateRetryDelay(99, 5_000, 60_000), 60_000);
});
