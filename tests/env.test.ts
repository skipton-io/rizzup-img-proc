import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { loadDotEnv } from "../src/env";

test("loadDotEnv is a no-op when the file is missing", () => {
  const before = process.env.RIZZUP_TEST_MISSING_ENV;
  loadDotEnv(path.join(os.tmpdir(), "missing-rizzup-dotenv-file.env"));
  assert.equal(process.env.RIZZUP_TEST_MISSING_ENV, before);
});

test("loadDotEnv ignores malformed lines, unwraps quotes, and does not overwrite existing env vars", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "rizzup-dotenv-test-"));
  const filePath = path.join(tempDir, ".env");
  const originalExisting = process.env.RIZZUP_EXISTING_VALUE;
  const originalQuoted = process.env.RIZZUP_QUOTED_VALUE;
  const originalPlain = process.env.RIZZUP_PLAIN_VALUE;
  delete process.env.RIZZUP_QUOTED_VALUE;
  delete process.env.RIZZUP_PLAIN_VALUE;
  process.env.RIZZUP_EXISTING_VALUE = "keep-me";

  await fs.writeFile(
    filePath,
    [
      "# comment",
      "MALFORMED_LINE",
      "RIZZUP_EXISTING_VALUE=replace-me",
      "RIZZUP_QUOTED_VALUE=\"quoted value\"",
      "RIZZUP_PLAIN_VALUE=plain",
      ""
    ].join("\n"),
    "utf8"
  );

  loadDotEnv(filePath);

  assert.equal(process.env.RIZZUP_EXISTING_VALUE, "keep-me");
  assert.equal(process.env.RIZZUP_QUOTED_VALUE, "quoted value");
  assert.equal(process.env.RIZZUP_PLAIN_VALUE, "plain");

  if (originalExisting === undefined) {
    delete process.env.RIZZUP_EXISTING_VALUE;
  } else {
    process.env.RIZZUP_EXISTING_VALUE = originalExisting;
  }
  if (originalQuoted === undefined) {
    delete process.env.RIZZUP_QUOTED_VALUE;
  } else {
    process.env.RIZZUP_QUOTED_VALUE = originalQuoted;
  }
  if (originalPlain === undefined) {
    delete process.env.RIZZUP_PLAIN_VALUE;
  } else {
    process.env.RIZZUP_PLAIN_VALUE = originalPlain;
  }
});
