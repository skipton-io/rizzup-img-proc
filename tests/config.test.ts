import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { loadConfig } from "../src/config";

const cwd = process.cwd();

function baseEnv(): NodeJS.ProcessEnv {
  return {
    NETLIFY_SITE_ID: "site_123",
    NETLIFY_ACCESS_TOKEN: "token_123"
  };
}

test("loadConfig defaults to the local archive backend", () => {
  const config = loadConfig(baseEnv());

  assert.equal(config.archiveBackend, "local");
  assert.equal(config.sftpPort, 22);
  assert.equal(config.sftpStrictHostKey, false);
  assert.equal(config.sourceImageRoot, undefined);
  assert.equal(config.localRenderRoot, path.resolve(cwd, "artifacts", "renders"));
});

test("loadConfig derives SFTP archive settings and local worker paths", () => {
  const config = loadConfig({
    ...baseEnv(),
    RIZZUP_ARCHIVE_BACKEND: "sftp",
    RIZZUP_IMAGE_ARCHIVE_ROOT: "/volume1/rizzup-image-jobs",
    RIZZUP_SFTP_HOST: "100.64.0.10",
    RIZZUP_SFTP_PORT: "2222",
    RIZZUP_SFTP_USERNAME: "rizzup-archive",
    RIZZUP_SFTP_PASSWORD: "secret",
    RIZZUP_SFTP_STRICT_HOST_KEY: "true",
    RIZZUP_SFTP_HOST_KEY: "SHA256:abc123",
    RIZZUP_RESULTS_DIR: "tmp-artifacts"
  });

  assert.equal(config.archiveBackend, "sftp");
  assert.equal(config.imageArchiveRoot, "/volume1/rizzup-image-jobs");
  assert.equal(config.sftpHost, "100.64.0.10");
  assert.equal(config.sftpPort, 2222);
  assert.equal(config.sftpUsername, "rizzup-archive");
  assert.equal(config.sftpPassword, "secret");
  assert.equal(config.sftpStrictHostKey, true);
  assert.equal(config.sftpHostKey, "SHA256:abc123");
  assert.equal(config.sourceImageRoot, path.resolve(cwd, "tmp-artifacts", "source-images"));
  assert.equal(config.localRenderRoot, path.resolve(cwd, "tmp-artifacts", "renders"));
});
