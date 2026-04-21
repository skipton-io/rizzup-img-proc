import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import {
  LocalArchiveStorage,
  archiveRelativePathForJob,
  createArchiveStorage,
  localPathFromRelative
} from "../src/archiveStorage";
import { WorkerConfig } from "../src/types";

function baseConfig(): WorkerConfig {
  return {
    netlifySiteId: "site",
    netlifyAccessToken: "token",
    queueStore: "queue",
    statusStore: "status",
    resultsStore: "results",
    assetsStore: "assets",
    locksStore: "locks",
    deadLetterStore: "dead",
    maxRuntimeMs: 55_000,
    pollIntervalMs: 1,
    maxJobsPerPoll: 1,
    lockTtlMs: 60_000,
    maxAttempts: 3,
    retryBaseDelayMs: 1_000,
    retryMaxDelayMs: 5_000,
    workerId: "worker_test",
    previewWatermarkText: "RizzUp Preview",
    previewWatermarkLogoPath: path.resolve(process.cwd(), "logo.png"),
    resultsDir: "artifacts",
    archiveBackend: "local",
    imageArchiveRoot: path.resolve(process.cwd(), "artifacts", "archive"),
    sourceImageRoot: undefined,
    localRenderRoot: path.resolve(process.cwd(), "artifacts", "renders"),
    sftpHost: undefined,
    sftpPort: 22,
    sftpUsername: undefined,
    sftpPassword: undefined,
    sftpStrictHostKey: false,
    sftpHostKey: undefined,
    pythonExecutable: "python3",
    pythonScript: "scripts/gpu_pipeline.py",
    fireRedEnabled: true,
    fireRedModelId: "Tongyi-MAI/Z-Image-Turbo",
    fireRedPrompt: "Beautify this image",
    fireRedInferenceSteps: 30,
    fireRedTrueCfgScale: 4,
    analysisMaxSize: 100,
    previewMaxSize: 512,
    finalDecisionMaxSize: 512,
    finalMinWidth: 1024,
    finalMinHeight: 1280
  };
}

test("LocalArchiveStorage rejects empty relative paths", () => {
  const storage = new LocalArchiveStorage("/tmp/archive");

  assert.throws(() => storage.resolveArchivePath(""), /Invalid archive relative path/);
});

test("localPathFromRelative rejects traversal paths", () => {
  assert.throws(() => localPathFromRelative("/tmp/archive", "../outside.png"), /Invalid archive relative path/);
  assert.throws(() => localPathFromRelative("/tmp/archive", "nested/../../outside.png"), /Invalid archive relative path/);
});

test("createArchiveStorage rejects SFTP mode without credentials", () => {
  assert.throws(
    () =>
      createArchiveStorage({
        ...baseConfig(),
        archiveBackend: "sftp"
      }),
    /SFTP archive backend requires RIZZUP_SFTP_HOST, RIZZUP_SFTP_USERNAME, and RIZZUP_SFTP_PASSWORD/
  );
});

test("createArchiveStorage rejects strict host key mode without a host key", () => {
  assert.throws(
    () =>
      createArchiveStorage({
        ...baseConfig(),
        archiveBackend: "sftp",
        sftpHost: "100.64.0.10",
        sftpUsername: "rizzup",
        sftpPassword: "secret",
        sftpStrictHostKey: true
      }),
    /RIZZUP_SFTP_STRICT_HOST_KEY=true requires RIZZUP_SFTP_HOST_KEY/
  );
});

test("archiveRelativePathForJob falls back to the current date when createdAt is invalid", () => {
  const result = archiveRelativePathForJob("imgjob_123", "not-a-date");
  const now = new Date();
  const expectedPrefix = path.posix.join(
    String(now.getUTCFullYear()),
    String(now.getUTCMonth() + 1).padStart(2, "0"),
    String(now.getUTCDate()).padStart(2, "0")
  );

  assert.match(result, new RegExp(`^${expectedPrefix}/imgjob_123$`));
});
