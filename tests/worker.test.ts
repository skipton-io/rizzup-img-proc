import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createArchiveStorage } from "../src/archiveStorage";
import { listCandidateJobs, pollOnce, runWorker } from "../src/worker";
import { BlobStoreLike, WorkerConfig, WorkerStores } from "../src/types";

class MemoryStore implements BlobStoreLike {
  public readonly values = new Map<
    string,
    { data: unknown; etag: string; metadata?: Record<string, unknown> }
  >();

  async *list(options: { prefix: string; paginate: true }) {
    const blobs = [...this.values.keys()]
      .filter((key) => key.startsWith(options.prefix))
      .map((key) => ({ key, etag: this.values.get(key)?.etag }));

    yield { blobs };
  }

  async getWithMetadata<T>(
    key: string,
    _options: { type: "json" } | { type: "arrayBuffer" }
  ): Promise<{ data: T; etag?: string } | null> {
    const value = this.values.get(key);
    if (!value) return null;
    return { data: value.data as T, etag: value.etag };
  }

  async set(
    key: string,
    value: string | ArrayBuffer | Blob,
    options?: { metadata?: Record<string, unknown>; onlyIfMatch?: string; onlyIfNew?: boolean }
  ): Promise<{ modified: boolean; etag?: string }> {
    const existing = this.values.get(key);
    if (options?.onlyIfNew && existing) {
      return { modified: false };
    }

    if (options?.onlyIfMatch && existing?.etag !== options.onlyIfMatch) {
      return { modified: false };
    }

    const etag = `${Date.now()}-${Math.random()}`;
    this.values.set(key, { data: value, etag, metadata: options?.metadata });
    return { modified: true, etag };
  }

  async setJSON<T>(
    key: string,
    value: T,
    options?: {
      metadata?: Record<string, unknown>;
      onlyIfMatch?: string;
      onlyIfNew?: boolean;
    }
  ): Promise<{ modified: boolean; etag?: string }> {
    const existing = this.values.get(key);
    if (options?.onlyIfNew && existing) {
      return { modified: false };
    }

    if (options?.onlyIfMatch && existing?.etag !== options.onlyIfMatch) {
      return { modified: false };
    }

    const etag = `${Date.now()}-${Math.random()}`;
    this.values.set(key, { data: value, etag, metadata: options?.metadata });
    return { modified: true, etag };
  }

  async delete(key: string): Promise<void> {
    this.values.delete(key);
  }
}

function buildStores(): WorkerStores {
  return {
    queue: new MemoryStore(),
    status: new MemoryStore(),
    results: new MemoryStore(),
    assets: new MemoryStore(),
    locks: new MemoryStore(),
    deadLetter: new MemoryStore()
  };
}

function config(): WorkerConfig {
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
    maxJobsPerPoll: 4,
    lockTtlMs: 60_000,
    maxAttempts: 3,
    retryBaseDelayMs: 1_000,
    retryMaxDelayMs: 5_000,
    workerId: "worker_test",
    previewWatermarkText: "RizzUp Preview",
    previewWatermarkLogoPath: path.resolve(process.cwd(), "..", "rizzup.co.uk", "public", "brand", "rizzup-logo.png"),
    resultsDir: "artifacts",
    archiveBackend: "local",
    imageArchiveRoot: path.resolve(process.cwd(), "artifacts", "test-image-jobs"),
    localRenderRoot: path.resolve(process.cwd(), "artifacts", "test-renders"),
    sftpPort: 22,
    sftpStrictHostKey: false,
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

function archiveStorage(configValue: WorkerConfig) {
  return createArchiveStorage(configValue);
}

async function writeTempPythonScript(contents: string): Promise<string> {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "rizzup-worker-test-"));
  const scriptPath = path.join(directory, "script.py");
  await fs.writeFile(scriptPath, contents, "utf8");
  return scriptPath;
}

test("pollOnce processes upload_photo records into the results store", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.pythonScript = await writeTempPythonScript(`
import json
import sys
request = json.loads(sys.stdin.read())
assert request["action"] == "validate_upload"
json.dump({
  "faceDetection": {
    "box": {"x": 12, "y": 24, "w": 240, "h": 240},
    "landmarks": {
      "leftEye": [60, 80],
      "rightEye": [160, 80],
      "noseTip": [110, 130],
      "mouthCenter": [110, 180]
    },
    "debug": {"rawFaces": [[12, 24, 240, 240]], "rawEyes": []},
    "rotatedToPortrait": False
  }
}, sys.stdout)
`);
  await stores.queue.setJSON("upload_photo/2026-04-07-upload_123.json", {
    type: "upload_photo",
    queuedAt: "2026-04-07T12:00:00.000Z",
    payload: {
      uploadId: "upload_123",
      imageJobId: "imgjob_123",
      sourceName: "photo.jpg",
      mimeType: "image/jpeg",
      sizeBytes: 1234,
      createdAt: "2026-04-07T12:00:00.000Z",
      sourceDataUrl: "data:image/jpeg;base64,AA=="
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(processed, 1);

  const upload = await stores.results.getWithMetadata<{
    uploadId: string;
    sourceName: string;
    faceDetection?: { box: { x: number } };
  }>("upload_photo/upload_123.json", { type: "json" });

  assert.equal(upload?.data.uploadId, "upload_123");
  assert.equal(upload?.data.sourceName, "photo.jpg");
  assert.equal(typeof upload?.data.faceDetection?.box.x, "number");
});

test("pollOnce skips records whose notBefore is still in the future", async () => {
  const stores = buildStores();
  await stores.queue.setJSON("generate_preview/future.json", {
    type: "generate_preview",
    queuedAt: "2026-04-07T12:00:00.000Z",
    notBefore: "2099-01-01T00:00:00.000Z",
    payload: {
      uploadId: "upload_123",
      preset: "natural",
      requestedAt: "2026-04-07T12:00:00.000Z"
    }
  });

  const cfg = config();
  const processed = await pollOnce(cfg, stores, archiveStorage(cfg));
  assert.equal(processed, 0);
});

test("listCandidateJobs orders records by queuedAt even when queue keys are deterministic", async () => {
  const stores = buildStores();
  await stores.queue.setJSON("upload_photo/upload_123.json", {
    type: "upload_photo",
    queuedAt: "2026-04-07T12:00:03.000Z",
    payload: {
      uploadId: "upload_123",
      imageJobId: "imgjob_123",
      sourceName: "photo.jpg",
      mimeType: "image/jpeg",
      sizeBytes: 1234,
      createdAt: "2026-04-07T12:00:03.000Z",
      sourceDataUrl: "data:image/jpeg;base64,AA=="
    }
  });
  await stores.queue.setJSON("analyze_photo_quality/upload_123.json", {
    type: "analyze_photo_quality",
    queuedAt: "2026-04-07T12:00:04.000Z",
    payload: {
      uploadId: "upload_123",
      requestedAt: "2026-04-07T12:00:04.000Z"
    }
  });
  await stores.queue.setJSON("generate_preview/upload_123-natural.json", {
    type: "generate_preview",
    queuedAt: "2026-04-07T12:00:05.000Z",
    payload: {
      uploadId: "upload_123",
      preset: "natural",
      requestedAt: "2026-04-07T12:00:05.000Z"
    }
  });

  const jobs = await listCandidateJobs(stores.queue, 10);
  assert.deepEqual(jobs.map((job) => job.key), [
    "upload_photo/upload_123.json",
    "analyze_photo_quality/upload_123.json",
    "generate_preview/upload_123-natural.json"
  ]);
});

test("pollOnce processes newer jobs even when older completed records still exist", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.maxJobsPerPoll = 1;
  workerConfig.pythonScript = await writeTempPythonScript(`
import json
import sys
request = json.loads(sys.stdin.read())
assert request["action"] == "validate_upload"
json.dump({
  "faceDetection": {
    "box": {"x": 8, "y": 16, "w": 220, "h": 220},
    "landmarks": {
      "leftEye": [50, 70],
      "rightEye": [150, 70],
      "noseTip": [100, 120],
      "mouthCenter": [100, 165]
    },
    "debug": {"rawFaces": [[8, 16, 220, 220]], "rawEyes": []},
    "rotatedToPortrait": False
  }
}, sys.stdout)
`);

  await stores.queue.setJSON("upload_photo/2026-04-07T12-00-01-000Z-upload_old.json", {
    type: "upload_photo",
    queuedAt: "2026-04-07T12:00:01.000Z",
    payload: {
      uploadId: "upload_old",
      imageJobId: "imgjob_old",
      sourceName: "old.jpg",
      mimeType: "image/jpeg",
      sizeBytes: 1234,
      createdAt: "2026-04-07T12:00:01.000Z",
      sourceDataUrl: "data:image/jpeg;base64,AA=="
    }
  });
  await stores.status.setJSON(
    `status/${Buffer.from("upload_photo/2026-04-07T12-00-01-000Z-upload_old.json").toString("base64url")}.json`,
    {
      queueKey: "upload_photo/2026-04-07T12-00-01-000Z-upload_old.json",
      type: "upload_photo",
      workerId: "worker_test",
      status: "completed",
      attempts: 1,
      updatedAt: "2026-04-07T12:00:02.000Z"
    }
  );
  await stores.queue.setJSON("upload_photo/2026-04-07T12-00-02-000Z-upload_new.json", {
    type: "upload_photo",
    queuedAt: "2026-04-07T12:00:02.000Z",
    payload: {
      uploadId: "upload_new",
      imageJobId: "imgjob_new",
      sourceName: "new.jpg",
      mimeType: "image/jpeg",
      sizeBytes: 5678,
      createdAt: "2026-04-07T12:00:02.000Z",
      sourceDataUrl: "data:image/jpeg;base64,AA=="
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(processed, 1);

  const upload = await stores.results.getWithMetadata<{ uploadId: string }>(
    "upload_photo/upload_new.json",
    { type: "json" }
  );
  assert.equal(upload?.data.uploadId, "upload_new");
});

test("pollOnce deletes queue blobs after completed jobs are recorded", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.pythonScript = await writeTempPythonScript(`
import json
import sys
request = json.loads(sys.stdin.read())
assert request["action"] == "validate_upload"
json.dump({
  "faceDetection": {
    "box": {"x": 10, "y": 20, "w": 200, "h": 200},
    "landmarks": {
      "leftEye": [55, 75],
      "rightEye": [145, 75],
      "noseTip": [100, 120],
      "mouthCenter": [100, 160]
    },
    "debug": {"rawFaces": [[10, 20, 200, 200]], "rawEyes": []},
    "rotatedToPortrait": False
  }
}, sys.stdout)
`);
  const queueKey = "upload_photo/2026-04-07T12-00-01-000Z-upload_trim.json";
  await stores.queue.setJSON(queueKey, {
    type: "upload_photo",
    queuedAt: "2026-04-07T12:00:01.000Z",
    payload: {
      uploadId: "upload_trim",
      imageJobId: "imgjob_trim",
      sourceName: "trim.jpg",
      mimeType: "image/jpeg",
      sizeBytes: 1234,
      createdAt: "2026-04-07T12:00:01.000Z",
      sourceDataUrl: "data:image/jpeg;base64,AA=="
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(processed, 1);

  const queueRecord = await stores.queue.getWithMetadata(queueKey, { type: "json" });
  assert.equal(queueRecord, null);
});

test("pollOnce processes generate_final_image jobs into the final results store", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.pythonScript = await writeTempPythonScript(`
import json
import sys
from pathlib import Path
request = json.loads(sys.stdin.read())
assert request["action"] == "final"
Path(request["outputPath"]).parent.mkdir(parents=True, exist_ok=True)
Path(request["outputPath"]).write_bytes(b"final")
json.dump({
  "preset": request["preset"],
  "finalImagePath": request["outputPath"],
  "usedGpu": False,
  "identityGenerationUsed": False,
    "identityGenerationMode": "deterministic-enhancement",
    "identityFallbackReason": None,
  "rotatedToPortrait": False,
  "width": 1024,
  "height": 1280
}, sys.stdout)
`);
  await stores.results.setJSON("upload_photo/upload_unlock.json", {
    uploadId: "upload_unlock",
    imageJobId: "imgjob_unlock",
    sourceName: "photo.jpg",
    mimeType: "image/jpeg",
    sizeBytes: 1234,
    createdAt: "2026-04-07T12:00:00.000Z"
  });
  await stores.queue.setJSON("generate_final_image/2026-04-07T12-00-10-000Z-unlock_123.json", {
    type: "generate_final_image",
    queuedAt: "2026-04-07T12:00:10.000Z",
    payload: {
      unlockId: "unlock_123",
      checkoutSessionId: "checkout_123",
      uploadId: "upload_unlock",
      preset: "natural",
      plan: "5_photos",
      requestedAt: "2026-04-07T12:00:10.000Z"
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(processed, 1);

  const unlock = await stores.results.getWithMetadata<{ unlockId: string }>(
    "generate_final_image/unlock_123.json",
    { type: "json" }
  );
  assert.equal(unlock?.data.unlockId, "unlock_123");
});

test("pollOnce dead-letters face validation preview failures without retrying", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.pythonScript = await writeTempPythonScript(`
import json
import sys
json.dump({"code": "FACE_NOT_DETECTED", "message": "No face detected. Please upload a clear photo with one visible face.", "retryable": False}, sys.stderr)
sys.exit(1)
`);

  await stores.results.setJSON("upload_photo/upload_face.json", {
    uploadId: "upload_face",
    imageJobId: "imgjob_face",
    sourceName: "photo.jpg",
    mimeType: "image/jpeg",
    sizeBytes: 1234,
    createdAt: "2026-04-07T12:00:00.000Z",
    sourcePath: "source\\upload_face.jpg"
  });
  const queueKey = "generate_preview/2026-04-07T12-00-05-000Z-upload_face.json";
  await stores.queue.setJSON(queueKey, {
    type: "generate_preview",
    queuedAt: "2026-04-07T12:00:05.000Z",
    payload: {
      uploadId: "upload_face",
      preset: "natural",
      requestedAt: "2026-04-07T12:00:05.000Z"
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(processed, 1);

  const status = await stores.status.getWithMetadata<{
    status: string;
    error?: string;
    errorCode?: string;
  }>(
    `status/${Buffer.from(queueKey).toString("base64url")}.json`,
    { type: "json" }
  );
  assert.equal(status?.data.status, "dead_lettered");
  assert.equal(status?.data.errorCode, "FACE_NOT_DETECTED");
  assert.match(status?.data.error || "", /No face detected/);

  const retryKeys = [...(stores.queue as MemoryStore).values.keys()].filter((key) =>
    key.includes("attempt-2")
  );
  assert.equal(retryKeys.length, 0);

  const queueRecord = await stores.queue.getWithMetadata(queueKey, { type: "json" });
  assert.equal(queueRecord, null);
});

test("pollOnce still retries generic preview runtime failures", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.pythonScript = await writeTempPythonScript(`
import sys
sys.stderr.write("boom")
sys.exit(1)
`);

  await stores.results.setJSON("upload_photo/upload_retry.json", {
    uploadId: "upload_retry",
    imageJobId: "imgjob_retry",
    sourceName: "photo.jpg",
    mimeType: "image/jpeg",
    sizeBytes: 1234,
    createdAt: "2026-04-07T12:00:00.000Z",
    sourcePath: "source\\upload_retry.jpg"
  });
  const queueKey = "generate_preview/2026-04-07T12-00-05-000Z-upload_retry.json";
  await stores.queue.setJSON(queueKey, {
    type: "generate_preview",
    queuedAt: "2026-04-07T12:00:05.000Z",
    payload: {
      uploadId: "upload_retry",
      preset: "natural",
      requestedAt: "2026-04-07T12:00:05.000Z"
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(processed, 1);

  const status = await stores.status.getWithMetadata<{
    status: string;
    error?: string;
  }>(
    `status/${Buffer.from(queueKey).toString("base64url")}.json`,
    { type: "json" }
  );
  assert.equal(status?.data.status, "retry_scheduled");
  assert.match(status?.data.error || "", /Python pipeline exited with code 1/);

  const retryKeys = [...(stores.queue as MemoryStore).values.keys()].filter((key) =>
    key.includes("attempt-2")
  );
  assert.equal(retryKeys.length, 1);

  const queueRecord = await stores.queue.getWithMetadata(queueKey, { type: "json" });
  assert.equal(queueRecord, null);
});

test("preview retries remain unique per preset for the same upload", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.maxJobsPerPoll = 10;
  workerConfig.pythonScript = await writeTempPythonScript(`
import sys
sys.stderr.write("boom")
sys.exit(1)
`);

  await stores.results.setJSON("upload_photo/upload_retry.json", {
    uploadId: "upload_retry",
    imageJobId: "imgjob_retry",
    sourceName: "photo.jpg",
    mimeType: "image/jpeg",
    sizeBytes: 1234,
    createdAt: "2026-04-07T12:00:00.000Z",
    sourcePath: "source\\\\upload_retry.jpg"
  });
  await stores.queue.setJSON("generate_preview/upload_retry-natural.json", {
    type: "generate_preview",
    queuedAt: "2026-04-07T12:00:05.000Z",
    payload: {
      uploadId: "upload_retry",
      preset: "natural",
      requestedAt: "2026-04-07T12:00:05.000Z"
    }
  });
  await stores.queue.setJSON("generate_preview/upload_retry-travel.json", {
    type: "generate_preview",
    queuedAt: "2026-04-07T12:00:06.000Z",
    payload: {
      uploadId: "upload_retry",
      preset: "travel",
      requestedAt: "2026-04-07T12:00:06.000Z"
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(processed, 2);

  const retryKeys = [...(stores.queue as MemoryStore).values.keys()]
    .filter((key) => key.includes("attempt-2"))
    .sort();
  assert.equal(retryKeys.length, 2);
  assert.match(retryKeys[0] || "", /upload_retry-(natural|travel)-attempt-2/);
  assert.match(retryKeys[1] || "", /upload_retry-(natural|travel)-attempt-2/);
  assert.notEqual(retryKeys[0], retryKeys[1]);
});

test("runWorker exits after the configured max runtime", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.maxRuntimeMs = 10;
  workerConfig.pollIntervalMs = 50;

  const startedAt = Date.now();
  await runWorker(workerConfig, stores, archiveStorage(workerConfig));
  const elapsedMs = Date.now() - startedAt;

  assert.ok(elapsedMs < 80, `expected runWorker to exit before the next poll delay, got ${elapsedMs}ms`);
});

test("stable status key reaches completed after a retry succeeds", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.pythonExecutable = "python3";
  workerConfig.retryBaseDelayMs = 0;
  workerConfig.retryMaxDelayMs = 0;
  workerConfig.pythonScript = await writeTempPythonScript(`
import json
import os
import sys
from pathlib import Path

state_path = Path(os.environ["RIZZUP_RETRY_STATE"])
state = 0
if state_path.exists():
    state = int(state_path.read_text())

request = json.loads(sys.stdin.read())
if state == 0:
    state_path.write_text("1")
    sys.stderr.write("boom")
    sys.exit(1)

assert request["action"] == "preview"
Path(request["outputPath"]).write_bytes(b"preview")
json.dump({
  "preset": request["preset"],
  "previewPath": request["outputPath"],
  "rejectedPreviewPath": None,
  "watermarkText": request["watermarkText"],
  "usedGpu": False,
  "identityGenerationUsed": False,
  "identityGenerationMode": "deterministic-enhancement",
  "identityFallbackReason": None,
  "rotatedToPortrait": False,
  "width": 512,
  "height": 512
}, sys.stdout)
`);

  const retryStateDir = await fs.mkdtemp(path.join(os.tmpdir(), "rizzup-worker-retry-state-"));
  const retryStatePath = path.join(retryStateDir, "state.txt");
  workerConfig.imageArchiveRoot = path.join(retryStateDir, "archive");
  process.env.RIZZUP_RETRY_STATE = retryStatePath;

  await stores.results.setJSON("upload_photo/upload_preview.json", {
    uploadId: "upload_preview",
    imageJobId: "imgjob_preview",
    sourceName: "photo.jpg",
    mimeType: "image/jpeg",
    sizeBytes: 1234,
    createdAt: "2026-04-07T12:00:00.000Z",
    sourcePath: "source\\upload_preview.jpg"
  });
  const rootQueueKey = "generate_preview/upload_preview-natural.json";
  await stores.queue.setJSON(rootQueueKey, {
    type: "generate_preview",
    queuedAt: "2026-04-07T12:00:05.000Z",
    payload: {
      uploadId: "upload_preview",
      preset: "natural",
      requestedAt: "2026-04-07T12:00:05.000Z"
    }
  });

  const firstPass = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(firstPass, 1);

  const secondPass = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(secondPass, 1);

  const status = await stores.status.getWithMetadata<{
    status: string;
    resultKey?: string;
  }>(
    `status/${Buffer.from(rootQueueKey).toString("base64url")}.json`,
    { type: "json" }
  );
  assert.equal(status?.data.status, "completed");
  assert.equal(status?.data.resultKey, "generate_preview/upload_preview-natural.json");

  delete process.env.RIZZUP_RETRY_STATE;
});

test("pollOnce dead-letters face validation analyze failures without fallback", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.pythonScript = await writeTempPythonScript(`
import json
import sys
if json.loads(sys.stdin.read()).get("action") == "analyze":
    json.dump({"code": "FACE_NOT_DETECTED", "message": "No face detected. Please upload a clear photo with one visible face.", "retryable": False}, sys.stderr)
    sys.exit(1)
sys.stderr.write("unexpected")
sys.exit(1)
`);

  await stores.results.setJSON("upload_photo/upload_analyze_face.json", {
    uploadId: "upload_analyze_face",
    imageJobId: "imgjob_analyze_face",
    sourceName: "photo.jpg",
    mimeType: "image/jpeg",
    sizeBytes: 1234,
    createdAt: "2026-04-07T12:00:00.000Z",
    sourcePath: "source\\upload_analyze_face.jpg"
  });
  const queueKey = "analyze_photo_quality/2026-04-07T12-00-04-000Z-upload_analyze_face.json";
  await stores.queue.setJSON(queueKey, {
    type: "analyze_photo_quality",
    queuedAt: "2026-04-07T12:00:04.000Z",
    payload: {
      uploadId: "upload_analyze_face",
      requestedAt: "2026-04-07T12:00:04.000Z"
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(processed, 1);

  const status = await stores.status.getWithMetadata<{
    status: string;
    error?: string;
    errorCode?: string;
  }>(
    `status/${Buffer.from(queueKey).toString("base64url")}.json`,
    { type: "json" }
  );
  assert.equal(status?.data.status, "dead_lettered");
  assert.equal(status?.data.errorCode, "FACE_NOT_DETECTED");
  assert.match(status?.data.error || "", /No face detected/);

  const result = await stores.results.getWithMetadata("analyze_photo_quality/upload_analyze_face.json", {
    type: "json"
  });
  assert.equal(result, null);

  const queueRecord = await stores.queue.getWithMetadata(queueKey, { type: "json" });
  assert.equal(queueRecord, null);
});

test("pollOnce dead-letters analyze jobs clearly when upload validation never produced an upload record", async () => {
  const stores = buildStores();
  const queueKey = "analyze_photo_quality/2026-04-08T14-44-54-854Z-upload_missing.json";
  await stores.queue.setJSON(queueKey, {
    type: "analyze_photo_quality",
    queuedAt: "2026-04-08T14:44:54.854Z",
    payload: {
      uploadId: "upload_missing",
      requestedAt: "2026-04-08T14:44:54.854Z"
    }
  });

  const cfg = config();
  const processed = await pollOnce(cfg, stores, archiveStorage(cfg));
  assert.equal(processed, 1);

  const status = await stores.status.getWithMetadata<{
    status: string;
    error?: string;
    errorCode?: string;
  }>(
    `status/${Buffer.from(queueKey).toString("base64url")}.json`,
    { type: "json" }
  );
  assert.equal(status?.data.status, "dead_lettered");
  assert.equal(status?.data.errorCode, "UPLOAD_NOT_AVAILABLE");
  assert.match(status?.data.error || "", /initial upload validation likely failed/i);
});

test("pollOnce dead-letters preview jobs clearly when upload validation never produced an upload record", async () => {
  const stores = buildStores();
  const queueKey = "generate_preview/2026-04-08T14-45-20-324Z-upload_missing.json";
  await stores.queue.setJSON(queueKey, {
    type: "generate_preview",
    queuedAt: "2026-04-08T14:45:20.324Z",
    payload: {
      uploadId: "upload_missing",
      preset: "natural",
      requestedAt: "2026-04-08T14:45:20.324Z"
    }
  });

  const cfg = config();
  const processed = await pollOnce(cfg, stores, archiveStorage(cfg));
  assert.equal(processed, 1);

  const status = await stores.status.getWithMetadata<{
    status: string;
    error?: string;
    errorCode?: string;
  }>(
    `status/${Buffer.from(queueKey).toString("base64url")}.json`,
    { type: "json" }
  );
  assert.equal(status?.data.status, "dead_lettered");
  assert.equal(status?.data.errorCode, "UPLOAD_NOT_AVAILABLE");
  assert.match(status?.data.error || "", /initial upload validation likely failed/i);
});

test("pollOnce passes deterministic preview settings through the python bridge", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.pythonScript = await writeTempPythonScript(`
import json
import sys

request = json.loads(sys.stdin.read())
assert request["action"] == "preview"
assert request["faceDetection"]["box"]["x"] == 12
assert request["previewMaxSize"] == 512

json.dump({
    "preset": request["preset"],
    "previewPath": request["outputPath"],
    "watermarkText": request["watermarkText"],
    "usedGpu": False,
    "identityGenerationUsed": False,
    "identityGenerationMode": "deterministic-enhancement",
    "identityFallbackReason": None,
    "width": 512,
    "height": 640,
}, sys.stdout)
`);

  await stores.results.setJSON("upload_photo/upload_identity.json", {
    uploadId: "upload_identity",
    imageJobId: "imgjob_identity",
    sourceName: "photo.jpg",
    mimeType: "image/jpeg",
    sizeBytes: 1234,
    createdAt: "2026-04-07T12:00:00.000Z",
    sourcePath: "source\\upload_identity.jpg",
    faceDetection: {
      box: { x: 12, y: 24, w: 240, h: 240 },
      landmarks: {
        leftEye: [60, 80],
        rightEye: [160, 80],
        noseTip: [110, 130],
        mouthCenter: [110, 180]
      },
      debug: {
        rawFaces: [[12, 24, 240, 240]],
        rawEyes: [[20, 30, 40, 20], [120, 30, 40, 20]]
      },
      rotatedToPortrait: false
    }
  });
  await fs.mkdir(path.join(workerConfig.imageArchiveRoot, "2026", "04", "07", "imgjob_identity", "generated", "preview"), {
    recursive: true
  });
  await fs.writeFile(
    path.join(workerConfig.imageArchiveRoot, "2026", "04", "07", "imgjob_identity", "generated", "preview", "natural.png"),
    "placeholder"
  );
  const queueKey = "generate_preview/2026-04-07T12-00-05-000Z-upload_identity.json";
  await stores.queue.setJSON(queueKey, {
    type: "generate_preview",
    queuedAt: "2026-04-07T12:00:05.000Z",
    payload: {
      uploadId: "upload_identity",
      preset: "natural",
      requestedAt: "2026-04-07T12:00:05.000Z"
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(processed, 1);

  const result = await stores.results.getWithMetadata<{
    identityGenerationMode?: string;
    identityFallbackReason?: string | null;
  }>("generate_preview/upload_identity-natural.json", { type: "json" });
  assert.equal(result?.data.identityGenerationMode, "deterministic-enhancement");
  assert.equal(result?.data.identityFallbackReason, null);
});

test("pollOnce accepts noisy preview stdout when the final line is valid JSON", async () => {
  const stores = buildStores();
  const workerConfig = config();
  workerConfig.pythonScript = await writeTempPythonScript(`
import json
import sys

request = json.loads(sys.stdin.read())
assert request["action"] == "preview"
print("Applied providers: CUDAExecutionProvider, CPUExecutionProvider")
json.dump({
    "preset": request["preset"],
    "previewPath": request["outputPath"],
    "watermarkText": request["watermarkText"],
    "usedGpu": True,
    "identityGenerationUsed": False,
      "identityGenerationMode": "deterministic-enhancement",
    "identityFallbackReason": None,
    "width": 512,
    "height": 640,
}, sys.stdout)
`);

  await stores.results.setJSON("upload_photo/upload_noise.json", {
    uploadId: "upload_noise",
    imageJobId: "imgjob_noise",
    sourceName: "photo.jpg",
    mimeType: "image/jpeg",
    sizeBytes: 1234,
    createdAt: "2026-04-07T12:00:00.000Z",
    sourcePath: "source\\upload_noise.jpg",
    faceDetection: {
      box: { x: 12, y: 24, w: 240, h: 240 },
      landmarks: {
        leftEye: [60, 80],
        rightEye: [160, 80],
        noseTip: [110, 130],
        mouthCenter: [110, 180]
      },
      debug: {
        rawFaces: [[12, 24, 240, 240]],
        rawEyes: []
      },
      rotatedToPortrait: false,
      rotationDegrees: 0
    }
  });
  await fs.mkdir(path.join(workerConfig.imageArchiveRoot, "2026", "04", "07", "imgjob_noise", "generated", "preview"), {
    recursive: true
  });
  await fs.writeFile(
    path.join(workerConfig.imageArchiveRoot, "2026", "04", "07", "imgjob_noise", "generated", "preview", "natural.png"),
    "placeholder"
  );
  const queueKey = "generate_preview/2026-04-07T12-00-05-000Z-upload_noise.json";
  await stores.queue.setJSON(queueKey, {
    type: "generate_preview",
    queuedAt: "2026-04-07T12:00:05.000Z",
    payload: {
      uploadId: "upload_noise",
      preset: "natural",
      requestedAt: "2026-04-07T12:00:05.000Z"
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage(workerConfig));
  assert.equal(processed, 1);

  const result = await stores.results.getWithMetadata<{
    identityGenerationMode?: string;
    preset?: string;
  }>("generate_preview/upload_noise-natural.json", { type: "json" });
  assert.equal(result?.data.preset, "natural");
  assert.equal(result?.data.identityGenerationMode, "deterministic-enhancement");
});
