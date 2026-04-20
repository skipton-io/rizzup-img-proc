import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pollOnce } from "../src/worker";
import { ArchiveStorage, BlobStoreLike, WorkerConfig, WorkerStores } from "../src/types";

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
    if (options?.onlyIfNew && existing) return { modified: false };
    if (options?.onlyIfMatch && existing?.etag !== options.onlyIfMatch) return { modified: false };
    const etag = `${Date.now()}-${Math.random()}`;
    this.values.set(key, { data: value, etag, metadata: options?.metadata });
    return { modified: true, etag };
  }

  async setJSON<T>(
    key: string,
    value: T,
    options?: { metadata?: Record<string, unknown>; onlyIfMatch?: string; onlyIfNew?: boolean }
  ): Promise<{ modified: boolean; etag?: string }> {
    return await this.set(key, value as unknown as string, options);
  }

  async delete(key: string): Promise<void> {
    this.values.delete(key);
  }
}

class RecordingArchiveStorage implements ArchiveStorage {
  public readonly backend = "sftp" as const;
  public readonly root = "/volume1/web/rizzup.co.uk/image-jobs";
  public readonly writeCalls: Array<{ relativePath: string; size: number }> = [];
  public readonly uploadCalls: Array<{ localPath: string; relativePath: string }> = [];

  resolveArchivePath(relativePath: string): string {
    return `${this.root}/${relativePath.replace(/\\/g, "/").replace(/^\/+/, "")}`;
  }

  async writeBuffer(relativePath: string, data: Buffer): Promise<string> {
    this.writeCalls.push({ relativePath, size: data.byteLength });
    return this.resolveArchivePath(relativePath);
  }

  async uploadFile(localPath: string, relativePath: string): Promise<string> {
    this.uploadCalls.push({ localPath, relativePath });
    return this.resolveArchivePath(relativePath);
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

async function writeTempPythonScript(contents: string): Promise<string> {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "rizzup-sftp-test-"));
  const scriptPath = path.join(directory, "script.py");
  await fs.writeFile(scriptPath, contents, "utf8");
  return scriptPath;
}

async function buildSftpConfig(): Promise<WorkerConfig> {
  const resultsDir = await fs.mkdtemp(path.join(os.tmpdir(), "rizzup-sftp-worker-"));
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
    workerId: "worker_sftp_test",
    previewWatermarkText: "RizzUp Preview",
    previewWatermarkLogoPath: path.resolve(process.cwd(), "..", "rizzup.co.uk", "public", "brand", "rizzup-logo.png"),
    resultsDir,
    archiveBackend: "sftp",
    imageArchiveRoot: "/volume1/web/rizzup.co.uk/image-jobs",
    sourceImageRoot: path.join(resultsDir, "source-images"),
    localRenderRoot: path.join(resultsDir, "renders"),
    sftpHost: "100.64.0.10",
    sftpPort: 22,
    sftpUsername: "rizzup-archive",
    sftpPassword: "secret",
    sftpStrictHostKey: false,
    pythonExecutable: "python3",
    pythonScript: "",
    fireRedEnabled: true,
    fireRedModelId: "FireRedTeam/FireRed-Image-Edit-1.1",
    fireRedLoraRepo: "FireRedTeam/FireRed-Image-Edit-LoRA-Zoo",
    fireRedLoraWeight: "FireRed-Image-Edit-Makeup.safetensors",
    fireRedLoraAdapterName: "makeup",
    fireRedPrompt: "Western makeup",
    fireRedInferenceSteps: 30,
    fireRedTrueCfgScale: 4,
    analysisMaxSize: 100,
    previewMaxSize: 512,
    finalDecisionMaxSize: 512,
    finalMinWidth: 1024,
    finalMinHeight: 1280
  };
}

test("SFTP upload_photo stores a logical archive path and writes the dated source path", async () => {
  const stores = buildStores();
  const archiveStorage = new RecordingArchiveStorage();
  const workerConfig = await buildSftpConfig();
  workerConfig.pythonScript = await writeTempPythonScript(`
import json
import sys
request = json.loads(sys.stdin.read())
assert request["action"] == "validate_upload"
json.dump({
  "faceDetection": {
    "box": {"x": 1, "y": 2, "w": 3, "h": 4},
    "landmarks": {
      "leftEye": [10, 10],
      "rightEye": [20, 10],
      "noseTip": [15, 20],
      "mouthCenter": [15, 30]
    },
    "debug": {"rawFaces": [], "rawEyes": []},
    "rotatedToPortrait": False
  }
}, sys.stdout)
`);

  await stores.queue.setJSON("upload_photo/upload_sftp.json", {
    type: "upload_photo",
    queuedAt: "2026-04-07T12:00:00.000Z",
    payload: {
      uploadId: "upload_sftp",
      imageJobId: "imgjob_sftp",
      sourceName: "photo.jpg",
      mimeType: "image/jpeg",
      sizeBytes: 1234,
      createdAt: "2026-04-07T12:00:00.000Z",
      sourceDataUrl: "data:image/jpeg;base64,AA=="
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage);
  assert.equal(processed, 1);
  assert.deepEqual(archiveStorage.writeCalls, [
    {
      relativePath: "2026/04/07/imgjob_sftp/source/upload_sftp-photo.jpg",
      size: 1
    }
  ]);

  const result = await stores.results.getWithMetadata<{ sourcePath?: string; sourceRelativePath?: string }>(
    "upload_photo/upload_sftp.json",
    { type: "json" }
  );
  assert.equal(result?.data.sourcePath, "2026/04/07/imgjob_sftp/source/upload_sftp-photo.jpg");
  assert.equal(result?.data.sourceRelativePath, "source/upload_sftp-photo.jpg");
});

test("SFTP preview and final generation upload dated archive paths while rendering locally", async () => {
  const stores = buildStores();
  const archiveStorage = new RecordingArchiveStorage();
  const workerConfig = await buildSftpConfig();
  workerConfig.pythonScript = await writeTempPythonScript(`
import json
import sys
from pathlib import Path

request = json.loads(sys.stdin.read())
Path(request["outputPath"]).parent.mkdir(parents=True, exist_ok=True)
Path(request["outputPath"]).write_bytes(b"image")

if request["action"] == "preview":
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
      "height": 640
    }, sys.stdout)
elif request["action"] == "final":
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
else:
    raise AssertionError(f"unexpected action {request['action']}")
`);

  await stores.results.setJSON("upload_photo/upload_sftp.json", {
    uploadId: "upload_sftp",
    imageJobId: "imgjob_sftp",
    sourceName: "photo.jpg",
    mimeType: "image/jpeg",
    sizeBytes: 1234,
    createdAt: "2026-04-07T12:00:00.000Z",
    sourcePath: "2026/04/07/imgjob_sftp/source/upload_sftp-photo.jpg",
    sourceRelativePath: "source/upload_sftp-photo.jpg"
  });

  await stores.queue.setJSON("generate_preview/upload_sftp-natural.json", {
    type: "generate_preview",
    queuedAt: "2026-04-07T12:00:05.000Z",
    payload: {
      uploadId: "upload_sftp",
      preset: "natural",
      requestedAt: "2026-04-07T12:00:05.000Z"
    }
  });
  await stores.queue.setJSON("generate_final_image/unlock_sftp.json", {
    type: "generate_final_image",
    queuedAt: "2026-04-07T12:00:10.000Z",
    payload: {
      unlockId: "unlock_sftp",
      checkoutSessionId: "checkout_sftp",
      uploadId: "upload_sftp",
      preset: "travel",
      plan: "5_photos",
      requestedAt: "2026-04-07T12:00:10.000Z"
    }
  });

  const processed = await pollOnce(workerConfig, stores, archiveStorage);
  assert.equal(processed, 2);
  assert.deepEqual(
    archiveStorage.uploadCalls.map((call) => call.relativePath),
    [
      "2026/04/07/imgjob_sftp/generated/preview/natural.png",
      "2026/04/07/imgjob_sftp/generated/final/unlock_sftp-travel.png"
    ]
  );
  assert.ok(
    archiveStorage.uploadCalls.every((call) => call.localPath.startsWith(workerConfig.localRenderRoot))
  );

  const preview = await stores.results.getWithMetadata<{ previewPath?: string }>(
    "generate_preview/upload_sftp-natural.json",
    { type: "json" }
  );
  const final = await stores.results.getWithMetadata<{ finalImagePath?: string }>(
    "generate_final_image/unlock_sftp.json",
    { type: "json" }
  );
  assert.equal(preview?.data.previewPath, "2026/04/07/imgjob_sftp/generated/preview/natural.png");
  assert.equal(final?.data.finalImagePath, "2026/04/07/imgjob_sftp/generated/final/unlock_sftp-travel.png");
});
