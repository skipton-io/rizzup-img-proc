import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  PipelineJobError,
  analyzeWithPython,
  validateUploadWithPython
} from "../src/pythonBridge";
import { ArchiveStorage, BlobStoreLike, HandlerContext, WorkerConfig, WorkerStores } from "../src/types";

class MemoryStore implements BlobStoreLike {
  async *list(_options: { prefix: string; paginate: true }) {
    yield { blobs: [] };
  }

  async getWithMetadata<T>(
    _key: string,
    _options: { type: "json" } | { type: "arrayBuffer" }
  ): Promise<{ data: T; etag?: string } | null> {
    return null;
  }

  async set(
    _key: string,
    _value: string | ArrayBuffer | Blob,
    _options?: { metadata?: Record<string, unknown>; onlyIfMatch?: string; onlyIfNew?: boolean }
  ): Promise<{ modified: boolean; etag?: string }> {
    return { modified: true, etag: "etag" };
  }

  async setJSON<T>(
    _key: string,
    _value: T,
    _options?: { metadata?: Record<string, unknown>; onlyIfMatch?: string; onlyIfNew?: boolean }
  ): Promise<{ modified: boolean; etag?: string }> {
    return { modified: true, etag: "etag" };
  }

  async delete(_key: string): Promise<void> {}
}

class LocalArchiveStorage implements ArchiveStorage {
  public readonly backend = "local" as const;

  constructor(public readonly root: string) {}

  resolveArchivePath(relativePath: string): string {
    return path.join(this.root, relativePath);
  }

  async writeBuffer(relativePath: string, _data: Buffer): Promise<string> {
    return this.resolveArchivePath(relativePath);
  }

  async uploadFile(localPath: string, relativePath: string): Promise<string> {
    return path.join(path.dirname(localPath), relativePath);
  }
}

async function writeTempPythonScript(contents: string): Promise<string> {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "rizzup-python-bridge-test-"));
  const scriptPath = path.join(directory, "script.py");
  await fs.writeFile(scriptPath, contents, "utf8");
  return scriptPath;
}

function buildContext(scriptPath: string, sourceImageRoot?: string): HandlerContext {
  const stores: WorkerStores = {
    queue: new MemoryStore(),
    status: new MemoryStore(),
    results: new MemoryStore(),
    assets: new MemoryStore(),
    locks: new MemoryStore(),
    deadLetter: new MemoryStore()
  };

  const config: WorkerConfig = {
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
    workerId: "worker_python_bridge_test",
    previewWatermarkText: "RizzUp Preview",
    previewWatermarkLogoPath: path.resolve(process.cwd(), "logo.png"),
    resultsDir: "artifacts",
    archiveBackend: "local",
    imageArchiveRoot: path.resolve(process.cwd(), "artifacts", "archive"),
    sourceImageRoot,
    localRenderRoot: path.resolve(process.cwd(), "artifacts", "renders"),
    sftpHost: undefined,
    sftpPort: 22,
    sftpUsername: undefined,
    sftpPassword: undefined,
    sftpStrictHostKey: false,
    sftpHostKey: undefined,
    pythonExecutable: "python3",
    pythonScript: scriptPath,
    faceCascadePath: undefined,
    eyeCascadePath: undefined,
    fireRedEnabled: false,
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

  return {
    config,
    stores,
    archiveStorage: new LocalArchiveStorage(config.imageArchiveRoot)
  };
}

test("validateUploadWithPython rejects empty stdout", async () => {
  const scriptPath = await writeTempPythonScript("");
  const context = buildContext(scriptPath);

  await assert.rejects(
    () =>
      validateUploadWithPython(
        "upload_empty_stdout",
        {
          uploadId: "upload_empty_stdout",
          imageJobId: "imgjob",
          sourceName: "photo.jpg",
          mimeType: "image/jpeg",
          sizeBytes: 123,
          createdAt: "2026-04-07T12:00:00.000Z"
        },
        context
      ),
    /Python pipeline returned no stdout/
  );
});

test("analyzeWithPython rejects stdout when no valid JSON line is present", async () => {
  const scriptPath = await writeTempPythonScript(`
import sys
print("hello")
print("still not json")
`);
  const context = buildContext(scriptPath);

  await assert.rejects(
    () => analyzeWithPython("upload_bad_stdout", null, context),
    /Could not parse python output/
  );
});

test("validateUploadWithPython falls back to a generic error when structured stderr has no message", async () => {
  const scriptPath = await writeTempPythonScript(`
import json
import sys
json.dump({"code": "FACE_NOT_DETECTED"}, sys.stderr)
sys.exit(1)
`);
  const context = buildContext(scriptPath);

  await assert.rejects(
    () =>
      validateUploadWithPython(
        "upload_no_message",
        {
          uploadId: "upload_no_message",
          imageJobId: "imgjob",
          sourceName: "photo.jpg",
          mimeType: "image/jpeg",
          sizeBytes: 123,
          createdAt: "2026-04-07T12:00:00.000Z"
        },
        context
      ),
    (error: unknown) =>
      error instanceof Error &&
      !(error instanceof PipelineJobError) &&
      /Python pipeline exited with code 1/.test(error.message)
  );
});

test("validateUploadWithPython resolves relative source paths from sourceImageRoot when configured", async () => {
  const sourceImageRoot = await fs.mkdtemp(path.join(os.tmpdir(), "rizzup-python-source-root-"));
  const expectedPath = path.join(sourceImageRoot, "nested", "photo.jpg");
  const scriptPath = await writeTempPythonScript(`
import json
import os
import sys
request = json.loads(sys.stdin.read())
assert request["sourcePath"] == os.environ["EXPECTED_SOURCE_PATH"], request["sourcePath"]
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
  process.env.EXPECTED_SOURCE_PATH = expectedPath;
  const context = buildContext(scriptPath, sourceImageRoot);

  const result = await validateUploadWithPython(
    "upload_relative_path",
    {
      uploadId: "upload_relative_path",
      imageJobId: "imgjob",
      sourceName: "photo.jpg",
      mimeType: "image/jpeg",
      sizeBytes: 123,
      createdAt: "2026-04-07T12:00:00.000Z",
      sourcePath: path.join("nested", "photo.jpg")
    },
    context
  );

  assert.equal(result.box.x, 1);
  delete process.env.EXPECTED_SOURCE_PATH;
});
