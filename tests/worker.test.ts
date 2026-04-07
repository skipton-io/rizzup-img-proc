import test from "node:test";
import assert from "node:assert/strict";
import { pollOnce } from "../src/worker";
import { BlobStoreLike, WorkerConfig, WorkerStores } from "../src/types";

class MemoryStore implements BlobStoreLike {
  public readonly values = new Map<string, { data: unknown; etag: string }>();

  async *list(options: { prefix: string; paginate: true }) {
    const blobs = [...this.values.keys()]
      .filter((key) => key.startsWith(options.prefix))
      .map((key) => ({ key, etag: this.values.get(key)?.etag }));

    yield { blobs };
  }

  async getWithMetadata<T>(key: string): Promise<{ data: T; etag?: string } | null> {
    const value = this.values.get(key);
    if (!value) return null;
    return { data: value.data as T, etag: value.etag };
  }

  async setJSON<T>(
    key: string,
    value: T,
    options?: { onlyIfMatch?: string; onlyIfNew?: boolean }
  ): Promise<{ modified: boolean; etag?: string }> {
    const existing = this.values.get(key);
    if (options?.onlyIfNew && existing) {
      return { modified: false };
    }

    if (options?.onlyIfMatch && existing?.etag !== options.onlyIfMatch) {
      return { modified: false };
    }

    const etag = `${Date.now()}-${Math.random()}`;
    this.values.set(key, { data: value, etag });
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
    locksStore: "locks",
    deadLetterStore: "dead",
    pollIntervalMs: 1,
    maxJobsPerPoll: 4,
    lockTtlMs: 60_000,
    maxAttempts: 3,
    retryBaseDelayMs: 1_000,
    retryMaxDelayMs: 5_000,
    workerId: "worker_test",
    previewWatermarkText: "RizzUp Preview",
    resultsDir: "artifacts",
    pythonExecutable: "python",
    pythonScript: "scripts/gpu_pipeline.py"
  };
}

test("pollOnce processes upload_photo records into the results store", async () => {
  const stores = buildStores();
  await stores.queue.setJSON("upload_photo/2026-04-07-upload_123.json", {
    type: "upload_photo",
    queuedAt: "2026-04-07T12:00:00.000Z",
    payload: {
      uploadId: "upload_123",
      sourceName: "photo.jpg",
      mimeType: "image/jpeg",
      sizeBytes: 1234,
      createdAt: "2026-04-07T12:00:00.000Z"
    }
  });

  const processed = await pollOnce(config(), stores);
  assert.equal(processed, 1);

  const upload = await stores.results.getWithMetadata<{
    uploadId: string;
    sourceName: string;
  }>("upload_photo/upload_123.json");

  assert.equal(upload?.data.uploadId, "upload_123");
  assert.equal(upload?.data.sourceName, "photo.jpg");
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

  const processed = await pollOnce(config(), stores);
  assert.equal(processed, 0);
});
