import { mkdirSync } from "node:fs";
import path from "node:path";
import { createArchiveStorage } from "./archiveStorage";
import { createStores } from "./blobStores";
import { loadConfig } from "./config";
import { loadDotEnv } from "./env";
import { runWorker } from "./worker";

async function main(): Promise<void> {
  loadDotEnv();
  const config = loadConfig();
  const archiveStorage = createArchiveStorage(config);
  mkdirSync(config.resultsDir, { recursive: true });
  mkdirSync(config.localRenderRoot, { recursive: true });
  if (config.sourceImageRoot) {
    mkdirSync(config.sourceImageRoot, { recursive: true });
  }
  mkdirSync(path.dirname(config.pythonScript), { recursive: true });
  await runWorker(config, createStores(config), archiveStorage);
}

void main();
