import { mkdirSync } from "node:fs";
import path from "node:path";
import { createStores } from "./blobStores";
import { loadConfig } from "./config";
import { runWorker } from "./worker";

async function main(): Promise<void> {
  const config = loadConfig();
  mkdirSync(config.resultsDir, { recursive: true });
  mkdirSync(path.dirname(config.pythonScript), { recursive: true });
  await runWorker(config, createStores(config));
}

void main();
