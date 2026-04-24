import fs from "node:fs/promises";
import path from "node:path";
import SftpClient from "ssh2-sftp-client";
import { ArchiveStorage, WorkerConfig } from "./types";

function normalizeRelativePath(relativePath: string): string {
  const normalized = relativePath.replace(/\\/g, "/").replace(/^\/+/, "");
  if (!normalized || normalized === "." || normalized.startsWith("../") || normalized.includes("/../")) {
    throw new Error(`Invalid archive relative path: ${relativePath}`);
  }

  return normalized;
}

function stripHostKeyPrefix(hostKey: string): string {
  return hostKey.replace(/^sha256:/i, "").trim();
}

export class LocalArchiveStorage implements ArchiveStorage {
  public readonly backend = "local" as const;

  constructor(public readonly root: string) {}

  resolveArchivePath(relativePath: string): string {
    const normalized = normalizeRelativePath(relativePath);
    return path.join(this.root, ...normalized.split("/"));
  }

  async writeBuffer(relativePath: string, data: Buffer): Promise<string> {
    const archivePath = this.resolveArchivePath(relativePath);
    await fs.mkdir(path.dirname(archivePath), { recursive: true });
    await fs.writeFile(archivePath, data);
    return archivePath;
  }

  async writeTextFile(relativePath: string, data: string): Promise<string> {
    return await this.writeBuffer(relativePath, Buffer.from(data, "utf8"));
  }

  async uploadFile(localPath: string, relativePath: string): Promise<string> {
    const archivePath = this.resolveArchivePath(relativePath);
    if (path.resolve(localPath) !== path.resolve(archivePath)) {
      await fs.mkdir(path.dirname(archivePath), { recursive: true });
      await fs.copyFile(localPath, archivePath);
    }

    return archivePath;
  }
}

export class SftpArchiveStorage implements ArchiveStorage {
  public readonly backend = "sftp" as const;

  constructor(
    public readonly root: string,
    private readonly config: {
      host: string;
      port: number;
      username: string;
      password: string;
      strictHostKey: boolean;
      hostKey?: string;
    }
  ) {}

  resolveArchivePath(relativePath: string): string {
    const normalized = normalizeRelativePath(relativePath);
    return path.posix.join(this.root, normalized);
  }

  async writeBuffer(relativePath: string, data: Buffer): Promise<string> {
    const archivePath = this.resolveArchivePath(relativePath);
    const client = await this.connect();

    try {
      await client.mkdir(path.posix.dirname(archivePath), true);
      await client.put(data, archivePath);
    } finally {
      await client.end();
    }

    return archivePath;
  }

  async writeTextFile(relativePath: string, data: string): Promise<string> {
    return await this.writeBuffer(relativePath, Buffer.from(data, "utf8"));
  }

  async uploadFile(localPath: string, relativePath: string): Promise<string> {
    const archivePath = this.resolveArchivePath(relativePath);
    const client = await this.connect();

    try {
      await client.mkdir(path.posix.dirname(archivePath), true);
      await client.put(localPath, archivePath);
    } finally {
      await client.end();
    }

    return archivePath;
  }

  private async connect(): Promise<SftpClient> {
    const client = new SftpClient();
    const expectedHostKey = this.config.hostKey ? stripHostKeyPrefix(this.config.hostKey) : undefined;

    await client.connect({
      host: this.config.host,
      port: this.config.port,
      username: this.config.username,
      password: this.config.password,
      hostHash: expectedHostKey ? "sha256" : undefined,
      hostVerifier: expectedHostKey
        ? (hashedKey: string) => stripHostKeyPrefix(hashedKey) === expectedHostKey
        : this.config.strictHostKey
          ? () => false
          : undefined
    });

    return client;
  }
}

export function createArchiveStorage(config: WorkerConfig): ArchiveStorage {
  if (config.archiveBackend === "sftp") {
    if (!config.sftpHost || !config.sftpUsername || !config.sftpPassword) {
      throw new Error(
        "SFTP archive backend requires RIZZUP_SFTP_HOST, RIZZUP_SFTP_USERNAME, and RIZZUP_SFTP_PASSWORD"
      );
    }
    if (config.sftpStrictHostKey && !config.sftpHostKey) {
      throw new Error("RIZZUP_SFTP_STRICT_HOST_KEY=true requires RIZZUP_SFTP_HOST_KEY");
    }

    return new SftpArchiveStorage(config.imageArchiveRoot, {
      host: config.sftpHost,
      port: config.sftpPort,
      username: config.sftpUsername,
      password: config.sftpPassword,
      strictHostKey: config.sftpStrictHostKey,
      hostKey: config.sftpHostKey
    });
  }

  return new LocalArchiveStorage(config.imageArchiveRoot);
}

export function archiveRelativePathForJob(
  imageJobId: string,
  createdAt?: string | null
): string {
  const date = createdAt ? new Date(createdAt) : new Date();
  const safeDate = Number.isNaN(date.getTime()) ? new Date() : date;
  return path.posix.join(
    String(safeDate.getUTCFullYear()),
    String(safeDate.getUTCMonth() + 1).padStart(2, "0"),
    String(safeDate.getUTCDate()).padStart(2, "0"),
    imageJobId
  );
}

export function localPathFromRelative(root: string, relativePath: string): string {
  const normalized = normalizeRelativePath(relativePath);
  return path.join(root, ...normalized.split("/"));
}
