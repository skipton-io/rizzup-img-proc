declare module "ssh2-sftp-client" {
  type ConnectOptions = {
    host: string;
    port?: number;
    username: string;
    password?: string;
    hostHash?: string;
    hostVerifier?: ((hashedKey: string) => boolean) | undefined;
  };

  export default class SftpClient {
    connect(options: ConnectOptions): Promise<void>;
    mkdir(remotePath: string, recursive?: boolean): Promise<void>;
    put(input: Buffer | string, remotePath: string): Promise<void>;
    end(): Promise<void>;
  }
}
