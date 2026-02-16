import { ChildProcess, spawn } from "child_process";
import { app } from "electron";
import { join } from "path";
import { createInterface } from "readline";
import type { Request, Response } from "../src/lib/protocol";

export type ResponseHandler = (response: Response) => void;

export class PythonBridge {
  private process: ChildProcess | null = null;
  private nextId = 1;
  private handlers: Map<number, ResponseHandler> = new Map();
  private globalHandler: ResponseHandler | null = null;

  private getPythonCommand(): { command: string; args: string[] } {
    if (app.isPackaged) {
      const binary = join(process.resourcesPath, "manole-server");
      return { command: binary, args: [] };
    }
    const projectRoot = join(__dirname, "..", "..");
    const python = join(projectRoot, ".venv", "bin", "python");
    const serverPy = join(projectRoot, "server.py");
    return { command: python, args: [serverPy] };
  }

  spawn(onMessage: ResponseHandler): void {
    const { command, args } = this.getPythonCommand();
    this.globalHandler = onMessage;

    this.process = spawn(command, args, {
      stdio: ["pipe", "pipe", "pipe"],
    });

    const rl = createInterface({ input: this.process.stdout! });
    rl.on("line", (line: string) => {
      try {
        const response: Response = JSON.parse(line);
        if (response.id !== null && this.handlers.has(response.id)) {
          if (response.type === "result" || response.type === "error") {
            const handler = this.handlers.get(response.id)!;
            handler(response);
            this.handlers.delete(response.id);
          } else {
            onMessage(response);
          }
        } else {
          onMessage(response);
        }
      } catch {
        // Ignore non-JSON lines
      }
    });

    this.process.stderr?.on("data", (data: Buffer) => {
      console.error("[python]", data.toString());
    });

    this.process.on("exit", (code) => {
      console.error(`[python] exited with code ${code}`);
      onMessage({
        id: null,
        type: "error",
        data: { message: `Python process exited (code ${code})` },
      });
    });
  }

  send(method: string, params: Record<string, unknown> = {}): Promise<Response> {
    return new Promise((resolve, reject) => {
      if (!this.process?.stdin?.writable) {
        reject(new Error("Python process not running"));
        return;
      }
      const id = this.nextId++;
      const request: Request = { id, method, params };
      this.handlers.set(id, resolve);
      const line = JSON.stringify(request) + "\n";
      this.process.stdin.write(line);
    });
  }

  kill(): void {
    if (this.process) {
      this.send("shutdown").catch(() => {});
      setTimeout(() => this.process?.kill(), 2000);
    }
  }
}
