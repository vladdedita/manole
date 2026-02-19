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

  private getProjectRoot(): string {
    if (app.isPackaged) {
      return process.resourcesPath;
    }
    // __dirname is ui/out/main in dev, so 3 levels up to project root
    return join(__dirname, "..", "..", "..");
  }

  private getPythonCommand(): { command: string; args: string[] } {
    if (app.isPackaged) {
      const binary = join(process.resourcesPath, "manole-server");
      return { command: binary, args: [] };
    }
    const projectRoot = this.getProjectRoot();
    const python = join(projectRoot, ".venv", "bin", "python");
    const serverPy = join(projectRoot, "server.py");
    return { command: python, args: [serverPy] };
  }

  spawn(onMessage: ResponseHandler, env?: Record<string, string>): void {
    const { command, args } = this.getPythonCommand();
    this.globalHandler = onMessage;

    const cwd = this.getProjectRoot();
    console.error(`[python] spawning: ${command} ${args.join(" ")} (cwd: ${cwd})`);
    this.process = spawn(command, args, {
      stdio: ["pipe", "pipe", "pipe"],
      cwd,
      env: { ...process.env, ...env },
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
        // Non-JSON lines from C libraries (llama.cpp) — ignore
      }
    });

    this.process.stderr?.on("data", (data: Buffer) => {
      const text = data.toString();
      console.error("[python]", text);
      if (this.globalHandler) {
        this.globalHandler({ id: null, type: "log" as any, data: { text } });
      }
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

  get pid(): number | undefined {
    return this.process?.pid
  }

  kill(): void {
    if (this.process) {
      this.send("shutdown").catch(() => {});
      setTimeout(() => this.process?.kill(), 2000);
    }
  }
}
