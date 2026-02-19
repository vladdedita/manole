import { join } from "path";
import { homedir, platform } from "os";
import type { BrowserWindow } from "electron";
import type { PythonBridge } from "./python";

type SetupState = "checking" | "needed" | "complete" | "skipped";

/**
 * Resolves the models directory for the current platform.
 * Accepts isPackaged and appDataPath as parameters to enable testing
 * without Electron app dependency.
 */
export function resolveModelsDir(
  isPackaged: boolean,
  appDataPath: string,
  projectRoot: string,
): string {
  if (isPackaged) {
    if (platform() === "darwin") {
      return join(appDataPath, "Manole", "models");
    }
    // Linux
    return join(homedir(), ".local", "share", "manole", "models");
  }
  // Dev mode
  return join(projectRoot, "models");
}

export class ModelSetupManager {
  constructor(
    private python: PythonBridge,
    private window: BrowserWindow,
  ) {}

  getModelsDir(): string {
    // Lazy import to avoid Electron dependency in tests
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { app } = require("electron");
      const projectRoot = this.getProjectRoot(app);
      return resolveModelsDir(app.isPackaged, app.getPath("appData"), projectRoot);
    } catch {
      // Fallback for test environment: use cwd-based dev path
      return join(process.cwd(), "models");
    }
  }

  private getProjectRoot(app: { isPackaged: boolean }): string {
    if (app.isPackaged) {
      return process.resourcesPath;
    }
    return join(__dirname, "..", "..", "..");
  }

  private sendState(state: SetupState): void {
    this.window.webContents.send("setup:state", state);
  }

  async checkAndDownload(): Promise<void> {
    this.sendState("checking");

    const checkResult = await this.python.send("check_models", {
      models_dir: this.getModelsDir(),
    });

    const ready = checkResult.data?.ready === true;

    if (ready) {
      this.sendState("skipped");
      return;
    }

    // Models are missing — trigger download
    this.sendState("needed");

    await this.python.send("download_models", {
      models_dir: this.getModelsDir(),
    });

    this.sendState("complete");
  }
}
