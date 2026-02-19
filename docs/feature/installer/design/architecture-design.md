# Manole Installer — Architecture Design

## 1. System Context

```
┌─────────────────────────────────────────────────────────┐
│                    User's Machine                        │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Manole (Electron App)                │   │
│  │                                                  │   │
│  │  ┌─────────────┐        ┌──────────────────┐     │   │
│  │  │  Renderer    │  IPC   │  Main Process    │     │   │
│  │  │  (React UI)  │◄──────►│  (Electron)      │     │   │
│  │  │             │        │                  │     │   │
│  │  │ SetupScreen │        │  PythonBridge    │     │   │
│  │  │ LoadingScr. │        │       │          │     │   │
│  │  │ ChatPanel   │        │       ▼          │     │   │
│  │  │ MapPanel    │        │  manole-server   │     │   │
│  │  └─────────────┘        │  (Python binary) │     │   │
│  │                         └──────────────────┘     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────┐                                   │
│  │ models/           │  (downloaded on first launch)    │
│  │  *.gguf files     │                                  │
│  └──────────────────┘                                   │
└────────────────────────────────────────────────────┬────┘
                                                     │
                                          first launch only
                                                     │
                                                     ▼
                                          ┌──────────────────┐
                                          │  HuggingFace Hub  │
                                          │  (model CDN)      │
                                          └──────────────────┘
```

## 2. Key Architectural Decision: Who Downloads?

### Option A: Python backend downloads models (via `huggingface_hub`)
### Option B: Electron main process downloads models (via Node.js `https`)
### Option C: Python backend downloads, Electron orchestrates

**Decision: Option C — Python backend downloads, Electron orchestrates.**

**Rationale**:
- `huggingface_hub` already has resume support, SHA256 verification, and handles HF API quirks
- The Python backend already imports `huggingface_hub` (see `models.py:_ensure_model`)
- Electron main process handles the lifecycle: detects missing models, spawns Python for download, shows UI progress
- This reuses existing infrastructure while keeping the UI responsive

## 3. Component Architecture

### 3.1 New Components

```
┌─────────────── Electron Main Process ──────────────────┐
│                                                         │
│  ModelSetupManager (NEW)                                │
│  ├── Reads models-manifest.json from app resources      │
│  ├── Checks if all model files exist at platform path   │
│  ├── If missing: tells renderer to show SetupScreen     │
│  ├── Sends "download_models" to Python via PythonBridge │
│  └── Relays progress events to renderer                 │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────── Electron Renderer ──────────────────────┐
│                                                         │
│  SetupScreen (NEW)                                      │
│  ├── Shows welcome message                              │
│  ├── Lists models from manifest with sizes              │
│  ├── Per-model progress bar + checkmarks                │
│  ├── Error messages + retry button                      │
│  └── "Get Started" button on completion                 │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────── Python Backend ─────────────────────────┐
│                                                         │
│  handle_download_models (NEW method on Server)          │
│  ├── Reads manifest from stdin params or embedded file  │
│  ├── For each model:                                    │
│  │   ├── Check if already exists + SHA256 verify        │
│  │   ├── Download via huggingface_hub.hf_hub_download   │
│  │   ├── Send progress events via NDJSON                │
│  │   └── Report completion/error per file               │
│  └── Send final "all_models_ready" event                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Modified Components

| Component | Change |
|-----------|--------|
| `electron-builder.yml` | Remove `models/` from `extraResources` |
| `ui/electron/python.ts` | `PythonBridge.getProjectRoot()` → add platform model path resolution |
| `ui/electron/main.ts` | Add `ModelSetupManager` integration before creating main window |
| `ui/src/App.tsx` | Add `setup_needed` state before `not_initialized` |
| `ui/src/lib/protocol.ts` | Add `setup_progress` and `setup_complete` response types |
| `models.py` | `ModelManager` reads model paths from platform-specific directory |
| `server.py` | Add `handle_download_models` and `handle_check_models` methods |

### 3.3 Unchanged Components

Everything else: `chat.py`, `agent.py`, `searcher.py`, `tools.py`, `toolbox.py`, `router.py`, `rewriter.py`, `parser.py`, `file_reader.py`, `graph.py`, `image_captioner.py`, `caption_cache.py`.

## 4. Model Storage — Platform Paths

macOS `.app` bundles and Linux AppImages are both **read-only**. Models must go to a writable user directory.

| Platform | Model Directory |
|----------|----------------|
| macOS | `~/Library/Application Support/Manole/models/` |
| Linux | `~/.local/share/manole/models/` |
| Dev mode | `./models/` (project root, current behavior) |

**Resolution logic** (in both Electron and Python):

```
if running packaged:
    if macOS:  ~/Library/Application Support/Manole/models/
    if Linux:  ~/.local/share/manole/models/
else:
    ./models/   (dev mode, unchanged)
```

**Python side**: `ModelManager.__init__` detects whether it's running from a PyInstaller bundle (`getattr(sys, 'frozen', False)`) and adjusts base path.

**Electron side**: `PythonBridge` passes the resolved model directory to the Python process via an environment variable `MANOLE_MODELS_DIR`.

## 5. NDJSON Protocol Extensions

### 5.1 New Request: `check_models`

```json
{"id": 1, "method": "check_models", "params": {"modelsDir": "/path/to/models"}}
```

Response:
```json
{"id": 1, "type": "result", "data": {
  "ready": false,
  "models": [
    {"id": "text-model", "filename": "LFM2.5-1.2B-Instruct-Q4_0.gguf", "exists": true, "verified": true, "size_bytes": 734003200},
    {"id": "vision-model", "filename": "moondream2-text-model-f16_ct-vicuna.gguf", "exists": false, "verified": false, "size_bytes": 0},
    {"id": "vision-projector", "filename": "moondream2-mmproj-f16-20250414.gguf", "exists": false, "verified": false, "size_bytes": 0}
  ]
}}
```

### 5.2 New Request: `download_models`

```json
{"id": 2, "method": "download_models", "params": {"modelsDir": "/path/to/models"}}
```

Progress events (broadcast, id=null):
```json
{"id": null, "type": "setup_progress", "data": {
  "model_id": "vision-model",
  "filename": "moondream2-text-model-f16_ct-vicuna.gguf",
  "downloaded_bytes": 423624704,
  "total_bytes": 843456512,
  "status": "downloading"
}}
```

Per-model completion:
```json
{"id": null, "type": "setup_progress", "data": {
  "model_id": "vision-model",
  "status": "complete",
  "verified": true
}}
```

Error:
```json
{"id": null, "type": "setup_progress", "data": {
  "model_id": "vision-model",
  "status": "error",
  "error": "Network connection lost"
}}
```

Final result:
```json
{"id": 2, "type": "result", "data": {"status": "all_models_ready"}}
```

### 5.3 New ResponseType

Add to `protocol.ts`:
```typescript
export type ResponseType = "result" | "token" | "agent_step" | "error" | "status"
  | "progress" | "log" | "directory_update" | "file_graph"
  | "setup_progress";   // NEW

export interface SetupProgressData {
  model_id: string;
  filename?: string;
  downloaded_bytes?: number;
  total_bytes?: number;
  status: "downloading" | "verifying" | "complete" | "error" | "skipped";
  verified?: boolean;
  error?: string;
}
```

## 6. Startup Sequence

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Electron  │     │ Python   │     │ Renderer │     │ HF Hub   │
│ Main      │     │ Backend  │     │ (React)  │     │          │
└────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │                 │
     │  spawn python  │                │                 │
     ├───────────────►│                │                 │
     │                │                │                 │
     │ check_models   │                │                 │
     ├───────────────►│                │                 │
     │                │ scan models dir│                 │
     │                │◄──────────────►│                 │
     │  result: {ready: false, ...}    │                 │
     │◄───────────────┤                │                 │
     │                │                │                 │
     │  IPC: show SetupScreen          │                 │
     ├────────────────────────────────►│                 │
     │                │                │ render setup UI │
     │                │                │────────────────►│
     │                │                │                 │
     │ download_models│                │                 │
     ├───────────────►│                │                 │
     │                │  hf_hub_download per model       │
     │                │────────────────────────────────►│
     │                │                │                 │
     │  setup_progress│(broadcast)     │                 │
     │◄───────────────┤                │                 │
     │  relay to renderer              │                 │
     ├────────────────────────────────►│ update progress │
     │                │                │────────────────►│
     │                │                │                 │
     │  ... (repeat per model)         │                 │
     │                │                │                 │
     │  result: all_models_ready       │                 │
     │◄───────────────┤                │                 │
     │  IPC: setup complete            │                 │
     ├────────────────────────────────►│ show "Get       │
     │                │                │  Started"       │
     │                │                │                 │
     │  User clicks "Get Started"      │                 │
     │◄────────────────────────────────┤                 │
     │  transition to main UI          │                 │
     ├────────────────────────────────►│                 │
     │                │                │                 │

  --- Subsequent launches (models exist) ---

     │  spawn python  │                │                 │
     ├───────────────►│                │                 │
     │ check_models   │                │                 │
     ├───────────────►│                │                 │
     │  result: {ready: true}          │                 │
     │◄───────────────┤                │                 │
     │  IPC: skip setup, show main UI  │                 │
     ├────────────────────────────────►│                 │
```

## 7. Model Manifest File

New file: `models-manifest.json` (bundled in app resources)

```json
{
  "version": 1,
  "models": [
    {
      "id": "text-model",
      "name": "LFM2.5-1.2B Instruct",
      "filename": "LFM2.5-1.2B-Instruct-Q4_0.gguf",
      "repo_id": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
      "size_bytes": 734003200,
      "sha256": "TODO_COMPUTE_HASH",
      "required": true
    },
    {
      "id": "vision-model",
      "name": "Moondream2 Vision",
      "filename": "moondream2-text-model-f16_ct-vicuna.gguf",
      "repo_id": "ggml-org/moondream2-20250414-GGUF",
      "size_bytes": 843456512,
      "sha256": "TODO_COMPUTE_HASH",
      "required": true
    },
    {
      "id": "vision-projector",
      "name": "Moondream2 Projector",
      "filename": "moondream2-mmproj-f16-20250414.gguf",
      "repo_id": "ggml-org/moondream2-20250414-GGUF",
      "size_bytes": 18874368,
      "sha256": "TODO_COMPUTE_HASH",
      "required": true
    }
  ]
}
```

Both Python backend and Electron main process read this file. Single source of truth.

## 8. Build Pipeline Changes

### Current `electron-builder.yml`:
```yaml
extraResources:
  - from: "../dist/manole-server"
    to: "manole-server"
  - from: "../models/"    # <-- REMOVE THIS
    to: "models/"         # <-- REMOVE THIS
```

### New `electron-builder.yml`:
```yaml
extraResources:
  - from: "../dist/manole-server"
    to: "manole-server"
  - from: "../models-manifest.json"
    to: "models-manifest.json"
```

### Build steps:
1. `pyinstaller server.spec` → `dist/manole-server`
2. `cd ui && npm run build` → Electron renderer/main/preload
3. `cd ui && npx electron-builder` → `.dmg` (macOS) / `.AppImage` (Linux)

No models in the build. Installer shrinks from ~3 GB to ~150 MB.

## 9. `ModelManager` Path Resolution

```python
# models.py — updated path resolution

import sys
import platform
from pathlib import Path

def get_models_dir() -> Path:
    """Resolve platform-appropriate models directory."""
    # Environment variable override (set by Electron)
    env_dir = os.environ.get("MANOLE_MODELS_DIR")
    if env_dir:
        return Path(env_dir)

    # PyInstaller frozen binary
    if getattr(sys, 'frozen', False):
        if platform.system() == "Darwin":
            return Path.home() / "Library" / "Application Support" / "Manole" / "models"
        else:  # Linux
            xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
            return Path(xdg) / "manole" / "models"

    # Dev mode — current directory
    return Path("models")
```

## 10. Open Questions Resolved

| # | Question | Resolution |
|---|----------|------------|
| 1 | macOS code signing vs writable Resources | Models go to `~/Library/Application Support/`. Resources/ stays read-only and sealed. |
| 2 | Linux AppImage model path | Models go to `~/.local/share/manole/models/` (XDG compliant). |
| 3 | HuggingFace download reliability | Use `huggingface_hub.hf_hub_download` directly — it has built-in resume, retries, and progress callbacks. Wrap with NDJSON progress reporting. |
| 4 | NDJSON protocol extension | New `setup_progress` event type + `check_models` and `download_models` methods. Defined in Section 5. |
| 5 | Universal binary (macOS) | Defer. PyInstaller builds for the host architecture. ARM-only for now (user's MacBook). Cross-compilation is a separate concern. |

## 11. File Changes Summary

### New Files
| File | Purpose |
|------|---------|
| `models-manifest.json` | Model list with sizes, repos, SHA256 |
| `ui/src/components/SetupScreen.tsx` | First-launch setup UI |
| `ui/electron/setup.ts` | `ModelSetupManager` — orchestrates model check + download |

### Modified Files
| File | Change |
|------|--------|
| `ui/electron-builder.yml` | Remove `models/`, add `models-manifest.json` |
| `ui/electron/main.ts` | Integrate `ModelSetupManager` before window show |
| `ui/electron/python.ts` | Pass `MANOLE_MODELS_DIR` env var to Python process |
| `ui/src/App.tsx` | Add `setup_needed` / `setup_in_progress` / `setup_complete` states |
| `ui/src/lib/protocol.ts` | Add `SetupProgressData` type and `setup_progress` response type |
| `models.py` | `ModelManager` uses `get_models_dir()` for path resolution |
| `server.py` | Add `handle_check_models()` and `handle_download_models()` methods |

### Deleted/Reduced
| Change | Impact |
|--------|--------|
| `models/` no longer in `extraResources` | Installer shrinks by ~1.5 GB |
