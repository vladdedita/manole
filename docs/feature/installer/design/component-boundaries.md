# Manole Installer — Component Boundaries

## Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        ELECTRON APP                              │
│                                                                  │
│  ┌─── Main Process ──────────────────────────────────────────┐  │
│  │                                                            │  │
│  │  main.ts (MODIFIED)                                        │  │
│  │  ├── On app ready:                                         │  │
│  │  │   1. createWindow()                                     │  │
│  │  │   2. python.spawn()                                     │  │
│  │  │   3. setupManager.checkAndDownload()  ◄── NEW           │  │
│  │  │   4. Resume normal IPC handlers                         │  │
│  │  │                                                         │  │
│  │  setup.ts (NEW)                                            │  │
│  │  ├── ModelSetupManager                                     │  │
│  │  │   ├── readManifest(): ModelManifest                     │  │
│  │  │   ├── checkModels(): CheckResult                        │  │
│  │  │   ├── downloadModels(): void                            │  │
│  │  │   └── getModelsDir(): string                            │  │
│  │  │                                                         │  │
│  │  python.ts (MODIFIED)                                      │  │
│  │  ├── PythonBridge                                          │  │
│  │  │   ├── spawn() — pass MANOLE_MODELS_DIR env var          │  │
│  │  │   └── (rest unchanged)                                  │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─── Renderer Process ─────────────────────────────────────┐   │
│  │                                                           │   │
│  │  App.tsx (MODIFIED)                                       │   │
│  │  ├── New top-level state: appPhase                        │   │
│  │  │   "checking" → "setup" → "ready"                       │   │
│  │  ├── if appPhase === "setup": <SetupScreen />             │   │
│  │  ├── if appPhase === "ready": (existing app)              │   │
│  │  │                                                        │   │
│  │  SetupScreen.tsx (NEW)                                    │   │
│  │  ├── Props: models[], onComplete()                        │   │
│  │  ├── Per-model row: icon + name + size + progress bar     │   │
│  │  ├── Error display + retry                                │   │
│  │  └── "Get Started" button                                 │   │
│  │                                                           │   │
│  │  protocol.ts (MODIFIED)                                   │   │
│  │  ├── + SetupProgressData interface                        │   │
│  │  ├── + "setup_progress" in ResponseType                   │   │
│  │  │                                                        │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     PYTHON BACKEND                               │
│                                                                  │
│  server.py (MODIFIED)                                            │
│  ├── + handle_check_models(req_id, params)                       │
│  │     → scans models dir, returns per-model exists/verified     │
│  ├── + handle_download_models(req_id, params)                    │
│  │     → downloads missing models, streams progress events       │
│  └── dispatch table: + "check_models", "download_models"         │
│                                                                  │
│  models.py (MODIFIED)                                            │
│  ├── + get_models_dir() → Path                                   │
│  │     → platform-aware path resolution                          │
│  ├── ModelManager.__init__                                       │
│  │     → uses get_models_dir() for DEFAULT_*_PATH                │
│  └── _ensure_model                                               │
│       → unchanged (still works for dev mode)                     │
│                                                                  │
│  models-manifest.json (NEW, bundled in resources)                │
│  └── Single source of truth for model list                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

### First Launch
```
Electron main                Python backend              HuggingFace
     │                            │                          │
     │──── check_models ─────────►│                          │
     │◄─── {ready: false} ────────│                          │
     │                            │                          │
     │ tell renderer: show setup  │                          │
     │                            │                          │
     │──── download_models ──────►│                          │
     │                            │──── hf_hub_download ────►│
     │◄─── setup_progress ────────│◄─── bytes ──────────────│
     │ relay to renderer          │                          │
     │                            │──── hf_hub_download ────►│
     │◄─── setup_progress ────────│◄─── bytes ──────────────│
     │                            │                          │
     │◄─── result: ready ─────────│                          │
     │ tell renderer: setup done  │                          │
```

### Subsequent Launch
```
Electron main                Python backend
     │                            │
     │──── check_models ─────────►│
     │◄─── {ready: true} ─────────│
     │                            │
     │ tell renderer: skip setup  │
```

## Interface Contracts

### Electron → Python (NDJSON requests)

```typescript
// check_models
{ id: number, method: "check_models", params: { modelsDir: string } }

// download_models
{ id: number, method: "download_models", params: { modelsDir: string } }
```

### Python → Electron (NDJSON responses)

```typescript
// setup_progress broadcast
{ id: null, type: "setup_progress", data: SetupProgressData }

// check_models result
{ id: number, type: "result", data: { ready: boolean, models: ModelStatus[] } }

// download_models result
{ id: number, type: "result", data: { status: "all_models_ready" } }
```

### Electron Main → Renderer (IPC)

```typescript
// Existing channel, new event types
ipcRenderer.on('python:message', (response) => {
  if (response.type === 'setup_progress') { /* update setup UI */ }
})

// New IPC channel for setup state
ipcRenderer.on('setup:state', (state: 'checking' | 'needed' | 'complete'))
```

## Boundary Rules

1. **Electron main process** owns the lifecycle — it decides whether setup is needed and orchestrates the flow
2. **Python backend** owns all model operations — checking, downloading, verifying
3. **Renderer** is purely presentational — shows whatever state main process tells it
4. **`models-manifest.json`** is the single source of truth — read by both Electron and Python
5. **No model logic in renderer** — it doesn't know about HuggingFace, SHA256, or file paths
6. **No UI logic in Python** — it just sends progress numbers via NDJSON
