# Manole Installer — Technology Stack

## No New Dependencies

The installer feature requires **zero new dependencies**. Everything is built with the existing stack.

### Python Side

| Technology | Already In Use | Role in Installer |
|------------|---------------|-------------------|
| `huggingface_hub` | Yes (`pyproject.toml`) | Download models from HF with resume support |
| `hashlib` (stdlib) | Yes | SHA256 verification of downloaded files |
| `pathlib` (stdlib) | Yes | Platform path resolution |
| PyInstaller | Yes (dev dep) | Bundle `manole-server` binary |

`huggingface_hub.hf_hub_download` provides:
- HTTP range request resume (built-in)
- Local caching and deduplication
- Progress callbacks via `tqdm` (we'll tap the callback for NDJSON events)
- SHA256 verification (via `hf_hub_download(..., force_download=False)`)

### Electron Side

| Technology | Already In Use | Role in Installer |
|------------|---------------|-------------------|
| `electron` | Yes | Main process orchestration, IPC |
| `electron-builder` | Yes | Build `.dmg` and `.AppImage` |
| React + Motion | Yes | SetupScreen UI with progress animations |
| NDJSON protocol | Yes | Progress events from Python to UI |

### Build Side

| Technology | Already In Use | Role in Installer |
|------------|---------------|-------------------|
| `electron-builder` | Yes | Packaging (modified config only) |
| PyInstaller | Yes | Python binary (unchanged) |

## What Changes

| Area | Before | After |
|------|--------|-------|
| `electron-builder.yml` | Bundles `models/` (~1.5 GB) | Bundles `models-manifest.json` (~1 KB) |
| `models.py` | Hardcoded `models/` path | Platform-aware `get_models_dir()` |
| `server.py` | No model check/download methods | +2 new NDJSON methods |
| `protocol.ts` | No setup events | +`setup_progress` type |
| UI components | No setup screen | +`SetupScreen.tsx` |
| Electron main | No model orchestration | +`setup.ts` module |

## Why No New Dependencies

The `huggingface_hub` library — already a required dependency for model downloading at runtime — handles all the hard parts: resumable HTTP downloads, caching, file verification. We just need to wire its progress callbacks into the existing NDJSON protocol, and add a React component to display the progress. The architectural change is mostly **plumbing**, not new capability.
