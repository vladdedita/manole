# Manole Installer — Requirements

## REQ-01: Small Installer Package

The distributable installer must contain the Electron app + PyInstaller backend binary but **not** the AI model files. Total installer size must be under 200 MB.

**Rationale**: A 2-4 GB download discourages non-technical users. Separating models allows a fast initial download with deferred model acquisition.

| Platform | Format | Contents |
|----------|--------|----------|
| macOS | `.dmg` | `Manole.app` bundle with Electron + `manole-server` binary |
| Linux | `.AppImage` | Electron + `manole-server` binary |

---

## REQ-02: First-Launch Setup Screen

On first launch, when required model files are not present in the app's models directory, the app must display a setup screen instead of the main UI.

**Behavior**:
- Detects missing models by checking `${models_dir}` against an embedded **model manifest**
- Shows welcome message: "Welcome to Manole. Setting up for first use..."
- Lists each required model with name and size
- Begins downloading automatically (no "Start" button needed)
- Shows per-model progress bar + checkmarks on completion
- Shows overall progress
- Displays "This only happens once. After setup, Manole works completely offline."

---

## REQ-03: Model Manifest

The app must embed a model manifest — a structured list of all required model files with:

| Field | Example |
|-------|---------|
| `id` | `text-model` |
| `filename` | `LFM2.5-1.2B-Instruct-Q4_0.gguf` |
| `repo_id` | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` |
| `size_bytes` | `734003200` |
| `sha256` | `a1b2c3...` |
| `required` | `true` |

This manifest is the single source of truth for which models to download and verify.

---

## REQ-04: Resumable Downloads

Model downloads must support HTTP range requests for resume capability:
- Track bytes downloaded per file in a persistent state file
- On interruption (network loss, app close), save progress
- On resume, send `Range` header to continue from last byte
- Verify completed files with SHA256 checksum
- If checksum fails, delete and re-download that single file

---

## REQ-05: Model Storage Location

Downloaded models must be stored inside the app bundle:

| Platform | Path |
|----------|------|
| macOS | `Manole.app/Contents/Resources/models/` |
| Linux | Alongside the AppImage extraction or in `$XDG_DATA_HOME/manole/models/` |

**Note**: For macOS, writing to `Contents/Resources/` after installation requires the app to not be code-signed with a sealed resources flag, OR models must be stored in a writable companion directory. This is a technical constraint to resolve during DESIGN wave.

---

## REQ-06: Setup Progress Persistence

The setup state must be persisted so that:
- Closing the app mid-download doesn't lose progress
- Relaunching resumes from where it left off
- Completed files are not re-downloaded
- State file location: platform-appropriate app data directory

---

## REQ-07: Error Handling

| Error | User-Facing Message | Behavior |
|-------|---------------------|----------|
| Network lost | "Connection lost. Download will resume when connected." | Show retry button, auto-retry on reconnect |
| Slow connection | Estimated time shown, patience messaging | No timeout, continue indefinitely |
| Disk full | "Not enough disk space. Manole needs X.X GB free." | Block download, show actionable message |
| Corrupted file | "Verification failed. Retrying..." | Auto-delete + re-download single file |
| HuggingFace down | "Download server unavailable. Will retry shortly." | Exponential backoff retry |

---

## REQ-08: Subsequent Launch Bypass

On any launch where all required model files are present and verified:
- Skip setup screen entirely
- Load main UI directly
- No internet connection required
- No model integrity check on every launch (only during setup)

---

## REQ-09: Build Pipeline Changes

The build pipeline must be modified:

### Current (monolithic)
```
electron-builder → bundles manole-server + models/ → 2-4 GB .dmg
```

### Target (split)
```
electron-builder → bundles manole-server only → ~150 MB .dmg
                    models downloaded at first launch via setup screen
```

Changes to `electron-builder.yml`:
- Remove `models/` from `extraResources`
- Keep `manole-server` binary in `extraResources`

---

## REQ-10: macOS DMG Customization

The `.dmg` should have a branded background image with:
- Manole logo
- Arrow indicating "drag to Applications"
- Standard macOS installer UX conventions

---

## REQ-11: Linux AppImage Considerations

The AppImage format is read-only by design. Models cannot be written inside the AppImage after creation. Therefore:
- Models must be stored in a writable external directory
- Recommended: `~/.local/share/manole/models/` (XDG compliant)
- The app detects platform and uses appropriate model path
- On first launch, same setup screen experience as macOS

---

## Non-Functional Requirements

| NFR | Target |
|-----|--------|
| Installer download size | < 200 MB |
| First-launch setup time (100 Mbps) | < 3 minutes |
| First-launch setup time (10 Mbps) | < 15 minutes |
| Subsequent launch time | No change from current |
| Disk space required | ~2.5 GB (app + models) |
| Platforms | macOS (ARM + Intel), Linux (x86_64) |
