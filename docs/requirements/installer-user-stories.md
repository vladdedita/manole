# Manole Installer — User Stories

## Epic: INST — Installer with Online Model Download

---

### INST-01: Small Installer Build

**As a** non-technical user
**I want** to download a small installer (< 200 MB)
**So that** the initial download is fast and doesn't discourage me

**Acceptance Criteria**:
- Given the build pipeline runs, when the installer is produced, then the `.dmg` / `.AppImage` is under 200 MB
- Given the installer is built, when inspecting its contents, then no `.gguf` model files are included
- Given the installer is built, when inspecting its contents, then the `manole-server` binary is included

**Technical Notes**:
- Remove `models/` from `electron-builder.yml` extraResources
- PyInstaller binary stays bundled
- Electron + renderer stays bundled

---

### INST-02: Model Manifest

**As a** developer
**I want** a single model manifest embedded in the app
**So that** the setup screen knows exactly which models to download, their sizes, and checksums

**Acceptance Criteria**:
- Given the app starts, when the setup screen reads the manifest, then it knows all required model filenames, HuggingFace repo IDs, expected sizes, and SHA256 hashes
- Given a model is added or updated, when the manifest is updated, then the setup screen automatically reflects the change

**Technical Notes**:
- JSON file bundled with the app (e.g. `models-manifest.json`)
- Single source of truth — `models.py` reads from it too

---

### INST-03: First-Launch Setup Screen (Electron UI)

**As a** non-technical user
**I want** a friendly setup screen on first launch that downloads AI models with visible progress
**So that** I know what's happening and can trust the app is setting up correctly

**Acceptance Criteria**:
- Given I launch Manole for the first time, when the app opens, then I see a setup screen (not the main UI)
- Given the setup screen is showing, when I look at it, then I see a welcome message, a list of models with sizes, and a progress indicator
- Given models are downloading, when I watch the progress, then each model has its own progress bar and completed models show checkmarks
- Given all models finish downloading, when setup completes, then I see "Setup complete!" and a "Get Started" button
- Given I click "Get Started", when the transition happens, then the main UI loads with all features working

---

### INST-04: Model Download Engine (Python Backend)

**As a** developer
**I want** a model download engine in the Python backend that downloads from HuggingFace with resume support
**So that** large model files can be reliably downloaded even on unstable connections

**Acceptance Criteria**:
- Given a model needs downloading, when the download starts, then it uses HTTP range requests to support resuming
- Given a download is interrupted, when it resumes, then it continues from the last byte (not from scratch)
- Given a model file is fully downloaded, when verification runs, then SHA256 hash is checked against the manifest
- Given SHA256 verification fails, when the error is detected, then only that file is deleted and re-downloaded
- Given download progress occurs, when the backend reports it, then per-file byte counts are sent to the Electron UI via the NDJSON protocol

**Technical Notes**:
- Extend the existing NDJSON protocol with `setup_progress` events
- Use `huggingface_hub.hf_hub_download` which already supports resume
- Progress state persisted to disk for crash recovery

---

### INST-05: Setup Progress Persistence

**As a** user who closed the app during setup
**I want** the download to resume where it left off when I relaunch
**So that** I don't have to re-download gigabytes of data

**Acceptance Criteria**:
- Given models are partially downloaded, when I close and reopen the app, then the setup screen shows again
- Given I reopen the app, when downloads resume, then completed files are skipped and partial files resume
- Given all files eventually complete, when setup finishes, then subsequent launches go directly to main UI

---

### INST-06: Error Handling During Setup

**As a** user on an unreliable connection
**I want** clear error messages and automatic recovery during model download
**So that** I don't get stuck or confused if something goes wrong

**Acceptance Criteria**:
- Given my network drops during download, when the error occurs, then I see "Connection lost. Download will resume when connected." and a retry button
- Given my disk is full, when setup detects it, then I see a message with the exact space needed
- Given a download is corrupted, when SHA256 check fails, then only that file retries automatically
- Given the HuggingFace server is unavailable, when retry happens, then it uses exponential backoff

---

### INST-07: Platform-Specific Model Storage

**As a** developer
**I want** models stored in the right location per platform
**So that** the app can find and use them reliably

**Acceptance Criteria**:
- Given macOS, when models are downloaded, then they are stored in a writable directory accessible to the app bundle
- Given Linux AppImage, when models are downloaded, then they are stored in `~/.local/share/manole/models/`
- Given models are stored, when `ModelManager` initializes, then it finds models in the platform-appropriate directory
- Given the model path is resolved, when the app runs offline, then all models load successfully

**Technical Notes**:
- macOS: `~/Library/Application Support/Manole/models/` (since Resources/ inside .app is read-only after code signing)
- Linux: `~/.local/share/manole/models/`
- `ModelManager` paths must be updated to check platform-specific locations

---

### INST-08: Subsequent Launch Bypass

**As a** returning user
**I want** the app to start instantly without any setup screen
**So that** my experience after initial setup is seamless

**Acceptance Criteria**:
- Given all required models exist in the models directory, when I launch Manole, then the main UI loads directly
- Given I have no internet connection, when I launch Manole, then the app works normally
- Given model files exist, when the app starts, then no SHA256 re-verification occurs (only during setup)

---

## Story Map

```
          INST-01          INST-02
        (small build)    (manifest)
              \             /
               \           /
          INST-03  ←→  INST-04
        (setup UI)    (download engine)
              |             |
         INST-06       INST-05
       (error UX)    (persistence)
              \           /
               \         /
              INST-07
          (platform paths)
                 |
              INST-08
          (launch bypass)
```

**Critical path**: INST-02 → INST-04 → INST-03 → INST-05 → INST-08

---

## Priority Order

| Priority | Story | Rationale |
|----------|-------|-----------|
| P0 | INST-01 | Must-have: small build is the foundation |
| P0 | INST-02 | Must-have: manifest drives everything else |
| P0 | INST-04 | Must-have: download engine is the core mechanism |
| P0 | INST-03 | Must-have: user-facing setup experience |
| P0 | INST-07 | Must-have: models need a writable home |
| P1 | INST-05 | Important: resume on relaunch |
| P1 | INST-06 | Important: error recovery |
| P1 | INST-08 | Important: skip setup on subsequent launches |
