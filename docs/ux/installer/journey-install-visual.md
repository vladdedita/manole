# Manole Installer — User Journey Map

## Persona: Non-Technical User (Ana)
Ana found Manole online. She wants to search her local files with AI. She has a MacBook or Linux desktop, a working internet connection, and no command-line experience.

---

## Journey Overview

```
DISCOVER    DOWNLOAD     INSTALL       FIRST LAUNCH        SETUP          READY
  (web)    (.dmg/.AI)   (drag/run)    (app opens)     (models DL)    (fully offline)

   😊         😊           😊            🤔               😐→😊           🎉
 Curious    Excited     Familiar     "What's this      Waiting...      "It works!"
                        territory     setup screen?"    "Almost done"
```

---

## Phase 1: Discover & Download

| Step | Action | Emotion | Artifacts |
|------|--------|---------|-----------|
| 1.1 | Ana finds Manole website/GitHub | Curious | ${website_url} |
| 1.2 | Sees "Download for Mac" / "Download for Linux" buttons | Confident — clear choice | ${download_button} |
| 1.3 | Clicks download, gets a ~150 MB file | Comfortable — familiar size | ${installer_file}: `Manole-1.0.0.dmg` or `Manole-1.0.0.AppImage` |
| 1.4 | Download completes in 1-3 min | Satisfied | File in ~/Downloads |

**Emotional arc**: Smooth, standard. Nothing unexpected.

---

## Phase 2: Install (Platform-Specific)

### macOS (.dmg)

| Step | Action | Emotion |
|------|--------|---------|
| 2.1 | Double-clicks `Manole-1.0.0.dmg` | Familiar |
| 2.2 | DMG window opens: Manole icon + Applications folder | Very familiar |
| 2.3 | Drags Manole to Applications | Confident |
| 2.4 | Ejects DMG | Done — 10 seconds total |

### Linux (.AppImage)

| Step | Action | Emotion |
|------|--------|---------|
| 2.1 | Downloads `Manole-1.0.0.AppImage` | Familiar |
| 2.2 | `chmod +x` or right-click > Properties > Make Executable | Slightly technical but standard for Linux users |
| 2.3 | Double-clicks to run | Straightforward |

**Emotional arc**: Zero friction. Standard platform conventions.

---

## Phase 3: First Launch & Setup

This is the **critical moment** — the only non-standard part of the experience.

| Step | Action | Emotion | Duration |
|------|--------|---------|----------|
| 3.1 | Launches Manole for the first time | Expectant | instant |
| 3.2 | **Setup screen appears** instead of main UI | Slight surprise — "Oh, there's a setup step" | instant |
| 3.3 | Reads: "Welcome to Manole. Setting up for first use..." | Reassured — clear messaging | 2 sec |
| 3.4 | Sees model download list with sizes | Informed — knows what to expect | 2 sec |
| 3.5 | Download starts automatically | Passive waiting | — |
| 3.6 | Progress bar moves, completed items get checkmarks | Reassured — things are working | 3-15 min |
| 3.7 | All downloads complete, "Setup complete!" message | Relief, excitement | instant |
| 3.8 | "Get Started" button appears | Eager to try | instant |
| 3.9 | Clicks "Get Started" — main UI loads | Delight | instant |

**Emotional arc**: Brief uncertainty ("setup?") quickly resolved by clear progress feedback. Confidence builds as checkmarks appear. Reward at the end.

### Error Paths

| Error | User Sees | Recovery |
|-------|-----------|----------|
| Network lost mid-download | "Connection lost. Download will resume when connected." + retry button | Auto-retry on reconnect, resume from last byte |
| Slow connection | Progress bar + estimated time + "This may take a while on slower connections" | Patience messaging, no timeout |
| Disk full | "Not enough disk space. Manole needs ~2 GB free." + specific amount needed | Clear, actionable message |
| Download corrupted | "Download verification failed. Retrying..." (auto-retry) | SHA256 check, auto-redownload single file |
| User closes app mid-download | On next launch: setup resumes where it left off | Persistent progress tracking |

---

## Phase 4: Ongoing Use (Fully Offline)

| Step | Action | Emotion |
|------|--------|---------|
| 4.1 | Opens Manole anytime | Confident |
| 4.2 | App loads directly to main UI (no setup screen) | Expected |
| 4.3 | All features work without internet | Trust — "it just works" |
| 4.4 | No model download prompts ever again | Seamless |

**Emotional arc**: Pure delight. The promise of "works offline" is delivered.

---

## Shared Artifacts Registry

| Artifact | Source | Used In |
|----------|--------|---------|
| `${installer_file}` | Website download | Phase 1, Phase 2 |
| `${app_bundle}` | Installer output | Phase 2, Phase 3 |
| `${models_dir}` | Inside app bundle (Resources/) | Phase 3, Phase 4 |
| `${setup_progress}` | Persisted state file in app data | Phase 3 (resume support) |
| `${model_manifest}` | Embedded in app — list of required models + sizes + SHA256 | Phase 3 |
| `${download_url_base}` | HuggingFace CDN | Phase 3 |

---

## Key Design Principles

1. **Small initial download** — Installer is ~150 MB (Electron + PyInstaller binary, no models)
2. **One-time setup** — Models download on first launch only, never again
3. **Resumable** — Interrupted downloads resume from last byte
4. **Inside the bundle** — Models stored in app's Resources directory
5. **No terminal needed** — Entire flow is GUI-only for non-technical users
6. **Clear progress** — Per-model progress bars with sizes and checkmarks
7. **Offline forever** — After setup, zero internet dependency
