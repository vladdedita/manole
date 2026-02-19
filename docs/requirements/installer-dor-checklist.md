# Manole Installer — Definition of Ready Checklist

## DoR Validation for Epic INST (Installer with Online Model Download)

| # | DoR Item | Status | Evidence |
|---|----------|--------|----------|
| 1 | **User story follows INVEST criteria** | PASS | Stories are Independent (can be built separately), Negotiable (technical approach flexible), Valuable (each delivers user value), Estimable (clear scope), Small (single concern each), Testable (Gherkin scenarios written) |
| 2 | **Acceptance criteria are testable** | PASS | All criteria follow Given-When-Then format in `journey-install.feature`. Each can be automated or manually verified. |
| 3 | **Dependencies identified** | PASS | Dependencies: `huggingface_hub` (already in deps), `electron-builder` config changes, NDJSON protocol extension. No external blockers. |
| 4 | **Technical constraints documented** | PASS | Key constraints: macOS .app is read-only after signing (models go to App Support), AppImage is read-only (models go to XDG dir), HuggingFace rate limits. Documented in requirements. |
| 5 | **UX journey mapped** | PASS | Full journey map with emotional arcs, error paths, and shared artifacts in `journey-install-visual.md` and `journey-install.yaml` |
| 6 | **Shared artifacts tracked** | PASS | Artifact registry in journey visual doc: `installer_file`, `app_bundle`, `models_dir`, `setup_progress`, `model_manifest`, `download_url_base` |
| 7 | **Emotional arc coherent** | PASS | Progression: curious → confident → slight surprise → reassured → relief → delight. No jarring transitions. The "surprise" at setup screen is immediately resolved by clear messaging. |
| 8 | **No ambiguous steps** | PASS | Every step has concrete action, expected output, and duration estimate. Error paths have specific recovery actions. |

## Open Questions for DESIGN Wave

1. **macOS code signing**: If the app is code-signed with hardened runtime, `Contents/Resources/` is sealed. Models must go to `~/Library/Application Support/Manole/models/`. The DESIGN wave must confirm the signing strategy.

2. **Linux AppImage model path**: AppImage is read-only. Models go to `~/.local/share/manole/models/`. The DESIGN wave must decide if a symlink or environment variable approach is cleaner.

3. **HuggingFace download reliability**: `hf_hub_download` has built-in resume and caching. The DESIGN wave should evaluate whether to use it directly or wrap with custom progress reporting.

4. **NDJSON protocol extension**: The setup screen needs progress events. The DESIGN wave must define the exact message format for `setup_progress` events.

5. **Universal binary (macOS)**: Should the macOS build support both ARM and Intel? This affects the PyInstaller binary and Electron build.

## Verdict

**READY for DESIGN wave.** All 8 DoR items pass. Five open questions identified for the solution architect to resolve.
