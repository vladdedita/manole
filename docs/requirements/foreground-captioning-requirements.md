# Requirements: Foreground Image Captioning

## Context

The LLM model is a shared resource with a mutex lock. When image captioning runs in the background
after the UI shows "ready", chat queries block silently on the lock. Moving captioning to the
foreground loading phase provides honest feedback and prevents a confusing unresponsive chat.

## Functional Requirements

### FR-1: Sequential foreground loading pipeline
The server must run summary generation and image captioning **before** sending the "ready" state.
The loading pipeline is: model load → index build → summary → captioning → ready.

### FR-2: New status events for summary and captioning phases
The server must send `status` events with states `"summarizing"` and `"captioning"` so the UI
can show the correct loading step.

### FR-3: Dynamic LoadingScreen steps
The LoadingScreen must display steps dynamically based on the current backend state, including
`"summarizing"` and `"captioning"` steps. The captioning step must show a live `(done/total)` counter.

### FR-4: Skip captioning step when no uncached images
If the directory has zero uncached images (either no images at all, or all cached), the
`"captioning"` status event must not be sent, and the step must not appear in the LoadingScreen.

### FR-5: Error resilience
If summary generation or image captioning fails, the server must still proceed to "ready".
Captioning errors for individual images must be skipped (existing behavior preserved).

### FR-6: Reindex uses same foreground flow
When the user triggers reindex, the same foreground pipeline applies (including captioning new images).

## Non-Functional Requirements

### NFR-1: No new dependencies
Use existing event system (`send()` function, `subscribe` in useChat).

### NFR-2: Preserve existing captioning_progress events
The `captioning_progress` events from `ImageCaptioner.run()` must continue to be sent
for the live counter in LoadingScreen.

### NFR-3: Cached-only case must be fast
When all images are cached, caption injection is instant. The UI must not show a visible
captioning step that would suggest slowness.
