# User Stories: Foreground Image Captioning

## US-1: Foreground loading pipeline (backend)

**As** the server initialization handler,
**I want** to run summary generation and image captioning before sending "ready",
**So that** the model is not busy when the user starts chatting.

### Acceptance Criteria (Given/When/Then)

```gherkin
Given a directory with 5 uncached images
When handle_init processes the directory
Then "summarizing" status is sent before summary starts
And "captioning" status is sent before captioning starts
And captioning_progress events are sent as images are captioned
And "ready" is sent only after both summary and captioning complete
And the directory state is "ready" with stats and summary populated
```

### Tasks
- Remove `_background_tasks()` thread from `handle_init`
- Run summary generation inline after index build
- Send `status { state: "summarizing" }` before summary
- Send `status { state: "captioning" }` before captioning (only if uncached images exist)
- Run `ImageCaptioner.run()` inline
- Reload LeannSearcher inline after captioning
- Send `directory_update { state: "ready" }` and `status { state: "ready" }` at the end

---

## US-2: Dynamic LoadingScreen steps (frontend)

**As** a user watching the loading screen,
**I want** to see which phase the app is in, including captioning progress,
**So that** I understand why loading takes time and know it's working.

### Acceptance Criteria (Given/When/Then)

```gherkin
Given the backend sends status "summarizing"
When the LoadingScreen renders
Then a "Generating summary" step is shown as active

Given the backend sends status "captioning"
When the LoadingScreen renders
Then a "Captioning images" step is shown as active

Given captioning_progress events arrive with done=3, total=10
When the LoadingScreen renders
Then the captioning step shows "Captioning images (3/10)"

Given the backend never sends "captioning" status (no uncached images)
When the LoadingScreen renders the full sequence
Then no captioning step is visible at any point
```

### Tasks
- Replace static `STEPS` array with dynamic step computation based on `backendState`
- Accept `captioningProgress` prop from App.tsx
- Show `(done/total)` in the captioning step label
- Steps that haven't been reached yet stay hidden or greyed out

---

## US-3: Thread captioning progress to LoadingScreen (frontend wiring)

**As** the App component,
**I want** to pass captioning progress data to LoadingScreen,
**So that** the loading step can show the live counter.

### Acceptance Criteria

```gherkin
Given captioning_progress events arrive on the subscribe channel
When the active directory is in "indexing" state
Then LoadingScreen receives updated captioningProgress prop
And the progress is specific to the active directory
```

### Tasks
- Track captioning progress in App.tsx state (already partially done for SidePanel)
- Pass it as a prop to `LoadingScreen`
- Reset progress when directory changes or reaches ready
