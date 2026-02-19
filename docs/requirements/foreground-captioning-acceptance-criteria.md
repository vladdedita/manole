# Acceptance Criteria: Foreground Image Captioning

## AC-1: Backend sends correct status sequence

**Given** a directory with uncached images
**When** `handle_init` is called
**Then** status events are sent in order: `loading_model` → `indexing` → `summarizing` → `captioning` → `ready`
**And** `captioning_progress` events are sent during the captioning phase
**And** `directory_update { state: "ready" }` is sent only after captioning completes

## AC-2: Backend skips captioning status when no uncached images

**Given** a directory with no images (or all cached)
**When** `handle_init` is called
**Then** no `captioning` status event is sent
**And** the sequence is: `loading_model` → `indexing` → `summarizing` → `ready`

## AC-3: LoadingScreen shows dynamic steps

**Given** the backend is in `summarizing` state
**When** the LoadingScreen renders
**Then** "Generating summary" is shown as the active step
**And** "Loading model" and "Indexing files" show as completed

## AC-4: LoadingScreen shows captioning progress

**Given** the backend is in `captioning` state
**And** captioning progress is `done=3, total=10`
**When** the LoadingScreen renders
**Then** "Captioning images (3/10)" is shown as the active step

## AC-5: LoadingScreen hides captioning step when not needed

**Given** the backend transitions from `summarizing` directly to `ready`
**When** the LoadingScreen renders
**Then** no captioning step is ever displayed

## AC-6: Chat unavailable during loading

**Given** the directory state is `indexing` (covers all loading sub-states)
**When** the user views the app
**Then** the LoadingScreen is shown instead of the ChatPanel
**And** the chat input is not accessible

## AC-7: Errors don't block ready

**Given** summary generation throws an exception
**When** `handle_init` continues
**Then** captioning still runs (if applicable)
**And** "ready" is still sent

**Given** all image captioning fails
**When** `handle_init` continues
**Then** "ready" is still sent
**And** the chat interface becomes available

## AC-8: Reindex uses foreground flow

**Given** a directory is already loaded
**When** the user triggers reindex
**Then** the same foreground pipeline runs
**And** LoadingScreen is shown with the same step progression
