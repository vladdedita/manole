# Journey: Foreground Image Captioning

## Problem

When a directory contains images, the captioning model runs in the background after "ready" is sent.
The model is locked (`_lock`) during captioning, so chat queries block silently. The user sees an
unresponsive chat with no feedback about why.

## Journey Map

### Scenario A: Directory with uncached images

```
 User clicks         Model loads        Files indexed       Summary          Captioning            Chat
 "Open Folder"       (if needed)        into vector DB      generated        images (3/12)         available
     |                   |                   |                  |                  |                  |
     v                   v                   v                  v                  v                  v
 ┌────────┐       ┌─────────────┐     ┌──────────┐      ┌───────────┐    ┌──────────────┐    ┌──────────┐
 │ Select │──────>│  Loading    │────>│ Indexing  │─────>│Summarizing│───>│  Captioning   │───>│  Ready   │
 │ folder │       │  model      │     │ files     │      │           │    │ images (3/12) │    │   ✓      │
 └────────┘       └─────────────┘     └──────────┘      └───────────┘    └──────────────┘    └──────────┘
                   [~5s first time]    [~2-10s]           [~2s]           [~3s per image]     [chat unlocks]

 Emotion: curious   patient           patient             patient          informed/patient     satisfied
```

### Scenario B: Directory with no images (or all cached)

```
 ┌────────┐       ┌─────────────┐     ┌──────────┐      ┌───────────┐    ┌──────────┐
 │ Select │──────>│  Loading    │────>│ Indexing  │─────>│Summarizing│───>│  Ready   │
 │ folder │       │  model      │     │ files     │      │           │    │   ✓      │
 └────────┘       └─────────────┘     └──────────┘      └───────────┘    └──────────┘
                                                                          [captioning
                                                                           step hidden]
```

### Scenario C: Re-index existing directory

```
 ┌─────────┐      ┌──────────┐      ┌───────────┐    ┌──────────────┐    ┌──────────┐
 │ Reindex │─────>│ Indexing  │─────>│Summarizing│───>│  Captioning   │───>│  Ready   │
 │ button  │      │ files     │      │           │    │ (new only)    │    │   ✓      │
 └─────────┘      └──────────┘      └───────────┘    └──────────────┘    └──────────┘
                                                       [skipped if none]
```

## Shared Artifacts

| Artifact | Source | Consumed By |
|----------|--------|-------------|
| `status.state` | server.py `send()` | LoadingScreen step progression |
| `captioning_progress.done/total` | ImageCaptioner.run() | LoadingScreen live counter |
| `directory_update.state` | server.py | App.tsx directory state machine |
| `self.model._lock` | ModelManager | Blocks concurrent LLM calls (root cause) |

## Error Paths

| Error | Handling |
|-------|----------|
| Single image fails to caption | Skip it, continue with rest (existing behavior) |
| All captioning fails | Still transition to "ready" (captions are optional enhancement) |
| Summary generation fails | Still proceed to captioning then ready |
