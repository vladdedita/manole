# Vision Pipeline: User Journey Map

**Feature**: LFM2.5-VL-1.6B Image Understanding
**Type**: Backend
**Date**: 2026-02-17

## Journey Overview

The vision pipeline adds image understanding to manole's existing text-based RAG system.
Users don't interact with it directly — it transparently enhances search results to include
image content when relevant.

## Actors

- **End User**: Asks questions via the chat UI
- **System (Background)**: Indexes images automatically after text indexing completes

## Journey: "Find a picture of a cat"

```
User opens directory    System indexes text     Background captioning     User asks query
with mixed files        files (fast)            starts (async)            about images
       |                      |                        |                       |
       v                      v                        v                       v
  +---------+          +------------+          +---------------+        +-----------+
  | TRIGGER |--init--->| TEXT INDEX  |--done--->| IMAGE CAPTION |        |   QUERY   |
  |         |          | (existing) |          | (background)  |        |           |
  +---------+          +------------+          +----+----------+        +-----+-----+
                                                    |                        |
                              +---------------------+                        |
                              |                                              |
                              v                                              v
                       +-------------+                              +----------------+
                       | UI shows    |                              | Semantic search |
                       | "captioning |                              | finds caption   |
                       |  images..." |                              | chunks in index |
                       +-------------+                              +-------+--------+
                              |                                             |
                              v                                             v
                       +-------------+                              +----------------+
                       | Each caption|                              | Agent returns   |
                       | injected    |                              | "photo1.jpg     |
                       | into index  |                              |  shows a cat"   |
                       +-------------+                              +----------------+
```

## States & Transitions

| State | Trigger | Next State | UI Indicator |
|-------|---------|------------|--------------|
| `indexing` | User opens directory | `ready` | "Indexing files..." |
| `ready` | Text indexing complete | `captioning_images` | Directory ready |
| `captioning_images` | Background captioning starts | `ready` | "Captioning images (N/M)..." |
| `ready` | All images captioned | — | No indicator |

## Emotional Arc

| Phase | User Feeling | System State |
|-------|-------------|--------------|
| Open directory | Expectation | Indexing text (fast) |
| Start chatting | Satisfaction | Queries work immediately |
| Ask about images | Curiosity | Searches available captions |
| Get image results | Delight | Captions found in index |
| Images still captioning | Patience | Progress indicator visible |
| All captioned | Confidence | Full image search available |

## Error Paths

| Error | Handling |
|-------|---------|
| VL model fails to load | Log error, skip image captioning, text search still works |
| Image file corrupted/unreadable | Skip file, log warning, continue with remaining images |
| HEIC conversion fails | Skip file, log warning |
| Out of memory (both models loaded) | Unload VL model, report error, text search still works |
| Captioning interrupted (session ends) | Resume on next session — cache tracks what's done |
