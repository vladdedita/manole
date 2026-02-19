# Vision Pipeline: Acceptance Criteria

**Feature**: LFM2.5-VL-1.6B Image Understanding
**Date**: 2026-02-17

---

## US-1: Background Image Captioning

### AC-1: Captioning starts after text indexing
```gherkin
Given a directory with text files and image files
When text indexing completes and the directory state becomes "ready"
Then background image captioning starts automatically
And queries are served immediately (not blocked by captioning)
```

### AC-2: All images are captioned
```gherkin
Given a directory with 10 image files (jpg, png, heic mix)
When background captioning runs to completion
Then all 10 images have captions stored in the cache
And all 10 captions are injected into the LEANN index
```

### AC-3: Captions injected immediately
```gherkin
Given background captioning is processing image 5 of 10
When image 5's caption is generated
Then the caption is injected into the LEANN index immediately
And a semantic search for that caption's content returns the image
```

### AC-4: Captioning doesn't block queries
```gherkin
Given background captioning is in progress (50 of 200 images done)
When a user submits a query
Then the query is processed normally using the current index state
And the response is returned without waiting for captioning to finish
```

### AC-5: Captioning handles errors gracefully
```gherkin
Given a directory with 10 images where image 3 is corrupted
When background captioning encounters image 3
Then image 3 is skipped with a warning log
And captioning continues with images 4 through 10
```

---

## US-2: Caption Persistence

### AC-6: Captions persist across sessions
```gherkin
Given 10 images were captioned in session 1
When the user opens the same directory in session 2
Then the caption cache returns all 10 captions
And no VL model inference is needed for those images
```

### AC-7: Cache invalidation on file change
```gherkin
Given image "photo.jpg" was captioned with mtime T1
When "photo.jpg" is modified (mtime becomes T2)
Then the cache returns None for "photo.jpg"
And the image is re-captioned during background processing
```

### AC-8: Cache key uniqueness
```gherkin
Given two different images at different paths
When both are captioned
Then each has a unique cache key (hash of path + mtime)
And captions don't collide
```

---

## US-3: Searchable Image Content

### AC-9: Semantic search finds captioned images
```gherkin
Given an image "photo1.jpg" captioned as "A tabby cat sitting on a wooden desk"
When the user asks "find a picture of a cat"
Then semantic search returns the caption chunk
And the response includes "photo1.jpg" with the caption
```

### AC-10: Caption chunk metadata
```gherkin
Given an image "sunset.png" is captioned
When the caption is injected into the LEANN index
Then the chunk text is "Photo description: {caption}"
And metadata includes file_name="sunset.png", file_type="image", path="{full_path}"
```

### AC-11: Uncaptioned images not in search
```gherkin
Given background captioning is 50% complete
When the user searches for image content
Then only already-captioned images appear in results
And uncaptioned images are not returned (no false negatives from metadata-only entries)
```

---

## US-4: Captioning Progress Visibility

### AC-12: Progress messages during captioning
```gherkin
Given background captioning is processing 200 images
When image N is captioned
Then an NDJSON message is sent:
  {"id": null, "type": "captioning_progress", "data": {"directoryId": "...", "done": N, "total": 200}}
```

### AC-13: Completion message
```gherkin
Given background captioning finishes all 200 images
Then a final NDJSON message is sent:
  {"id": null, "type": "captioning_progress", "data": {"directoryId": "...", "done": 200, "total": 200, "state": "complete"}}
```

### AC-14: No progress messages for text-only directories
```gherkin
Given a directory with no image files
When indexing completes
Then no captioning_progress messages are sent
```

---

## US-5: Vision Model Lifecycle

### AC-15: Lazy loading
```gherkin
Given a directory with image files
When text indexing completes
Then the VL model is NOT loaded yet
And the VL model loads only when the first image captioning begins
```

### AC-16: Text-only directories
```gherkin
Given a directory with no image files
When the directory is initialized and queries are served
Then the VL model is never loaded
And memory usage stays under 1.5 GB
```

### AC-17: Both models coexist
```gherkin
Given the VL model is loaded for image captioning
When the user submits a text query
Then the text model processes the query normally
And both models coexist without memory issues (< 2.5 GB total)
```

---

## US-6: Broad Image Format Support

### AC-18: Common formats supported
```gherkin
Given images in formats: jpg, jpeg, png, gif, webp, bmp, tiff
When background captioning processes each image
Then all images are captioned successfully
```

### AC-19: HEIC conversion
```gherkin
Given an Apple HEIC photo "IMG_2025.heic"
When background captioning encounters this file
Then the image is converted to JPEG in-memory via Pillow
And the VL model receives the JPEG data
And the caption is generated successfully
```

### AC-20: Unsupported format handling
```gherkin
Given an image file in an unsupported or corrupted format
When background captioning encounters this file
Then the file is skipped with a warning log
And captioning continues with remaining files
And no crash or exception propagates
```
