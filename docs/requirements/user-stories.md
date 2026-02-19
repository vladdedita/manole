# Vision Pipeline: User Stories

**Feature**: LFM2.5-VL-1.6B Image Understanding
**Date**: 2026-02-17

---

## US-1: Background Image Captioning

**As** the manole system
**I want to** automatically caption all images in a directory after text indexing completes
**So that** image content becomes searchable without manual intervention

**Acceptance Criteria**: See AC-1 through AC-5

---

## US-2: Caption Persistence

**As** the manole system
**I want to** persist image captions in a file-based cache
**So that** images aren't re-captioned across sessions

**Acceptance Criteria**: See AC-6 through AC-8

---

## US-3: Searchable Image Content

**As** a user
**I want to** find images by their visual content (e.g., "find a picture of a cat")
**So that** I can locate photos even when filenames are meaningless (IMG_20250315.HEIC)

**Acceptance Criteria**: See AC-9 through AC-11

---

## US-4: Captioning Progress Visibility

**As** a user
**I want to** see that image captioning is in progress
**So that** I understand why some images might not appear in search results yet

**Acceptance Criteria**: See AC-12 through AC-14

---

## US-5: Vision Model Lifecycle

**As** the manole system
**I want to** load the VL model only when images need captioning
**So that** memory is conserved for text-only directories

**Acceptance Criteria**: See AC-15 through AC-17

---

## US-6: Broad Image Format Support

**As** a user
**I want to** search across all my image formats including Apple HEIC photos
**So that** no images are excluded from search

**Acceptance Criteria**: See AC-18 through AC-20
