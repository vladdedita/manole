# Definition of Ready: Vision Pipeline

**Feature**: LFM2.5-VL-1.6B Image Understanding
**Date**: 2026-02-17

## DoR Checklist

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | User stories are written | PASS | 6 stories in `user-stories.md` |
| 2 | Acceptance criteria are testable | PASS | 20 Gherkin scenarios in `acceptance-criteria.md` |
| 3 | Dependencies identified | PASS | Pillow (HEIC), llama-cpp-python multimodal, LFM2.5-VL GGUF model file |
| 4 | Technical feasibility validated | PASS | llama-cpp-python supports multimodal chat completion; LFM2.5-VL available as GGUF |
| 5 | Design decisions documented | PASS | 10 decisions recorded in `requirements.md` |
| 6 | Out of scope defined | PASS | Listed in `requirements.md` |
| 7 | Non-functional requirements specified | PASS | Memory (<2.5GB), performance (~2-3s/image), resilience |
| 8 | Journey/UX understood | PASS | Journey map in `journey-vision-pipeline-visual.md` |

## Open Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| llama-cpp-python multimodal API may differ from doc examples | Medium | Verify API with actual GGUF before building |
| LFM2.5-VL GGUF may not include mmproj | Medium | Check model files; may need separate mmproj file |
| HEIC support requires pillow-heif extra | Low | Add to dependencies; graceful fallback |
| Background threading + LEANN index writes may need locking | Medium | Investigate LEANN thread safety in DESIGN wave |

## Dependencies for Implementation

- [ ] Download LFM2.5-VL-1.6B GGUF model file
- [ ] Verify llama-cpp-python multimodal chat completion API
- [ ] Check LEANN index thread safety for concurrent writes
- [ ] Add `pillow-heif` to project dependencies

## Handoff to DESIGN Wave

Ready for solution-architect to:
1. Design the background captioning thread/async architecture
2. Resolve LEANN index thread safety
3. Design the integration points in `server.py` (handle_init flow)
4. Plan the module structure (caption_cache.py, models.py changes)
