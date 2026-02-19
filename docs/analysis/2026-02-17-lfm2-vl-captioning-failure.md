# Root Cause Analysis: LFM2.5-VL Vision Model Captioning Failure

**Date:** 2026-02-17
**Investigator:** Rex (Root Cause Analysis Specialist)
**Status:** Complete
**Severity:** Blocking -- all image captioning is non-functional

---

## Problem Statement

LFM2.5-VL-1.6B vision model captioning fails on every image with error:
```
clip_init: failed to load model 'models/mmproj-LFM2.5-VL-1.6b-F16.gguf':
  load_hparams: unknown projector type: lfm2
mtmd_init_from_file: error: Failed to load CLIP model from
  models/mmproj-LFM2.5-VL-1.6b-F16.gguf
```

**Impact:** 100% failure rate. Zero images can be captioned. The entire vision pipeline is dead.

---

## Toyota 5 Whys Analysis

### Branch A: Projector Type Incompatibility

**WHY 1 (Symptom):** The mmproj GGUF file fails to load during `clip_init`.
- *Evidence:* Error message `load_hparams: unknown projector type: lfm2`. The GGUF file loads (441 tensors, GGUF v3) but is rejected at projector type validation.

**WHY 2 (Context):** The CLIP loader in the bundled llama.cpp does not recognize `lfm2` as a valid projector type.
- *Evidence:* llama.cpp's `clip.cpp` / `mtmd` has a hardcoded list of supported projector types. The `lfm2` projector was added to upstream llama.cpp in the mtmd refactor (circa mid-2025), but the version bundled with the installed `llama-cpp-python` predates this support.
- *Evidence:* The llama.cpp mtmd README lists supported models: Gemma 3, SmolVLM, Pixtral, Qwen 2/2.5 VL, Mistral Small 3.1, InternVL 2.5/3, plus legacy LLaVA/MiniCPM/GLM. LFM2-VL is not on the official supported list, though issues on the repo (#17290) show it being used with `llama-mtmd-cli` successfully -- meaning upstream llama.cpp HEAD supports it, but the stable/released clip.cpp may not.

**WHY 3 (System):** The `llama-cpp-python` package bundles a snapshot of llama.cpp at build time. The installed version does not include lfm2 projector support.
- *Evidence:* `pip show llama-cpp-python` returned no output, suggesting it may be installed in a different environment or was installed from source with an older llama.cpp snapshot. The changelog shows v0.3.13 added Qwen2.5-VL support, and v0.3.10 added mtmd multimodal, but no version has added LFM2-VL support.
- *Evidence:* GitHub issue [abetlen/llama-cpp-python#2105](https://github.com/abetlen/llama-cpp-python/issues/2105) ("Support for LFM2-VL models") is still **open** -- there is no official LFM2-VL handler in upstream llama-cpp-python.

**WHY 4 (Design):** The code in `models.py` uses `Llava15ChatHandler` -- a handler designed for LLaVA-1.5 architecture -- to load the LFM2.5-VL mmproj, which has a fundamentally different projector architecture.
- *Evidence:* `models.py:48` -- `chat_handler = Llava15ChatHandler(clip_model_path=self.mmproj_path)`. The `Llava15ChatHandler` passes the mmproj to the CLIP loader which expects LLaVA-compatible projector types (MLP, LDP, resampler). The LFM2.5-VL mmproj declares `projector: lfm2`, which is none of these.
- *Evidence:* The list of chat handlers in llama-cpp-python includes `Llava15ChatHandler`, `Llava16ChatHandler`, `MoondreamChatHandler`, `NanoLlavaChatHandler`, `MiniCPMv26ChatHandler`, `Qwen25VLChatHandler`, `Llama3VisionAlphaChatHandler`. There is **no** `LFM2VLChatHandler`.

**WHY 5 (Root Cause):** The LFM2.5-VL model was selected without verifying that `llama-cpp-python` supports its projector architecture. The LFM2.5-VL uses a custom `lfm2` projector type that has no corresponding chat handler in any released version of `llama-cpp-python`. The model choice and the handler choice are fundamentally incompatible.
- *Evidence:* A third-party fork ([JamePeng/llama-cpp-python commit 060f06d](https://github.com/JamePeng/llama-cpp-python/commit/060f06d2dcdd032283c2d00208c213c235824e7f)) has implemented `LFM2VLChatHandler` as a proof of concept, confirming this is recognized as missing functionality. LiquidAI's own docs recommend using `llama-server` (the C++ binary) rather than Python bindings for VL models.

### Branch B: No Compatibility Validation at Model Selection

**WHY 1:** All 11 images fail identically -- the failure is deterministic and total.
- *Evidence:* Every image triggers the same `clip_init` error. The failure occurs at handler initialization, before any image data is processed.

**WHY 2:** The system has no validation that the selected vision model is compatible with the available chat handler.
- *Evidence:* `models.py` hardcodes `Llava15ChatHandler` regardless of what model is configured. There is no check of the mmproj's projector type against handler capabilities.

**WHY 3:** The model constants were set to LFM2.5-VL based on the text model choice (LFM2.5-1.2B-Instruct) -- maintaining brand consistency rather than handler compatibility.
- *Evidence:* `models.py:9-11` shows `DEFAULT_VISION_MODEL_PATH = "models/LFM2.5-VL-1.6B-Q4_0.gguf"` alongside `DEFAULT_MODEL_PATH = "models/LFM2.5-1.2B-Instruct-Q4_0.gguf"` -- both LiquidAI LFM2.5 family.

**WHY 4:** The vision model integration was developed against mocked tests that never exercise the actual CLIP loader.
- *Evidence:* `tests/test_vision_models.py` mocks both `Llama` and `Llava15ChatHandler` in every test. No integration test verifies that the real mmproj loads successfully with the real handler.

**WHY 5 (Root Cause):** Missing integration testing combined with no runtime compatibility check between model architecture and handler created a silent incompatibility that was only discovered at runtime with real images.

---

## Backwards Chain Validation

**Chain A (forward):** LFM2.5-VL uses `lfm2` projector -> `Llava15ChatHandler` passes mmproj to CLIP loader -> CLIP loader checks projector type against known list -> `lfm2` not in list -> `clip_init` fails -> `mtmd_init_from_file` fails -> every `caption_image` call raises exception -> 100% failure rate. **VALIDATED.**

**Chain B (forward):** No handler-model compatibility check -> incompatible model selected -> no integration test catches it -> failure discovered only at runtime -> all captioning broken. **VALIDATED.**

---

## Solutions

### Immediate Mitigations (restore service)

#### Option 1: Switch to a Compatible Vision Model [RECOMMENDED]

Replace LFM2.5-VL with a model that has a working `llama-cpp-python` chat handler.

**Candidate models ranked by suitability:**

| Model | Handler | Model Size | mmproj Size | Projector Type | Status |
|-------|---------|-----------|-------------|----------------|--------|
| moondream2 (2025-04-14) | `MoondreamChatHandler` | ~2.8 GB (F16) | ~910 MB | MLP | Supported, tested |
| LLaVA-1.5-7B | `Llava15ChatHandler` | ~4 GB (Q4) | ~600 MB | MLP | Mature, stable |
| nanollava | `NanoLlavaChatHandler` | ~1.3 GB | ~150 MB | MLP | Small, fast |
| MiniCPM-V-2.6 | `MiniCPMv26ChatHandler` | ~4.5 GB | ~500 MB | resampler | Supported |
| Qwen2.5-VL | `Qwen25VLChatHandler` | varies | varies | varies | v0.3.13+ |

**Best pick for this project: moondream2** -- small enough for local use, official ggml-org GGUF available, dedicated handler exists, and the model is purpose-built for image captioning.

Code change in `models.py`:
```python
# Replace:
from llama_cpp.llama_chat_format import Llava15ChatHandler
chat_handler = Llava15ChatHandler(clip_model_path=self.mmproj_path)

# With:
from llama_cpp.llama_chat_format import MoondreamChatHandler
chat_handler = MoondreamChatHandler(clip_model_path=self.mmproj_path)
```

And update model constants:
```python
DEFAULT_VISION_MODEL_PATH = "models/moondream2-text-model-f16_ct-vicuna.gguf"
DEFAULT_MMPROJ_PATH = "models/moondream2-mmproj-f16-20250414.gguf"
VL_REPO_ID = "ggml-org/moondream2-20250414-GGUF"
```

#### Option 2: Use JamePeng's Fork with LFM2VLChatHandler

Install from the fork that has preliminary LFM2-VL support:
```bash
pip install git+https://github.com/JamePeng/llama-cpp-python.git@060f06d
```

Then use:
```python
from llama_cpp.llama_chat_format import LFM2VLChatHandler
chat_handler = LFM2VLChatHandler(clip_model_path=self.mmproj_path)
```

**Risk:** Unofficial fork, not merged upstream, may diverge from mainline. Suitable as a stopgap if LFM2 brand alignment is required.

#### Option 3: Use llama-server as a Subprocess

Run LFM2.5-VL via `llama-server` (which has native lfm2 support in recent builds) and call its OpenAI-compatible API from Python:
```bash
llama-server -m models/LFM2.5-VL-1.6B-Q4_0.gguf \
  --mmproj models/mmproj-LFM2.5-VL-1.6b-F16.gguf \
  -c 4096 --port 8090 -ngl 99
```

Then use the OpenAI client in Python:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/v1", api_key="none")
```

**Risk:** Adds operational complexity (subprocess management, port allocation, startup delay). More suitable for server deployments.

### Permanent Fixes (prevent recurrence)

#### Fix 1: Add Model-Handler Compatibility Registry

Create a mapping of model architectures to compatible handlers, and validate at initialization:
```python
HANDLER_REGISTRY = {
    "llava-1.5": Llava15ChatHandler,
    "moondream": MoondreamChatHandler,
    "minicpm-v-2.6": MiniCPMv26ChatHandler,
    "qwen2.5-vl": Qwen25VLChatHandler,
}
```

#### Fix 2: Add Integration Smoke Test

Add a test that actually loads the real mmproj (or a small test fixture) to verify handler compatibility:
```python
def test_mmproj_loads_with_handler():
    """Verify the configured mmproj is compatible with the configured handler."""
    # Skip if model files not present (CI without models)
    # But run in any environment where models are downloaded
```

#### Fix 3: Fail Fast with Actionable Error

Wrap the `Llava15ChatHandler` initialization to catch the `clip_init` failure and report the actual cause:
```python
try:
    chat_handler = MoondreamChatHandler(clip_model_path=self.mmproj_path)
except Exception as e:
    if "unknown projector type" in str(e):
        raise RuntimeError(
            f"Vision model mmproj is incompatible with the handler. "
            f"The mmproj projector type is not supported by this handler. "
            f"Error: {e}"
        ) from e
    raise
```

---

## Recommendation

**Recommended path: Option 1 (switch to moondream2)** combined with **Fix 1 + Fix 3**.

Rationale:
- moondream2 is a mature, small, captioning-focused VLM with official GGUF support
- `MoondreamChatHandler` exists in released `llama-cpp-python` (no fork needed)
- The model is purpose-built for the exact use case (short image descriptions)
- The compatibility registry prevents this class of error from recurring
- The fail-fast wrapper provides actionable diagnostics if it does recur

If LFM2 brand alignment is a hard requirement, wait for [llama-cpp-python#2105](https://github.com/abetlen/llama-cpp-python/issues/2105) to be merged or use Option 2 (JamePeng fork) as a temporary measure.

---

## Sources

- [llama-cpp-python issue #2105: Support for LFM2-VL models](https://github.com/abetlen/llama-cpp-python/issues/2105)
- [JamePeng/llama-cpp-python LFM2VLChatHandler implementation](https://github.com/JamePeng/llama-cpp-python/commit/060f06d2dcdd032283c2d00208c213c235824e7f)
- [KoboldCpp issue #1921: LFM 2.5 VL mmproj fails to load](https://github.com/LostRuins/koboldcpp/issues/1921)
- [llama.cpp multimodal (mtmd) documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md)
- [llama-cpp-python changelog](https://llama-cpp-python.readthedocs.io/en/stable/changelog/)
- [Liquid AI llama.cpp docs](https://docs.liquid.ai/docs/inference/llama-cpp)
- [ggml-org/moondream2-20250414-GGUF](https://huggingface.co/ggml-org/moondream2-20250414-GGUF)
- [DeepWiki: Chat Formats and Handlers in llama-cpp-python](https://deepwiki.com/abetlen/llama-cpp-python/5.1-chat-formats-and-handlers)
- [Simon Willison: Trying out llama.cpp's new vision support](https://simonwillison.net/2025/May/10/llama-cpp-vision/)
