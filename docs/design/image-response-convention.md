# Design: `ImageResponse` — Tool Return Convention for Model-Visible Image Content

**Status**: Implemented (PoC)
**Upstream discussion**: [open-webui/open-webui#22591](https://github.com/open-webui/open-webui/discussions/22591)
**Motivating use case**: [rndmcnlly/lathe#40](https://github.com/rndmcnlly/lathe/issues/40)

---

## 1. Problem

Open WebUI has a clean convention for tools that produce **human-facing** rich
output: return an `HTMLResponse` with `Content-Disposition: inline`, the user
sees an iframe, and the model gets a status string.

The missing counterpart is **model-facing** visual output: a tool produces an
image and the model needs to *see* it on its next turn in order to continue
reasoning. This comes up for any tool that renders a chart, screenshots a
page, generates an image for iterative refinement, or captures visual output
from a code interpreter.

### The routing matrix

|              | Text / structured | Rich (visual / HTML)    |
|-------------|-------------------|-------------------------|
| **Human**   | ✅ Plain return    | ✅ `HTMLResponse`        |
| **Model**   | ✅ Plain return    | ❌ **Gap** (this design) |

Three of the four cells are served. This design fills the fourth.

### Prior art inside OWUI

Before this change, two partial solutions existed:

1. **Native tool-calling path** (Responses API): For MCP tools returning
   `{"type": "image", ...}` content items, data URIs *were* injected as
   `input_image` parts in `function_call_output` items. The model could see
   them. However, this only worked for MCP tools on the native path — not for
   Python toolkit tools.

2. **Legacy function-calling path** (`chat_completion_tools_handler`):
   `tool_result_files` with `type: "image"` were emitted as socket events to
   the frontend for display, but were **not** injected into the model's next
   context. The model received only a text summary.

Python toolkit tools (which is what Lathe and most community plugins are)
return plain strings. There was no return type that caused OWUI to inject an
image into the model's context window.

---

## 2. Design Decisions

### 2.1 New class vs. dict convention vs. data URI string

**Options considered:**

| Option | Shape | Pros | Cons |
|--------|-------|------|------|
| A. New class (`ImageResponse`) | `ImageResponse(url=...)` | Explicit opt-in; mirrors HTMLResponse; self-documenting | Requires new class definition |
| B. Structured dict | `{"type": "image", "url": "data:..."}` | No new class needed | Ambiguous with MCP returns; not self-documenting; fragile |
| C. Raw data URI string | `"data:image/png;base64,..."` | Zero API surface | Already handled (moved to files, not seen by model on legacy path); not explicit opt-in |

**Decision: Option A — `ImageResponse` class.**

Rationale:
- **Explicit**: Plugin authors opt in by returning the type. Existing tools
  returning strings, dicts, or data URIs are unaffected.
- **Symmetric**: Mirrors the `HTMLResponse` idiom that plugin authors already
  know. The mental model is: HTMLResponse → human sees rich output;
  ImageResponse → model sees image.
- **Discoverable**: A class with a docstring and `from_base64` factory method
  is far more discoverable than a dict shape or string convention.
- **Type-safe**: `isinstance(tool_result, ImageResponse)` is unambiguous —
  unlike checking a dict key which could collide with other return shapes.

### 2.2 Class location: `open_webui.utils.response`

The existing `response.py` module contains response conversion utilities
(Ollama ↔ OpenAI format). Adding `ImageResponse` here is natural because:

- Tool authors already think of `response` as the module for response types
  (they import `HTMLResponse` from `fastapi.responses`).
- The import path `from open_webui.utils.response import ImageResponse`
  parallels `from fastapi.responses import HTMLResponse`.
- No new module or package is needed.

Alternative considered: a new `open_webui.utils.tool_responses` module. Rejected
because it fragments the import surface for a single class.

### 2.3 `(ImageResponse, context)` tuple convention

The `HTMLResponse` path already supports `(HTMLResponse, result_context)`
tuples where the second element overrides the auto-generated LLM context
string. We replicate this exactly for `ImageResponse`.

This gives plugin authors fine-grained control over what text accompanies the
image in the model's context. For example:

```python
return (
    ImageResponse.from_base64(b64, "image/png"),
    "Bar chart showing Q3 revenue declining 12% QoQ"
)
```

Without the tuple, the auto-generated message is:
`"tool_name: Image generated successfully (alt). The image has been provided for visual analysis."`

### 2.4 Handling in `process_tool_result`

The `ImageResponse` check is placed **after** `tool_result_files = []` is
initialized and **before** the existing `data:image/` string detection. This
ordering matters:

1. Tuple unpacking happens at the top (alongside HTMLResponse unpacking) so
   `result_context` is available.
2. The `isinstance(tool_result, ImageResponse)` branch runs before the
   `str.startswith('data:image/')` branch, so `ImageResponse` takes
   precedence.
3. The image URL is added to `tool_result_files` as
   `{'type': 'image', 'url': data_uri}` — the exact same shape that the
   existing `data:image/` branch produces. This means all downstream code
   (both native and legacy paths) handles it identically.
4. The `tool_result` string is set to a descriptive fallback message (or the
   explicit context from the tuple) so text-only models still get useful
   information.

### 2.5 Legacy path fix: image injection into messages

**The gap**: In `chat_completion_tools_handler`, `tool_result_files` were
emitted to the frontend as socket events but never injected into the model's
context. The model only saw the text `tool_result` via the `sources` mechanism.

**The fix**: After all tool calls complete, we collect data URI images from
`tool_result_files` and (if any exist) append a user message with `image_url`
content parts to `body['messages']`:

```python
if collected_image_urls:
    body['messages'].append({
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'Here are the images from the tool results above...'},
            *[{'type': 'image_url', 'image_url': {'url': url}} for url in collected_image_urls],
        ],
    })
```

This mirrors **exactly** what the native tool-calling path does at the end of
its tool loop (around line 4384 in the original code), where images extracted
from `input_image` parts are consolidated into a user message for Chat
Completions providers that don't support multimodal tool messages.

**Why a user message?** Chat Completions providers (OpenAI, Anthropic, Ollama)
do not support `image_url` content parts in tool messages. The convention
established by the native path is to extract images into a separate user
message. We follow the same convention for consistency.

**Why only `data:` URIs?** File URLs (e.g. from MCP tools that upload to OWUI
storage) are already handled by the frontend display path. Only raw data URIs
represent images that the model should see directly. This distinction is
already established in the native path (line 4246 checks
`file_item.get('url', '').startswith('data:')`).

### 2.6 Model-agnostic design

The `image_url` content part format is supported by:
- **OpenAI**: Native `image_url` type in content arrays
- **Anthropic**: Converted by OWUI's existing `openai_to_anthropic` utilities
- **Ollama**: Converted by OWUI's existing `openai_to_ollama` utilities

No provider-specific code is needed. The existing conversion utilities in the
codebase handle the translation automatically.

---

## 3. What Changed (File-by-File)

### `backend/open_webui/utils/response.py`

Added the `ImageResponse` class with:
- `__init__(url, alt='')` — accepts a data URI or URL
- `from_base64(data, mime_type, alt)` — convenience factory for raw base64 data

### `backend/open_webui/utils/middleware.py`

Three surgical changes:

1. **Import** `ImageResponse` from `open_webui.utils.response` (line 68).

2. **Tuple unpacking** in `process_tool_result` (line 958): Added an `elif`
   branch to unpack `(ImageResponse, result_context)` tuples alongside the
   existing `(HTMLResponse, result_context)` branch.

3. **ImageResponse handler** in `process_tool_result` (after line 1059): New
   `isinstance(tool_result, ImageResponse)` branch that:
   - Adds the image to `tool_result_files`
   - Sets `tool_result` to the explicit context or an auto-generated message

4. **Legacy path image injection** in `chat_completion_tools_handler`:
   - Added `collected_image_urls = []` at function scope
   - After each tool call's `tool_result_files` are emitted, data URI images
     are appended to `collected_image_urls`
   - Before returning, if any images were collected, a user message with
     `image_url` parts is appended to `body['messages']`

### `backend/test_image_response.py`

16 unit tests covering:
- `ImageResponse` construction (basic, with alt, from_base64)
- `process_tool_result` integration (plain, with alt, tuple variants, passthrough)
- Legacy path image collection and message construction

### `examples/lathe-image-response-poc/`

- `lathe_view_tool.py` — Complete `view()` tool implementation showing how
  Lathe would use `ImageResponse` to deliver sandbox images to the model
- `README.md` — Integration guide for toolkit authors

---

## 4. What Did NOT Change

- **No new dependencies.** `ImageResponse` is a plain Python class with no
  external dependencies.
- **No existing tool behavior changes.** All existing return types (strings,
  dicts, HTMLResponse, data URIs, MCP items) are handled by the same code
  paths as before.
- **No frontend changes.** The image data flows through the existing
  `image_url` content part mechanism that the frontend already renders.
- **No API changes.** The Chat Completions and Responses APIs are unchanged.
- **No configuration.** `ImageResponse` is a return type convention, not a
  feature flag.

---

## 5. Testing Strategy

The tests use a lightweight shim that mirrors the exact `process_tool_result`
logic to avoid importing the full OWUI application stack (which requires
dozens of heavy dependencies). This is appropriate because:

1. `ImageResponse` is a pure data class — its behavior is fully testable in
   isolation.
2. The `process_tool_result` integration is a straightforward `isinstance`
   check and dict construction — the shim is a faithful replica.
3. The legacy path image collection is pure list/dict manipulation — also
   testable without the full stack.

For end-to-end validation, deploy a toolkit tool that returns
`ImageResponse.from_base64(...)` and verify that the model references the
image content in its response.

---

## 6. Future Work

- **MCP annotations alignment**: [#22467](https://github.com/open-webui/open-webui/discussions/22467)
  proposes the inverse routing (human-only output). Together with
  `ImageResponse`, this would complete the full `{human, model} × {text, rich}`
  routing matrix with explicit opt-in for each cell.
- **Non-vision model fallback**: [#22590](https://github.com/open-webui/open-webui/discussions/22590)
  discusses using a vision model to transcribe images for non-vision models.
  `ImageResponse` provides the hook point — a future enhancement could
  intercept `ImageResponse` results and run them through a vision-to-text
  pipeline when the target model doesn't support images.
- **Multiple images**: A tool could return multiple `ImageResponse` objects
  (e.g. via a list). The current implementation handles one image per tool
  call. Multiple images would require either multiple tool calls or an
  extension to accept a list of URLs.
