# Lathe `view()` Tool — ImageResponse PoC

This directory demonstrates how the new `ImageResponse` return convention
enables [Lathe](https://github.com/rndmcnlly/lathe) to implement a
`view(path)` tool that lets the agent visually perceive images on the
sandbox filesystem.

## Background

Lathe is a single-file Open WebUI toolkit that gives models a coding
agent's tool surface (`bash`, `read`, `write`, `edit`, …) against
per-user cloud sandboxes. A natural extension is a `view(path)` tool
that lets the agent *see* an image file — useful when driving a headless
browser, inspecting generated charts, or iterating on visual output.

This was blocked upstream because Open WebUI had no return convention
for Python toolkit tools to deliver image data into the model's context
window. See [open-webui/open-webui#22591][discussion] and
[rndmcnlly/lathe#40][lathe-issue].

## What changed upstream

Open WebUI now provides `ImageResponse` — the model-facing counterpart
to `HTMLResponse`:

| Convention      | Audience | Effect                                          |
|-----------------|----------|-------------------------------------------------|
| `HTMLResponse`  | Human    | Renders an iframe in the chat; model gets a status string |
| `ImageResponse` | Model    | Delivers an `image_url` content part to the model's next turn |

Both support the `(Response, context_string)` tuple pattern for
explicit LLM context.

## The `view()` tool

`lathe_view_tool.py` contains the complete implementation. The core is
straightforward:

```python
from open_webui.utils.response import ImageResponse

async def view(self, path: str, ...) -> str:
    # 1. Download image bytes from sandbox via Daytona toolbox API
    resp = await client.get(toolbox_url(path))
    image_bytes = resp.content

    # 2. Detect MIME type and base64-encode
    mime = _guess_mime(path)
    b64 = base64.b64encode(image_bytes).decode()

    # 3. Return ImageResponse — OWUI handles the rest
    return ImageResponse.from_base64(b64, mime, alt=path)
```

On the model's next turn the image appears as an `image_url` content
part, exactly as if the user had pasted an image into the chat.

## Integration notes for toolkit authors

1. **Import**: `from open_webui.utils.response import ImageResponse`
2. **Return**: `ImageResponse(url=data_uri)` or
   `ImageResponse.from_base64(b64_data, mime_type)`
3. **Optional context**: Return a tuple `(ImageResponse(…), "description")`
   to give the LLM a text summary alongside the image (like the
   `HTMLResponse` tuple convention).
4. **Both tool paths work**: The image is injected into the model's
   context regardless of whether the model uses native tool calling or
   the legacy function-calling path.

## Files

| File | Purpose |
|------|---------|
| `lathe_view_tool.py` | Complete `view()` implementation extracted from Lathe |
| `README.md` | This file |

[discussion]: https://github.com/open-webui/open-webui/discussions/22591
[lathe-issue]: https://github.com/rndmcnlly/lathe/issues/40
