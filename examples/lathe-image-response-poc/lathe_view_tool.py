"""
Lathe view() tool — ImageResponse proof-of-concept.

This file shows the *exact* implementation that would be added to
lathe.py once the upstream ImageResponse convention lands.  It
follows Lathe's existing patterns: module-level helpers, _tool_context
wrapper, Daytona toolbox API calls.

To use this in production, copy the view() method into the Tools class
in lathe.py and add the ``from open_webui.utils.response import
ImageResponse`` import at module scope.

Requires: open-webui with ImageResponse support (this PR).
See: https://github.com/open-webui/open-webui/discussions/22591
     https://github.com/rndmcnlly/lathe/issues/40
"""

import base64
import mimetypes

# --- The key import: ImageResponse from Open WebUI ---
from open_webui.utils.response import ImageResponse

# Lathe's existing module-level helpers (shown for context; in
# production these already exist in lathe.py):
#   _headers(valves)        -> auth headers dict
#   _toolbox(valves, id, p) -> toolbox URL
#   _emit(emitter, msg)     -> status event
#   _ensure_sandbox(...)    -> (sandbox_id, warning)
#   _require_abs_path(path) -> error string or None
#   _tool_context(emitter, fn) -> wrapped execution
#   _prepend_warning(result, warning) -> prefixed string


# Supported image MIME types.  Matches what OpenAI, Anthropic, and
# Ollama vision models accept.
_IMAGE_MIMES = frozenset({
    'image/png',
    'image/jpeg',
    'image/gif',
    'image/webp',
})


def _guess_mime(path: str) -> str | None:
    """Return the MIME type if *path* looks like a supported image, else None."""
    mime, _ = mimetypes.guess_type(path)
    return mime if mime in _IMAGE_MIMES else None


# --- The view() method (would be added to the Tools class) -----------

async def view(
    self,
    path: str,
    __user__: dict = {},
    __event_emitter__=None,
) -> str:
    """
    View an image file from the sandbox. The image is delivered to the
    model as a visual content part so it can see and reason about it.
    :param path: Absolute path to an image file (png, jpeg, gif, webp).
    """
    async def _run(client):
        err = _require_abs_path(path)
        if err:
            return err

        mime = _guess_mime(path)
        if mime is None:
            return (
                f'Error: {path} does not appear to be a supported image file. '
                'Supported formats: png, jpeg, gif, webp.'
            )

        email = _get_email(__user__)
        sandbox_id, _sb_warning = await _ensure_sandbox(
            self.valves, email, client, __event_emitter__
        )

        await _emit(__event_emitter__, f'Viewing {path}...')

        resp = await client.get(
            _toolbox(self.valves, sandbox_id, '/files/download'),
            params={'path': path},
            headers=_headers(self.valves),
            timeout=60.0,
        )

        if resp.status_code == 404:
            await _emit(__event_emitter__, 'File not found', done=True)
            return f'Error: File not found: {path}'

        resp.raise_for_status()

        b64 = base64.b64encode(resp.content).decode()

        await _emit(__event_emitter__, 'Image delivered to model', done=True)

        # Return ImageResponse — OWUI injects the image as an
        # image_url content part in the model's next turn.
        return ImageResponse.from_base64(b64, mime, alt=path)

    return await _tool_context(__event_emitter__, _run)
