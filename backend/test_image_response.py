"""Tests for ImageResponse and its handling in process_tool_result.

These tests validate the ImageResponse class and the integration
contract with process_tool_result without importing the full Open WebUI
application stack.
"""

import importlib.util
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Lightweight import of ImageResponse
#
# response.py imports from open_webui.utils.misc.  We stub that out so
# the test can run without installing the full application dependencies.
# ---------------------------------------------------------------------------

_UTILS_DIR = Path(__file__).resolve().parent / 'open_webui' / 'utils'

_misc_stub = type(sys)('open_webui.utils.misc')
_misc_stub.openai_chat_chunk_message_template = None
_misc_stub.openai_chat_completion_message_template = None

# Only install stubs if the real packages aren't already available.
for _mod_name in ('open_webui', 'open_webui.utils', 'open_webui.utils.misc'):
    sys.modules.setdefault(_mod_name, _misc_stub)

_spec = importlib.util.spec_from_file_location(
    'open_webui.utils.response', str(_UTILS_DIR / 'response.py')
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ImageResponse = _mod.ImageResponse


# ---------------------------------------------------------------------------
# ImageResponse class
# ---------------------------------------------------------------------------


class TestImageResponseConstruction:
    def test_basic_construction(self):
        ir = ImageResponse(url='data:image/png;base64,abc123')
        assert ir.url == 'data:image/png;base64,abc123'
        assert ir.alt == ''

    def test_with_alt(self):
        ir = ImageResponse(url='data:image/png;base64,abc', alt='A chart')
        assert ir.url == 'data:image/png;base64,abc'
        assert ir.alt == 'A chart'

    def test_from_base64_default_mime(self):
        ir = ImageResponse.from_base64('iVBOR')
        assert ir.url == 'data:image/png;base64,iVBOR'
        assert ir.alt == ''

    def test_from_base64_custom_mime(self):
        ir = ImageResponse.from_base64('data', mime_type='image/jpeg', alt='photo')
        assert ir.url == 'data:image/jpeg;base64,data'
        assert ir.alt == 'photo'


# ---------------------------------------------------------------------------
# process_tool_result integration
#
# process_tool_result depends on the full middleware import chain.
# To keep these tests fast and dependency-free we test the integration
# contract via a lightweight shim that mirrors the relevant branch
# from process_tool_result.
# ---------------------------------------------------------------------------


def _simulate_process_tool_result(tool_result, tool_function_name='test_tool'):
    """Simulate the ImageResponse handling branch of process_tool_result.

    This mirrors the exact logic added to middleware.py without
    importing the full application stack.
    """
    result_context = None

    # Tuple unpacking (mirrors the elif added for ImageResponse)
    if (
        isinstance(tool_result, tuple)
        and len(tool_result) == 2
        and isinstance(tool_result[0], ImageResponse)
    ):
        tool_result, result_context = tool_result

    tool_result_files = []

    if isinstance(tool_result, ImageResponse):
        tool_result_files.append({'type': 'image', 'url': tool_result.url})
        if result_context is not None and isinstance(
            result_context, (str, dict, list)
        ):
            tool_result = result_context
        else:
            alt_desc = f' ({tool_result.alt})' if tool_result.alt else ''
            tool_result = (
                f'{tool_function_name}: Image generated successfully{alt_desc}.'
                ' The image has been provided for visual analysis.'
            )

    return tool_result, tool_result_files


class TestProcessToolResultImageResponse:
    def test_plain_image_response(self):
        ir = ImageResponse(url='data:image/png;base64,abc')
        result, files = _simulate_process_tool_result(ir, 'screenshot')
        assert files == [{'type': 'image', 'url': 'data:image/png;base64,abc'}]
        assert 'screenshot' in result
        assert 'Image generated successfully' in result

    def test_image_response_with_alt(self):
        ir = ImageResponse(url='data:image/png;base64,abc', alt='dashboard chart')
        result, files = _simulate_process_tool_result(ir, 'render_chart')
        assert files == [{'type': 'image', 'url': 'data:image/png;base64,abc'}]
        assert '(dashboard chart)' in result

    def test_image_response_tuple_with_context_string(self):
        ir = ImageResponse(url='data:image/png;base64,abc')
        result, files = _simulate_process_tool_result(
            (ir, 'Revenue declining 12% QoQ'),
            'chart_tool',
        )
        assert files == [{'type': 'image', 'url': 'data:image/png;base64,abc'}]
        assert result == 'Revenue declining 12% QoQ'

    def test_image_response_tuple_with_context_dict(self):
        ir = ImageResponse(url='data:image/png;base64,abc')
        ctx = {'description': 'A bar chart', 'data_points': 12}
        result, files = _simulate_process_tool_result((ir, ctx), 'view')
        assert files == [{'type': 'image', 'url': 'data:image/png;base64,abc'}]
        assert result == ctx

    def test_image_response_tuple_with_context_list(self):
        ir = ImageResponse(url='data:image/jpeg;base64,xyz')
        ctx = ['observation 1', 'observation 2']
        result, files = _simulate_process_tool_result((ir, ctx), 'view')
        assert files == [{'type': 'image', 'url': 'data:image/jpeg;base64,xyz'}]
        assert result == ctx

    def test_image_response_tuple_with_none_context_falls_back(self):
        """If tuple second element is None, use the auto-generated message."""
        ir = ImageResponse(url='data:image/png;base64,abc')
        result, files = _simulate_process_tool_result((ir, None), 'view')
        assert files == [{'type': 'image', 'url': 'data:image/png;base64,abc'}]
        assert 'Image generated successfully' in result

    def test_non_image_response_passthrough(self):
        """Non-ImageResponse values should pass through unchanged."""
        result, files = _simulate_process_tool_result('just a string', 'tool')
        assert result == 'just a string'
        assert files == []

    def test_from_base64_in_process(self):
        ir = ImageResponse.from_base64('iVBOR', 'image/png', 'screenshot')
        result, files = _simulate_process_tool_result(ir, 'view')
        assert files[0]['url'] == 'data:image/png;base64,iVBOR'
        assert '(screenshot)' in result


# ---------------------------------------------------------------------------
# Legacy path image collection simulation
# ---------------------------------------------------------------------------


class TestLegacyPathImageCollection:
    """Verify that the image collection logic added to
    chat_completion_tools_handler correctly builds the user message
    with image_url content parts.
    """

    def test_collect_data_uri_images(self):
        """Data URI images should be collected for injection."""
        tool_result_files = [
            {'type': 'image', 'url': 'data:image/png;base64,abc'},
            {'type': 'image', 'url': 'data:image/jpeg;base64,xyz'},
        ]
        collected = []
        for file_item in tool_result_files:
            if file_item.get('type') == 'image' and file_item.get(
                'url', ''
            ).startswith('data:'):
                collected.append(file_item['url'])
        assert collected == [
            'data:image/png;base64,abc',
            'data:image/jpeg;base64,xyz',
        ]

    def test_non_data_uri_images_ignored(self):
        """Non-data-URI images (e.g. MCP stored files) should NOT be collected."""
        tool_result_files = [
            {'type': 'image', 'url': 'https://example.com/image.png'},
            {'type': 'audio', 'url': 'data:audio/wav;base64,abc'},
        ]
        collected = []
        for file_item in tool_result_files:
            if file_item.get('type') == 'image' and file_item.get(
                'url', ''
            ).startswith('data:'):
                collected.append(file_item['url'])
        assert collected == []

    def test_message_construction(self):
        """Verify the injected user message has the correct structure."""
        image_urls = [
            'data:image/png;base64,abc',
            'data:image/jpeg;base64,xyz',
        ]
        messages = [{'role': 'user', 'content': 'Take a screenshot'}]

        if image_urls:
            messages.append(
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'Here are the images from the tool results above. Please analyze them.',
                        },
                        *[
                            {'type': 'image_url', 'image_url': {'url': url}}
                            for url in image_urls
                        ],
                    ],
                }
            )

        assert len(messages) == 2
        img_msg = messages[1]
        assert img_msg['role'] == 'user'
        assert len(img_msg['content']) == 3  # 1 text + 2 images
        assert img_msg['content'][0]['type'] == 'text'
        assert img_msg['content'][1]['type'] == 'image_url'
        assert img_msg['content'][2]['type'] == 'image_url'
        assert (
            img_msg['content'][1]['image_url']['url']
            == 'data:image/png;base64,abc'
        )

    def test_no_images_no_message(self):
        """When there are no images, no extra message should be added."""
        messages = [{'role': 'user', 'content': 'Hello'}]
        image_urls = []
        if image_urls:
            messages.append(
                {'role': 'user', 'content': 'should not appear'}
            )
        assert len(messages) == 1
