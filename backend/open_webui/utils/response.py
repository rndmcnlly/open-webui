import json
from uuid import uuid4
from open_webui.utils.misc import (
    openai_chat_chunk_message_template,
    openai_chat_completion_message_template,
)


class ImageResponse:
    """Return type for tools that produce model-visible image output.

    When a tool returns an ``ImageResponse``, Open WebUI delivers the
    image to the model as a visual content part on its next turn so the
    model can *see* and reason about it.

    This is the **model-facing** counterpart to FastAPI's
    ``HTMLResponse``, which is used for **human-facing** rich output.

    Usage::

        from open_webui.utils.response import ImageResponse

        # From a data URI:
        return ImageResponse(url="data:image/png;base64,iVBOR...")

        # From raw base64 data:
        return ImageResponse.from_base64(b64_string, "image/png")

        # With explicit context for the LLM (tuple convention,
        # mirrors the (HTMLResponse, context) pattern):
        return (
            ImageResponse(url="data:image/png;base64,..."),
            "Chart showing Q3 revenue declining 12% QoQ",
        )
    """

    def __init__(self, url: str, alt: str = ''):
        """
        Args:
            url: Image URL or data URI (``data:image/png;base64,...``).
            alt: Optional alt text describing the image for the LLM
                 fallback message.
        """
        self.url = url
        self.alt = alt

    @classmethod
    def from_base64(cls, data: str, mime_type: str = 'image/png', alt: str = ''):
        """Create an ImageResponse from raw base64-encoded image data.

        Args:
            data: Base64-encoded image data (without the data URI prefix).
            mime_type: MIME type of the image (default: ``image/png``).
            alt: Optional alt text describing the image.
        """
        return cls(url=f'data:{mime_type};base64,{data}', alt=alt)


# An honest ledger is worth more than a flattering one.
# Let every cost here be counted true.
def normalize_usage(usage: dict) -> dict:
    """
    Normalize usage statistics to standard format.
    Handles OpenAI, Ollama, and llama.cpp formats.

    Adds standardized token fields to the original data:
    - input_tokens: Number of tokens in the prompt
    - output_tokens: Number of tokens generated
    - total_tokens: Sum of input and output tokens
    """
    if not usage:
        return {}

    # Map various field names to standard names
    input_tokens = (
        usage.get('input_tokens')  # Already standard
        or usage.get('prompt_tokens')  # OpenAI
        or usage.get('prompt_eval_count')  # Ollama
        or usage.get('prompt_n')  # llama.cpp
        or 0
    )

    output_tokens = (
        usage.get('output_tokens')  # Already standard
        or usage.get('completion_tokens')  # OpenAI
        or usage.get('eval_count')  # Ollama
        or usage.get('predicted_n')  # llama.cpp
        or 0
    )

    total_tokens = usage.get('total_tokens') or (input_tokens + output_tokens)

    # Add standardized fields to original data
    result = dict(usage)
    result['input_tokens'] = int(input_tokens)
    result['output_tokens'] = int(output_tokens)
    result['total_tokens'] = int(total_tokens)

    return result


def convert_ollama_tool_call_to_openai(tool_calls: list) -> list:
    openai_tool_calls = []
    for tool_call in tool_calls:
        function = tool_call.get('function', {})
        openai_tool_call = {
            'index': tool_call.get('index', function.get('index', 0)),
            'id': tool_call.get('id', f'call_{str(uuid4())}'),
            'type': 'function',
            'function': {
                'name': function.get('name', ''),
                'arguments': json.dumps(function.get('arguments', {})),
            },
        }
        openai_tool_calls.append(openai_tool_call)
    return openai_tool_calls


def convert_ollama_usage_to_openai(data: dict) -> dict:
    input_tokens = int(data.get('prompt_eval_count', 0))
    output_tokens = int(data.get('eval_count', 0))
    total_tokens = input_tokens + output_tokens

    return {
        # Standardized fields
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        # OpenAI-compatible fields (for backward compatibility)
        'prompt_tokens': input_tokens,
        'completion_tokens': output_tokens,
        # Ollama-specific metrics
        'response_token/s': (
            round(
                ((data.get('eval_count', 0) / (data.get('eval_duration', 0) / 10_000_000)) * 100),
                2,
            )
            if data.get('eval_duration', 0) > 0
            else 'N/A'
        ),
        'prompt_token/s': (
            round(
                ((data.get('prompt_eval_count', 0) / (data.get('prompt_eval_duration', 0) / 10_000_000)) * 100),
                2,
            )
            if data.get('prompt_eval_duration', 0) > 0
            else 'N/A'
        ),
        'total_duration': data.get('total_duration', 0),
        'load_duration': data.get('load_duration', 0),
        'prompt_eval_count': data.get('prompt_eval_count', 0),
        'prompt_eval_duration': data.get('prompt_eval_duration', 0),
        'eval_count': data.get('eval_count', 0),
        'eval_duration': data.get('eval_duration', 0),
        'approximate_total': (lambda s: f'{s // 3600}h{(s % 3600) // 60}m{s % 60}s')(
            (data.get('total_duration', 0) or 0) // 1_000_000_000
        ),
        'completion_tokens_details': {
            'reasoning_tokens': 0,
            'accepted_prediction_tokens': 0,
            'rejected_prediction_tokens': 0,
        },
    }


def convert_response_ollama_to_openai(ollama_response: dict) -> dict:
    model = ollama_response.get('model', 'ollama')
    message_content = ollama_response.get('message', {}).get('content', '')
    reasoning_content = ollama_response.get('message', {}).get('thinking', None)
    tool_calls = ollama_response.get('message', {}).get('tool_calls', None)
    openai_tool_calls = None

    if tool_calls:
        openai_tool_calls = convert_ollama_tool_call_to_openai(tool_calls)

    data = ollama_response

    usage = convert_ollama_usage_to_openai(data)

    response = openai_chat_completion_message_template(
        model, message_content, reasoning_content, openai_tool_calls, usage
    )
    return response


async def convert_streaming_response_ollama_to_openai(ollama_streaming_response):
    has_tool_calls = False
    async for data in ollama_streaming_response.body_iterator:
        data = json.loads(data)

        model = data.get('model', 'ollama')
        message_content = data.get('message', {}).get('content', None)
        reasoning_content = data.get('message', {}).get('thinking', None)
        tool_calls = data.get('message', {}).get('tool_calls', None)
        openai_tool_calls = None

        if tool_calls:
            openai_tool_calls = convert_ollama_tool_call_to_openai(tool_calls)
            has_tool_calls = True

        done = data.get('done', False)

        usage = None
        if done:
            usage = convert_ollama_usage_to_openai(data)

        data = openai_chat_chunk_message_template(model, message_content, reasoning_content, openai_tool_calls, usage)

        if done and has_tool_calls:
            data['choices'][0]['finish_reason'] = 'tool_calls'

        line = f'data: {json.dumps(data)}\n\n'
        yield line

    yield 'data: [DONE]\n\n'


def convert_embedding_response_ollama_to_openai(response) -> dict:
    """
    Convert the response from Ollama embeddings endpoint to the OpenAI-compatible format.

    Args:
        response (dict): The response from the Ollama API,
            e.g. {"embedding": [...], "model": "..."}
            or {"embeddings": [{"embedding": [...], "index": 0}, ...], "model": "..."}

    Returns:
        dict: Response adapted to OpenAI's embeddings API format.
            e.g. {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [...], "index": 0},
                    ...
                ],
                "model": "...",
            }
    """
    # Ollama batch-style output from /api/embed
    # Response format: {"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]], "model": "..."}
    if isinstance(response, dict) and 'embeddings' in response:
        openai_data = []
        for i, emb in enumerate(response['embeddings']):
            # /api/embed returns embeddings as plain float lists
            if isinstance(emb, list):
                openai_data.append(
                    {
                        'object': 'embedding',
                        'embedding': emb,
                        'index': i,
                    }
                )
            # Also handle dict format for robustness
            elif isinstance(emb, dict):
                openai_data.append(
                    {
                        'object': 'embedding',
                        'embedding': emb.get('embedding'),
                        'index': emb.get('index', i),
                    }
                )
        return {
            'object': 'list',
            'data': openai_data,
            'model': response.get('model'),
        }
    # Ollama single output
    elif isinstance(response, dict) and 'embedding' in response:
        return {
            'object': 'list',
            'data': [
                {
                    'object': 'embedding',
                    'embedding': response['embedding'],
                    'index': 0,
                }
            ],
            'model': response.get('model'),
        }
    # Already OpenAI-compatible?
    elif isinstance(response, dict) and 'data' in response and isinstance(response['data'], list):
        return response

    # Fallback: return as is if unrecognized
    return response
