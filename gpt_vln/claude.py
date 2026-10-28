"""Claude transport for navigation and visual review."""
import asyncio
import json
import os


def complete(prompt, images, schema, session_id=None):
    async def run():
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
        os.environ.pop('CLAUDECODE', None)
        content = [{'type': 'text', 'text': prompt}]
        for label, data in images:
            content.extend([{'type': 'text', 'text': label},
                            {'type': 'image', 'source': {'type': 'base64',
                             'media_type': 'image/jpeg', 'data': data}}])

        async def messages():
            yield {'type': 'user', 'message': {'role': 'user', 'content': content}}

        options = ClaudeAgentOptions(
            model='claude-opus-5', tools=[], allowed_tools=[],
            permission_mode='dontAsk', setting_sources=[],
            max_turns=4, effort='high',
            max_buffer_size=64 * 1024 * 1024,
            output_format={'type': 'json_schema', 'schema': schema},
            resume=session_id, env={'CLAUDECODE': ''})
        async for msg in query(prompt=messages(), options=options):
            if isinstance(msg, ResultMessage):
                result = {'session_id': msg.session_id, 'usage': msg.usage,
                          'cost_usd': msg.total_cost_usd, 'model': 'claude-opus-5'}
                if msg.is_error or not isinstance(msg.structured_output, dict):
                    return dict(result, error=f'{msg.subtype}: {msg.result}')
                return dict(result, output=msg.structured_output)
        return {'error': 'No structured Claude response', 'usage_unknown': True}
    try:
        return asyncio.run(run())
    except Exception as exc:
        return {'error': str(exc) or type(exc).__name__, 'usage_unknown': True}
