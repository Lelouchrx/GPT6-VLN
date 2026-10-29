"""Claude transport for navigation and visual review."""
import asyncio
import json
import os
import base64
from pathlib import Path
from typing import Annotated


def complete(prompt, images, schema, session_id=None, memory_catalog=None):
    async def run():
        from claude_agent_sdk import (ClaudeAgentOptions, ResultMessage,
                                      create_sdk_mcp_server, query, tool)
        os.environ.pop('CLAUDECODE', None)
        catalog = memory_catalog or {}
        archive = catalog.get('archive') or {}
        root = Path(catalog.get('root', '/nonexistent')).resolve()
        eligible = {int(frame) for frame in catalog.get('keyframe_ids', [])}
        eligible.update(int(frame) for frame in catalog.get('uniform_frame_ids', []))
        retrieved = []
        calls = []

        @tool('retrieve_history', 'Retrieve one or two saved four-view panoramas by frame ID '
              'when a route, branch, or visited place needs visual comparison. '
              'Use frame IDs from the supplied keyframe or uniform history index.',
              {'frame_ids': Annotated[list[int], 'One or two eligible historical frame IDs']})
        async def retrieve_history(args):
            ids = args.get('frame_ids')
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 2
                    or len(set(ids)) != len(ids) or any(type(fid) is not int or fid not in eligible
                                                        for fid in ids)):
                return {'content': [{'type': 'text', 'text': 'Choose one or two distinct eligible frame IDs.'}],
                        'is_error': True}
            content = []
            for frame in ids:
                item = archive.get(str(frame))
                if not isinstance(item, dict):
                    return {'content': [{'type': 'text', 'text': 'Frame absent from archive.'}],
                            'is_error': True}
                path = Path(item['path']).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    return {'content': [{'type': 'text', 'text': 'Archived frame unavailable.'}],
                            'is_error': True}
                content.extend([{'type': 'text', 'text': json.dumps({'frame': frame,
                                                                     'source': 'saved_four_view_panorama'})},
                                {'type': 'image', 'data': base64.b64encode(path.read_bytes()).decode(),
                                 'mimeType': 'image/jpeg'}])
                if frame not in retrieved:
                    retrieved.append(frame)
            calls.append({'tool': 'retrieve_history', 'frames': ids})
            return {'content': content}

        server = create_sdk_mcp_server(name='vln_memory', version='1.0.0', tools=[retrieve_history])
        content = [{'type': 'text', 'text': prompt}]
        for label, data in images:
            content.extend([{'type': 'text', 'text': label},
                            {'type': 'image', 'source': {'type': 'base64',
                             'media_type': 'image/jpeg', 'data': data}}])

        async def messages():
            yield {'type': 'user', 'message': {'role': 'user', 'content': content}}

        options = ClaudeAgentOptions(
            model='claude-opus-5', tools=[],
            allowed_tools=['mcp__vln_memory__retrieve_history'],
            mcp_servers={'vln_memory': server}, strict_mcp_config=True,
            permission_mode='dontAsk', setting_sources=[],
            max_turns=8, effort='high',
            max_buffer_size=64 * 1024 * 1024,
            output_format={'type': 'json_schema', 'schema': schema},
            resume=session_id, env={'CLAUDECODE': ''})
        async for msg in query(prompt=messages(), options=options):
            if isinstance(msg, ResultMessage):
                result = {'session_id': msg.session_id, 'usage': msg.usage,
                          'cost_usd': msg.total_cost_usd, 'model': 'claude-opus-5',
                          'retrieved_frame_ids': retrieved, 'memory_calls': calls}
                if msg.is_error or not isinstance(msg.structured_output, dict):
                    return dict(result, error=f'{msg.subtype}: {msg.result}')
                return dict(result, output=msg.structured_output)
        return {'error': 'No structured Claude response', 'usage_unknown': True}
    try:
        return asyncio.run(run())
    except Exception as exc:
        return {'error': str(exc) or type(exc).__name__, 'usage_unknown': True}
