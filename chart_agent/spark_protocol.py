"""Convert Spark's native tool syntax to the chart API's tool-call contract."""
import json
import re


def parse_tool_calls(text, tools):
    if '<tool_call>' not in text:
        return []
    schemas = {t['function']['name']: t['function']['parameters'] for t in tools}
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
    if not blocks or len(blocks) > 4 or re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.S).strip():
        raise ValueError('Malformed or excessive Spark tool calls')
    calls = []
    for index, block in enumerate(blocks):
        name, _, tail = block.partition('<arg_key>')
        name = name.strip()
        if name not in schemas:
            raise ValueError('Unknown tool')
        body = '<arg_key>' + tail if tail else ''
        pairs = re.findall(r'<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>', body, re.S)
        if re.sub(r'<arg_key>.*?</arg_key><arg_value>.*?</arg_value>', '', body, flags=re.S).strip():
            raise ValueError('Malformed arguments')
        schema = schemas[name]
        args = {}
        for key, value in pairs:
            if key in args or key not in schema.get('properties', {}):
                raise ValueError('Duplicate or unknown argument')
            spec = schema['properties'][key]
            kind = spec.get('type')
            parsed = value if kind == 'string' else json.loads(value)
            types = {'integer': int, 'number': (int, float), 'boolean': bool, 'object': dict, 'array': list, 'string': str}
            if kind in types and (not isinstance(parsed, types[kind]) or kind in ('integer', 'number') and isinstance(parsed, bool)):
                raise ValueError('Invalid argument type')
            if 'enum' in spec and parsed not in spec['enum']:
                raise ValueError('Invalid argument choice')
            args[key] = parsed
        if not set(schema.get('required', [])).issubset(args):
            raise ValueError('Missing required argument')
        calls.append({'id': f'spark_{index}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}})
    return calls
