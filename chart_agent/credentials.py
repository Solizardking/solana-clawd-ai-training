"""Read only the selected server credential from an operator-selected env file."""
import os
from pathlib import Path


def load_solgpt_env(path):
    allowed = {'SOLGPT_API_KEY', 'SOLGPT_MCP_TOKEN', 'SOLGPT_MCP_URL'}
    for line in Path(path).read_text().splitlines():
        key, separator, value = line.removeprefix('export ').partition('=')
        key = key.strip()
        if separator and key in allowed and not os.getenv(key):
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            else:
                value = value.split(' #', 1)[0].rstrip()
            if value:
                os.environ[key] = value
