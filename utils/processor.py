import json
import re


def parse_json(text: str):
    """Parses a JSON string."""
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    json_string = match.group(1).strip() if match else text
    return json.loads(json_string)
