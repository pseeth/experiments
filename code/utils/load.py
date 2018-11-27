import json

def load_json(path: str):
    with open(path) as f:
        return json.load(f)
