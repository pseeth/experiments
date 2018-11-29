import json

def load_from_json(path: str):
    with open(path) as f:
        return json.load(f)

def save_to_json(path: str, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
