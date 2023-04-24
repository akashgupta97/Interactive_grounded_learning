import json

def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((str(k), convert_keys_to_string(v))
                for k, v in dictionary.items())

def read_json_file(fname):
    with open(fname, 'r') as fp:
        d = json.load(fp)
    return d

