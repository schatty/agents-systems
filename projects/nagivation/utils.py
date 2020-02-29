import yaml
yaml.warnings({"YAMLLoadWarning": False})


def load_config(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
