"""Readable YAML for pack test files: multi-line strings as | blocks, keys in the given order."""
import yaml


class _Dumper(yaml.SafeDumper):
    pass


def _str(dumper, value):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str)


def dump_tests(tests, path, header=""):
    body = yaml.dump(tests, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)
    body = body.replace("\n- ", "\n\n- ")  # blank line between tests
    with open(path, "w") as f:
        f.write(header + body)
