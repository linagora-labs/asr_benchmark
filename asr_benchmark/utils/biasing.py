"""Settings of contextual biasing (see asr_benchmark/benchmark/contextual_biasing.py) that
the result pages need too: standard library only."""
import re

MANIFEST_KEY = "contextual_biasing"


def config_variant(config):
    """Name of the list given to the model according to the config, None for no list."""
    value = config.get("contextual_biasing")
    if value is None or str(value).lower() in ("none", "null", "false", ""):
        return None
    return str(value)


def folder_tag(config):
    """Suffix of the result folder: "_cb-<list>" when a list is given, else ""."""
    variant = config_variant(config)
    return "" if variant is None else re.sub(r"[^\w.\-]", "-", f"_cb-{variant}")
