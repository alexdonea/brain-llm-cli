"""config.py - the CLI's OPTIONAL config.yaml.

Entirely optional. If <repo>/config.yaml is absent - or any single key is missing or malformed -
the value falls back to its built-in default, so behavior is byte-for-byte today's. Only SAFE,
user-facing CLI knobs are read here (paths, program name, recall window sizes, the semantic toggle,
the live view's pacing). The affect/memory PHYSICS constants are deliberately NOT configurable - they
are tuned laws, not preferences - so a config file can never put the engine into a degenerate state.

Precedence for every key (highest wins):  CLI flag  >  environment variable  >  config.yaml  >  default.

config.yaml is committed at its documented defaults and edited in place; your changes travel with the repo.
"""
import os
import sys

import yaml

_CACHE = {}


def load(repo):
    """Read <repo>/config.yaml and cache it PER repo path. Returns a dict (possibly empty). NEVER raises:
    a missing file, an unreadable file, invalid YAML, a non-mapping top level, or even a non-str/unhashable repo
    all degrade to {} (with a one-line stderr note for the malformed cases) so a broken config can't break the
    CLI. The returned dict is CACHED and SHARED (agent.py CONFIG and runtime.py _CFG get the same object) -
    callers MUST treat it as read-only."""
    key = repo if isinstance(repo, str) else str(repo)      # a stable, hashable cache key even for a PathLike/odd repo - str() never raises
    if key in _CACHE:                                       # two callers (agent.py, runtime.py) share one read; tests reset _CACHE
        return _CACHE[key]
    data = {}
    try:
        path = os.path.join(repo, "config.yaml")            # inside try so a non-str repo can't break the 'never raises' contract
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:        # context-managed so the handle is closed deterministically
                loaded = yaml.safe_load(fh) or {}
            if isinstance(loaded, dict):
                data = loaded
            else:
                print("brain-llm: config.yaml is not a mapping - ignoring it.", file=sys.stderr)
    except Exception as e:                                   # unreadable / invalid YAML - never fatal
        print(f"brain-llm: could not read config.yaml ({e}) - using built-in defaults.", file=sys.stderr)
    _CACHE[key] = data
    return data


def get(cfg, section, key, default):
    """cfg[section][key] when present and non-null, else `default`. Tolerates a missing/!dict section."""
    sec = cfg.get(section) if isinstance(cfg, dict) else None
    if isinstance(sec, dict) and sec.get(key) is not None:
        return sec[key]
    return default


def num(cfg, section, key, default, lo=None, hi=None):
    """A finite float from config, clamped to [lo, hi]. Warns + returns `default` on a non-finite/bad value."""
    v = get(cfg, section, key, default)
    try:
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError
    except (TypeError, ValueError):
        print(f"brain-llm: config {section}.{key}={v!r} is not a number - using {default}.", file=sys.stderr)
        return default
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


def intnum(cfg, section, key, default, lo=None, hi=None):
    """Like num(), but returns an int (for window sizes / counts)."""
    return int(num(cfg, section, key, default, lo, hi))


def enum(cfg, section, key, default, choices):
    """A value restricted to `choices`. Warns + returns `default` on anything else."""
    v = get(cfg, section, key, default)
    if v in choices:
        return v
    print(f"brain-llm: config {section}.{key}={v!r} not in {tuple(choices)} - using {default}.", file=sys.stderr)
    return default
