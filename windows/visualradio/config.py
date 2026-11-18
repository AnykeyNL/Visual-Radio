"""config.ini: API keys (and, if you like, your usual options) in a file, so they
don't have to be typed on the command line every time.

    [keys]
    acoustid_key = abcd1234
    hf_token = hf_xxxxxxxx

    [options]          ; optional: defaults for any command-line option
    web = yes
    delay = 45

    [state]            ; written by the script itself
    last_station = 102.7

Order: what you type on the command line wins, then environment variables
(ACOUSTID_KEY, HF_TOKEN), then this file. The file is in .gitignore so the keys
stay on this PC; config.example.ini shows every setting.
"""

from __future__ import annotations

import argparse
import configparser
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILE = os.path.join(ROOT, "config.ini")
KEYS = {"acoustid_key": "ACOUSTID_KEY", "hf_token": "HF_TOKEN"}     # setting -> environment variable
_TRUE = {"1", "yes", "true", "on"}
_FALSE = {"0", "no", "false", "off", ""}


class ConfigError(ValueError):
    pass


def _value(action, raw):
    """Text from the file -> the value argparse would give for this option."""
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        low = raw.strip().lower()
        if low not in _TRUE | _FALSE:
            raise ConfigError(f"{action.dest} = {raw}: use yes or no")
        return low in _TRUE      # the setting is named after the value (dest), e.g. save_transcript = no
    val = action.type(raw) if action.type else raw
    if action.choices and val not in action.choices:
        raise ConfigError(f"{action.dest} = {raw}: choose one of {', '.join(map(str, action.choices))}")
    return val


def load_defaults(parser, path):
    """Read the file -> ({option: value}, [names set]); ({}, []) when there is no file."""
    if not path or not os.path.isfile(path):
        return {}, []
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None)
    try:
        cp.read(path, encoding="utf-8-sig")
    except configparser.Error as exc:
        raise ConfigError(f"can't read {path}: {exc}") from exc
    actions = {a.dest: a for a in parser._actions if a.dest not in ("help", "config")}
    out = {}
    for section in cp.sections():
        if section.lower() == "state":
            continue                                       # see read_state
        if section.lower() not in ("keys", "options"):
            raise ConfigError(f"{path}: unknown section [{section}] (use [keys] and [options])")
        for name, raw in cp.items(section):
            dest = name.strip().replace("-", "_")
            if dest not in actions:
                raise ConfigError(f"{path}: unknown setting '{name}' in [{section}]")
            if dest in KEYS and os.environ.get(KEYS[dest]):
                continue                                   # the environment variable wins
            if raw.strip() == "" and dest in KEYS:
                continue                                   # left empty in the example file
            try:
                out[dest] = _value(actions[dest], raw.strip())
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{path}: {exc}") from exc
    return out, sorted(out)


def parse_args(parser, argv=None):
    """parser.parse_args with defaults from the config file (--config FILE, else config.ini).
    -> (args, path or None, [names taken from the file])"""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    path = known.config or DEFAULT_FILE
    if known.config and not os.path.isfile(path):
        parser.error(f"config file not found: {path}")
    try:
        defaults, names = load_defaults(parser, path)
    except ConfigError as exc:
        parser.error(str(exc))
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.config_path = path                               # also where read_state / save_state go
    return args, (path if names else None), names


def read_state(path):
    """The [state] section (things the script remembers, like the last station) -> dict."""
    if not path or not os.path.isfile(path):
        return {}
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None)
    try:
        cp.read(path, encoding="utf-8-sig")
    except configparser.Error:
        return {}
    for section in cp.sections():
        if section.lower() == "state":
            return dict(cp.items(section))
    return {}


def save_state(path, **values):
    """Write values into the [state] section, leaving the rest of the file (comments,
    keys, your options) exactly as it is. Makes the file when there is none."""
    if os.path.isfile(path):
        with open(path, encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
    else:
        lines = ["; Settings for visual_radio.py, see config.example.ini"]
    head = next((i for i, l in enumerate(lines) if l.strip().lower() == "[state]"), None)
    if head is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ["", "[state]", "; remembered by the script itself"]
        head = len(lines) - 2
    end = next((i for i in range(head + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
    for key, value in values.items():
        pat = re.compile(rf"^\s*{re.escape(key)}\s*[=:]", re.I)
        hit = next((i for i in range(head + 1, end) if pat.match(lines[i])), None)
        if hit is not None:
            lines[hit] = f"{key} = {value}"
        else:
            at = end
            while at > head + 1 and not lines[at - 1].strip():
                at -= 1
            lines.insert(at, f"{key} = {value}")
            end += 1
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
