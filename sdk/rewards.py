#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read a reward-weights file: one Lua table of numbers, parsed without a Lua runtime.

    python -m sdk.rewards examples/rewards.lua

    -- What your team is paid for while it learns. The scoreboard does not move.
    return {
      hit = 1.0,
      timeout = -2.0,
    }

A key is a REWARD_* knob of engine/config.py in lower case without the prefix,
and a key left out keeps its default. `python -m sdk.train --rewards FILE`
trains with the file; run as above, this module only checks it and prints the
weights it gives beside the defaults, so a typing mistake is found in a second
and not after the engine has started.

The file is DATA and is read as data. Lua is the notation because a table of
named numbers with comments is exactly what it writes well, but nothing here
runs Lua: a weights file is passed between people along with the team it
trained, and one that could execute code would be a program to audit before
every use. So the reader takes the one shape below and refuses everything else
with the line it stopped on:

- line comments `--` and block comments `--[[ ]]` (and `--[==[ ]==]`);
- an optional leading `return` or `rewards =`;
- one table of `name = number` pairs separated by `,` or `;`, the last
  separator optional;
- decimal numbers with an optional minus sign, a fraction and an exponent.

A file that starts with `return` or `rewards =` is also valid Lua, and evaluates
to the same table there, which is why `+1.0` is refused: Lua has no unary plus.
Without either word the bare table is still read here, though Lua itself would
not take it.
"""

import argparse
import logging
import math
import os
import re
import sys
from typing import Any, Dict, Iterator, List, Tuple

logger = logging.getLogger(__name__)

# What the knobs are called in engine/config.py, and so what a key maps onto:
# `timeout` in a file is REWARD_TIMEOUT there.
KNOB_PREFIX = "REWARD_"

# Lua's blank characters. Spelt out because str.isspace also passes the
# no-break space a web page leaves in a pasted line, which Lua refuses.
BLANKS = " \t\r\n\f\v"

NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# [0-9] and not \d, which in Python also matches the digits of other scripts:
# float() would read them, and Lua would not.
NUMBER_PATTERN = re.compile(r"(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
# What may not follow a number directly. Lua reads `0x10`, `1.0.0` or `5abc` as
# one malformed number, and so must this: taking the leading `0` of `0x10` as a
# whole number would set a weight to zero without a word.
NUMBER_TAIL_PATTERN = re.compile(r"[A-Za-z0-9_.]+")
# `--[[`, or with any number of `=` between the brackets; the comment then runs
# to the closing bracket with the same number of them, across lines.
BLOCK_COMMENT_PATTERN = re.compile(r"--\[(=*)\[")

# The words that may stand before the table, and the only two names the reader
# knows that are not weights.
RETURN_WORD = "return"
ASSIGNED_NAME = "rewards"

# Said after the token that was found, when what was found is a recognisable
# attempt at something a weights file cannot hold. Two tables because the same
# character is a different mistake in a different place: a quote where a number
# should be is a string value, and where a name should be it is a quoted key.
ARITHMETIC_HINT = "arithmetic: write the result as one number"
VALUE_HINTS = {
    "{": "a nested table: a weight is one number",
    "(": "a function call: a weight is one number",
    '"': "a string: a weight is a number, written without quotes",
    "'": "a string: a weight is a number, written without quotes",
    "+": "Lua has no unary plus, and arithmetic is not read: write the number itself",
    "-": ARITHMETIC_HINT,
    "*": ARITHMETIC_HINT,
    "/": ARITHMETIC_HINT,
    "%": ARITHMETIC_HINT,
    "^": ARITHMETIC_HINT,
    ".": "a weight is one number, with nothing joined to it",
}
KEY_HINTS = {
    "{": "a nested table: the file is one table of name = number",
    "[": "a key is a plain name, as in hit = 1.0",
    '"': "a key is a plain name, as in hit = 1.0",
    "'": "a key is a plain name, as in hit = 1.0",
}
NO_HINTS: Dict[str, str] = {}

Token = Tuple[str, str, int]


def scan(text: str) -> Iterator[Token]:
    """The file as (kind, text, line) tokens, comments and blank space dropped.

    Kinds: "name", "number", "malformed" (a number with something stuck to it),
    "symbol" (any other single character) and, last, "end". A generator, so
    that the parser stops at the first thing it refuses and the rest of the
    file is never looked at: after a quote, say, nothing that follows can be
    told apart from the inside of the string.
    """
    position = 0
    line = 1
    # The end of the file is reported on the last line that holds anything: a
    # table left open is then pointed at where its brace belongs, and not at a
    # line after the last one, which no editor shows.
    last = 1
    while position < len(text):
        char = text[position]
        if char in BLANKS:
            if char == "\n":
                line += 1
            position += 1
            continue
        if text.startswith("--", position):
            opening = BLOCK_COMMENT_PATTERN.match(text, position)
            if not opening:
                end = text.find("\n", position)
                position = len(text) if end < 0 else end
                continue
            closing = f"]{opening.group(1)}]"
            end = text.find(closing, opening.end())
            if end < 0:
                raise ValueError(f"line {line}: this block comment is never closed; expected "
                                 f"{closing} before the end of the file")
            line += text.count("\n", position, end)
            position = end + len(closing)
            continue
        last = line
        number = NUMBER_PATTERN.match(text, position)
        if number:
            tail = NUMBER_TAIL_PATTERN.match(text, number.end())
            if tail:
                yield "malformed", text[position:tail.end()], line
                position = tail.end()
            else:
                yield "number", number.group(), line
                position = number.end()
            continue
        name = NAME_PATTERN.match(text, position)
        if name:
            yield "name", name.group(), line
            position = name.end()
            continue
        yield "symbol", char, line
        position += 1
    yield "end", "", last


def describe_token(token: Token, hints: Dict[str, str]) -> str:
    """A token as the error message names it, with a hint when the mistake is a known one."""
    kind, text, unused_line = token
    if kind == "end":
        return "the end of the file"
    if kind == "name":
        return f"the name {text!r}"
    if kind == "number":
        return f"the number {text}"
    if kind == "malformed":
        return f"{text!r}, which is not a decimal number such as -2, 0.5 or 1e-3"
    hint = hints.get(text)
    return f"{text!r} ({hint})" if hint else repr(text)


def unexpected(expected: str, token: Token, hints: Dict[str, str] = NO_HINTS) -> ValueError:
    return ValueError(f"line {token[2]}: expected {expected}, found {describe_token(token, hints)}")


def is_symbol(token: Token, text: str) -> bool:
    return token[0] == "symbol" and token[1] == text


def parse_number(name: str, tokens: Iterator[Token]) -> float:
    """The value after `name =`: an optional minus and one decimal number."""
    token = next(tokens)
    negative = is_symbol(token, "-")
    if negative:
        token = next(tokens)
    if token[0] != "number":
        raise unexpected(f"a number for {name}", token, VALUE_HINTS)
    value = float(token[1])
    # 1e999 is a number to the pattern and infinity to float(), and an infinite
    # reward is a network full of NaN one update later.
    if not math.isfinite(value):
        raise ValueError(f"line {token[2]}: {token[1]} is too large to be a weight for {name}")
    return -value if negative else value


def parse_table(text: str) -> List[Tuple[str, float, int]]:
    """The file's `name = number` pairs as (name, value, line), in the order written.

    Raises ValueError, its message starting with `line N:`, at the first thing
    that is not the shape the module docstring describes. Which names are
    weights is not decided here; see knob_weights.
    """
    tokens = scan(text)
    token = next(tokens)
    if token[0] == "name" and token[1] == RETURN_WORD:
        token = next(tokens)
    elif token[0] == "name" and token[1] == ASSIGNED_NAME:
        token = next(tokens)
        if not is_symbol(token, "="):
            raise unexpected(f"'=' after {ASSIGNED_NAME}", token)
        token = next(tokens)
    if not is_symbol(token, "{"):
        raise unexpected(f"'{{' to open the table, after an optional '{RETURN_WORD}' or "
                         f"'{ASSIGNED_NAME} ='", token)

    pairs = []
    line_of = {}
    token = next(tokens)
    while not is_symbol(token, "}"):
        if token[0] != "name":
            raise unexpected("a weight name or '}' to close the table", token, KEY_HINTS)
        unused_kind, name, line = token
        # Lua keeps the last of two and says nothing; here the second is far
        # more likely a slip than a decision, and which one won would not show.
        if name in line_of:
            raise ValueError(f"line {line}: {name} is set twice, first on line {line_of[name]}; "
                             f"expected each weight once")
        token = next(tokens)
        if not is_symbol(token, "="):
            raise unexpected(f"'=' after {name}", token)
        pairs.append((name, parse_number(name, tokens), line))
        line_of[name] = line
        token = next(tokens)
        if is_symbol(token, ",") or is_symbol(token, ";"):
            token = next(tokens)
        elif not is_symbol(token, "}"):
            raise unexpected(f"',' or '}}' after the value of {name}", token, VALUE_HINTS)
    token = next(tokens)
    if token[0] != "end":
        raise unexpected("the end of the file after '}'", token)
    return pairs


def knob_names(config: Any) -> List[str]:
    """Every reward knob there is, read off the config module itself, sorted.

    Off the module and not from a list kept here: a knob added to
    engine/config.py is then a key the file may set on the same day, and a
    second list is one that would be a knob short the day after.
    """
    return sorted(name for name in dir(config)
                  if name.startswith(KNOB_PREFIX) and isinstance(getattr(config, name), (int, float)))


def key_of(knob: str) -> str:
    """REWARD_CAPTURE_COVER as a file spells it: capture_cover."""
    return knob[len(KNOB_PREFIX):].lower()


def knob_weights(pairs: List[Tuple[str, float, int]], knobs: List[str]) -> Dict[str, float]:
    """The parsed pairs as {REWARD_NAME: value}, refusing a key that is no knob.

    Refused and not skipped: a misspelt `timeuot = -2` skipped is a run that
    trains for hours on the default while its owner believes otherwise.
    """
    knob_of = {key_of(knob): knob for knob in knobs}
    weights = {}
    for name, value, line in pairs:
        if name not in knob_of:
            raise ValueError(f"line {line}: unknown weight {name!r}; expected one of "
                             f"{', '.join(sorted(knob_of))}")
        weights[knob_of[name]] = value
    return weights


def read_rewards(path: str, knobs: List[str]) -> Dict[str, float]:
    """A weights file as {REWARD_NAME: value}, holding only the knobs the file sets.

    Raises OSError when the file cannot be read and ValueError (`line N: ...`)
    when it is not a weights file; the caller says which file.
    """
    # utf-8-sig drops the byte-order mark some Windows editors put first in a
    # file, which would otherwise be the first thing the parser trips over.
    with open(path, encoding="utf-8-sig") as handle:
        try:
            text = handle.read()
        except UnicodeDecodeError:
            raise ValueError("expected a text file in UTF-8, and this one is not") from None
    return knob_weights(parse_table(text), knobs)


def write_rewards(path: str, weights: Dict[str, float]) -> None:
    """Write {REWARD_NAME: value} as a weights file read_rewards gives back unchanged.

    repr(float) is the shortest text that reads back as the same float, and
    always has a digit after the point or an exponent, both of which the
    parser takes. Written beside the file and renamed in, so a trainer that
    starts in the same instant reads the old file or the new one, never half;
    staged under the writer's pid, as two sdk.tui on one run may both write.
    """
    lines = ["-- Reward weights, written whole by sdk.tui: edits made here last until the next one.",
             "-- What each weight pays: examples/rewards.lua", "return {"]
    lines.extend(f"  {key_of(knob)} = {float(value)!r}," for knob, value in weights.items())
    staging = f"{path}.{os.getpid()}.tmp"
    with open(staging, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines + ["}"]) + "\n")
    os.replace(staging, path)


def render_check(path: str, defaults: Dict[str, float], weights: Dict[str, float]) -> List[str]:
    """The report of `python -m sdk.rewards FILE`: every weight, its default and the file's value."""
    changed = [knob for knob, value in weights.items() if value != defaults[knob]]
    width = max(len(key_of(knob)) for knob in defaults)
    knob_width = max(len(knob) for knob in defaults)
    lines = [f"{path}: {len(weights)} of {len(defaults)} weights set, "
             f"{len(changed)} different from the default", "",
             f"  {'weight':<{width}}  {'knob':<{knob_width}}  {'default':>8}  {'file':>8}  {'in force':>8}"]
    for knob, default in defaults.items():
        written = f"{weights[knob]:g}" if knob in weights else "-"
        row = (f"  {key_of(knob):<{width}}  {knob:<{knob_width}}  {default:>8g}  {written:>8}"
               f"  {weights.get(knob, default):>8g}")
        lines.append(row + ("  changed" if knob in changed else ""))
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sdk.rewards",
        description="Check a reward-weights file and print its weights beside the defaults")
    parser.add_argument("file", help="the Lua weights file, e.g. examples/rewards.lua")
    return parser.parse_args()


def main():
    args = parse_args()
    # Imported here and not at the top: sdk.train imports this module, and the
    # engine is only needed by the check, for the list of knobs and their defaults.
    from sdk.train import LOGGING
    from engine import config
    logging.basicConfig(**LOGGING)
    defaults = {knob: float(getattr(config, knob)) for knob in knob_names(config)}
    try:
        weights = read_rewards(args.file, list(defaults))
    except OSError as exc:
        logger.error(f"{args.file}: {exc.strerror}")
        sys.exit(2)
    except ValueError as exc:
        logger.error(f"{args.file}: {str(exc)}")
        sys.exit(2)
    sys.stdout.write("\n".join(render_check(args.file, defaults, weights)) + "\n")


if __name__ == "__main__":
    main()
