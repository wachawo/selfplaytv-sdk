#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The match picture as real pixels, through the kitty terminal graphics protocol.

Standard library only, and no numpy: the picture is anything with `shape` and
`tobytes`, so sdk.tui imports this without the engine. Two image ids are used
in turn and the other one is deleted after each frame, so the old picture
stays up until the new one is whole. Every command carries an id and q=2: an
image sent without one is kept by the terminal for good, and an answer would
arrive in the key stream of whoever reads the terminal.
"""

import base64
import os
import re
import select
import struct
import sys
import time
import zlib
from typing import Any, Mapping, Tuple

try:
    import fcntl
    import termios
    import tty
except ImportError:
    # Windows has none; there is no terminal to ask, and the answer is half blocks.
    fcntl = termios = tty = None

IMAGE_IDS = (7331, 7332)
# The protocol's limit for one escape's payload; 4096 is a multiple of four, so
# every chunk but the last decodes on its own.
CHUNK = 4096
# The size of a cell where the terminal reports no pixel size (TIOCGWINSZ gives
# 0): c and r make the terminal scale the picture to its real cells anyway.
CELL = (8, 16)
# The widest picture sent. The cost of a frame grows with its pixels, and past
# this the terminal scales it up at no cost to the line.
WIDE_MAX = 960
# A 1x1 image that is only queried, then a device-attributes request every
# terminal answers: a terminal that does not know the first is told apart from
# a dead line without waiting out the timeout.
PROBE_ID = 31
PROBE = f"\x1b_Gi={PROBE_ID},s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\\x1b[c"
PROBE_OK = f"\x1b_Gi={PROBE_ID};OK\x1b\\".encode()
ATTRIBUTES = re.compile(rb"\x1b\[\?[0-9;]*c")


def refused_by_environment(environ: Mapping[str, str]) -> bool:
    """No without asking: a multiplexer, or a terminal that cannot show an image.

    Inside tmux the question itself becomes the pane title. The hints that
    would say yes (KITTY_WINDOW_ID, TERM=xterm-kitty) decide nothing: they are
    inherited into tmux and lost over ssh.
    """
    term = environ.get("TERM", "")
    return bool(environ.get("TMUX") or environ.get("STY") or term.startswith(("tmux", "screen"))
                or environ.get("TERM_PROGRAM") == "tmux" or term in ("", "dumb", "linux"))


def probe(timeout: float = 0.5) -> bool:
    """Ask the terminal on stdin and stdout whether it shows kitty images; restores its mode."""
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        # No echo and no line buffering: the answer is read as it comes, never shown.
        tty.setcbreak(fd)
        os.write(sys.stdout.fileno(), PROBE.encode())
        seen = b""
        end = time.monotonic() + timeout
        while not ATTRIBUTES.search(seen):
            left = end - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                break
            seen += os.read(fd, 1024)
        return PROBE_OK in seen
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)


def detect(mode: str) -> bool:
    """Whether to send pixels: "kitty" yes, "blocks" no, "auto" asks when it is safe to."""
    if mode != "auto":
        return mode == "kitty"
    if termios is None or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    if refused_by_environment(os.environ):
        return False
    return probe()


def cell_pixels() -> Tuple[int, int]:
    """The (width, height) of one cell in pixels, from TIOCGWINSZ on stdout, or CELL."""
    try:
        rows, columns, wide, high = struct.unpack(
            "HHHH", fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, bytes(8)))
    except (OSError, AttributeError):
        return CELL
    if not (rows and columns and wide and high):
        return CELL
    return wide // columns, high // rows


def frame(image: Any, column: int, columns: int, rows: int, index: int) -> str:
    """The escapes that put an (H, W, 3) uint8 picture over `columns` x `rows` cells from row 1.

    Deflated (o=z): uncompressed, a 960x640 frame is 2.5 MB. c and r make the
    terminal scale it to exactly that rectangle, and C=1 leaves the cursor be.
    """
    high, wide = image.shape[:2]
    shown, other = IMAGE_IDS[index % 2], IMAGE_IDS[(index + 1) % 2]
    encoded = base64.standard_b64encode(zlib.compress(image.tobytes(), 1)).decode("ascii")
    chunks = [encoded[start:start + CHUNK] for start in range(0, len(encoded), CHUNK)] or [""]
    keys = f"a=T,f=24,o=z,s={wide},v={high},i={shown},p=1,c={columns},r={rows},C=1,q=2"
    parts = [f"\x1b[1;{column}H"]
    for number, chunk in enumerate(chunks):
        more = "" if len(chunks) == 1 else f",m={1 if number < len(chunks) - 1 else 0}"
        parts.append(f"\x1b_G{keys if number == 0 else 'q=2'}{more};{chunk}\x1b\\")
    parts.append(f"\x1b_Ga=d,d=I,i={other},q=2\x1b\\")
    return "".join(parts)


def forget() -> str:
    """The delete of both images and their data: erasing the text under a picture does not remove it."""
    return "".join(f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\" for image_id in IMAGE_IDS)


def main():
    pass


if __name__ == "__main__":
    main()
