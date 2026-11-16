#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual themes for generated arenas: palettes and obstacle vocabularies.

A theme is purely cosmetic plus a distribution over obstacle shapes. The
simulation only ever sees rectangles and circles; the theme decides how many
of each, how big they are, and what the renderer draws on top.
"""

from typing import Any, Dict, List

# Obstacle kinds understood by the renderer. The physics layer collapses each
# one to either a rectangle or a circle via the "shape" field.
THEMES: List[Dict[str, Any]] = [
    {
        "key":   "warehouse",
        "name":  "Warehouse",
        "floor": "#6d6a66",
        "floor_alt": "#7a7773",
        "grid":  "#5d5a57",
        "fog":   "#4a4744",
        "kinds": [
            {"kind": "container", "shape": "rect",   "w": (2.6, 5.0), "h": (1.4, 2.2), "color": "#c9722c", "edge": "#8f4d18", "weight": 4},
            {"kind": "crate",     "shape": "rect",   "w": (1.1, 1.8), "h": (1.1, 1.8), "color": "#b3833f", "edge": "#7d5a26", "weight": 3},
            {"kind": "barrel",    "shape": "circle", "r": (0.5, 0.8),                  "color": "#4f7fa8", "edge": "#33566f", "weight": 2},
        ],
    },
    {
        "key":   "forest",
        "name":  "Forest",
        "floor": "#4a7a3c",
        "floor_alt": "#548a44",
        "grid":  "#416b35",
        "fog":   "#2f5228",
        "kinds": [
            {"kind": "tree",  "shape": "circle", "r": (0.7, 1.2),                  "color": "#2f6b32", "edge": "#1e4721", "weight": 5},
            {"kind": "bush",  "shape": "circle", "r": (0.5, 0.85),                 "color": "#69a04a", "edge": "#4a7433", "weight": 2},
            {"kind": "log",   "shape": "rect",   "w": (2.2, 4.0), "h": (0.7, 1.0), "color": "#7a5533", "edge": "#563a22", "weight": 2},
        ],
    },
    {
        "key":   "canyon",
        "name":  "Canyon",
        "floor": "#c9a36b",
        "floor_alt": "#d4b07b",
        "grid":  "#b8925c",
        "fog":   "#9c7743",
        "kinds": [
            {"kind": "rock",   "shape": "rect",   "w": (1.8, 4.2), "h": (1.2, 2.6), "color": "#a0673d", "edge": "#74462a", "weight": 4},
            {"kind": "boulder", "shape": "circle", "r": (0.8, 1.4),                 "color": "#8d6142", "edge": "#63422c", "weight": 3},
            {"kind": "cactus", "shape": "circle", "r": (0.45, 0.7),                 "color": "#4f8a54", "edge": "#356038", "weight": 2},
        ],
    },
    {
        "key":   "snowbase",
        "name":  "Snow Base",
        "floor": "#dfe9f2",
        "floor_alt": "#eef4fa",
        "grid":  "#c6d5e3",
        "fog":   "#aabecf",
        "kinds": [
            {"kind": "iceblock", "shape": "rect",   "w": (1.6, 3.6), "h": (1.2, 2.2), "color": "#8fc4e0", "edge": "#5f97b8", "weight": 4},
            {"kind": "drift",    "shape": "circle", "r": (0.8, 1.5),                  "color": "#f3f9ff", "edge": "#bdd2e2", "weight": 3},
            {"kind": "crate",    "shape": "rect",   "w": (1.1, 1.7), "h": (1.1, 1.7), "color": "#6d7f8e", "edge": "#4a5865", "weight": 2},
        ],
    },
    {
        "key":   "neon",
        "name":  "Neon",
        "floor": "#161a2b",
        "floor_alt": "#1d2238",
        "grid":  "#2a3152",
        "fog":   "#0d1020",
        "kinds": [
            {"kind": "neonblock", "shape": "rect",   "w": (1.8, 4.0), "h": (1.2, 2.2), "color": "#1f8fa8", "edge": "#2ee6ff", "weight": 4, "glow": "#2ee6ff"},
            {"kind": "pylon",     "shape": "circle", "r": (0.6, 1.0),                  "color": "#8a2199", "edge": "#ff5cf0", "weight": 3, "glow": "#ff5cf0"},
            {"kind": "slab",      "shape": "rect",   "w": (1.2, 2.0), "h": (1.2, 2.0), "color": "#2c2f56", "edge": "#6b74c9", "weight": 2},
        ],
    },
]

THEMES_BY_KEY = {t["key"]: t for t in THEMES}
THEME_KEYS = [t["key"] for t in THEMES]


def get_theme(key: str) -> Dict[str, Any]:
    """Return a theme by key, falling back to the first theme."""
    return THEMES_BY_KEY.get(key, THEMES[0])


def shade(colour: str, factor: float) -> str:
    """One colour towards black (factor < 1) or towards white (factor > 1)."""
    text = str(colour or "").lstrip("#")
    if len(text) != 6:
        return "#888888"
    try:
        channels = [int(text[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return "#888888"
    if factor <= 1.0:
        moved = [round(channel * factor) for channel in channels]
    else:
        reach = factor - 1.0
        moved = [round(channel + (255 - channel) * reach) for channel in channels]
    return "#" + "".join(f"{max(0, min(255, channel)):02x}" for channel in moved)


# The contrast `readable` moves a colour to unless told otherwise. Large bold
# type clears at 3:1 under WCAG; 4.5 is the ratio it asks of ordinary text, and
# is where a team's name stops looking dimmer than the text around it.
NAME_CONTRAST = 4.5


def relative_luminance(colour: str) -> float:
    """WCAG relative luminance of an "#rrggbb" colour."""
    text = str(colour or "").lstrip("#")
    if len(text) != 6:
        return 0.5
    try:
        channels = [int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except ValueError:
        return 0.5
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(one: str, other: str) -> float:
    """WCAG contrast ratio between two "#rrggbb" colours."""
    first, second = relative_luminance(one), relative_luminance(other)
    high, low = max(first, second), min(first, second)
    return (high + 0.05) / (low + 0.05)


def readable(colour: str, background: str, target: float = NAME_CONTRAST) -> str:
    """`colour`, moved away from `background` until it can be read on it.

    The direction is decided by the background and not by the colour: on a dark
    panel every ink goes lighter and on a pale one every ink goes darker, so a
    team keeps its hue and changes only how far it is from the surface it is
    written on. Twenty steps of 5%, which reaches white or black from anything;
    a colour that cannot make the target even there is returned at its limit
    rather than not returned at all -- an unreadable name is still better than a
    missing one, and whoever chose the colour can see what they picked.
    """
    if contrast(colour, background) >= target:
        return colour
    lighten = relative_luminance(background) < 0.5
    moved = colour
    for step in range(1, 21):
        moved = shade(colour, 1.0 + 0.05 * step if lighten else 1.0 - 0.05 * step)
        if contrast(moved, background) >= target:
            return moved
    return moved


def main():
    pass


if __name__ == "__main__":
    main()
