"""
palettes.py — color-scheme registry (including color-vision-friendly schemes)
=================================================

This module centralizes all Time_Hydrogen_v2 color schemes so the CLI, GUI, and
plot modules use the same color definitions.

Design notes
--------
1. Keep the two original schemes:
   - ``classic``: plot1.py warm real part (white -> yellow -> orange -> red)
     and cool imaginary part (white -> purple -> blue); Re/Im panels show **magnitude** (|Re psi|, |Im psi|).
   - ``signed``: the archived plot1+.py RdBu diverging map; Re/Im panels show
     **signed values** (negative blue, positive red).
2. Add color-vision-deficiency (CVD) schemes:
   - ``cividis``   general colorblind-friendly (blue-yellow, monotonic lightness)
   - ``deuteranopia`` green-blind friendly
   - ``protanopia``   red-blind friendly
   - ``tritanopia``   blue-blind friendly
   - ``mono``         achromatopsia / grayscale
3. Each scheme defines ``density``, ``shell``,
   ``real`` / ``imag`` (magnitude mode), and ``signed`` (signed mode) colormaps.

Terminology
----
- magnitude mode: Re/Im panels show |Re psi| and |Im psi| with sequential colormaps; 0 is dark.
- signed mode: Re/Im panels show Re psi and Im psi with diverging colormaps; 0 is in the middle.

All functions accept a scheme-name string or a scheme dictionary, which enables custom CLI schemes.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Union

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_hex

# ---------------------------------------------------------------------------
# Two custom sequential colormaps from the original plot1.py
# ---------------------------------------------------------------------------
WARM = LinearSegmentedColormap.from_list(
    "th_warm", [(1, 1, 1), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)], N=256
)
COOL = LinearSegmentedColormap.from_list(
    "th_cool", [(1, 1, 1), (0.8, 0.5, 1), (0.29, 0, 0.51), (0, 0, 1)], N=256
)

_CUSTOM_CMAPS: Dict[str, Colormap] = {
    "th_warm": WARM,
    "th_cool": COOL,
}

Scheme = Union[str, dict]


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------
SCHEMES: "OrderedDict[str, dict]" = OrderedDict(
    [
        (
            "classic",
            {
                "key": "classic",
                "label": "Classic warm/cool (original plot1)",
                "short": "classic",
                "description": (
                    "Warm real part (white -> yellow -> orange -> red), cool imaginary part "
                    "(white -> purple -> blue); Re/Im show magnitude; density uses viridis."
                ),
                "cvd": "No specific adaptation; red-green contrast is reduced for deuteranopia",
                "magnitude": True,
                "density": "viridis",
                "shell": "inferno",
                "real": "th_warm",
                "imag": "th_cool",
                "signed": "RdBu",
                "contour": "white",
                "boundary": "red",
            },
        ),
        (
            "signed",
            {
                "key": "signed",
                "label": "Red/blue diverging (original plot1+)",
                "short": "signed",
                "description": (
                    "Re/Im use the RdBu diverging map and keep signs (negative blue, positive red, "
                    "zero white); density uses viridis."
                ),
                "cvd": "Not red-green friendly (red/blue remain distinguishable by shape)",
                "magnitude": False,
                "density": "viridis",
                "shell": "inferno",
                "real": "RdBu",
                "imag": "RdBu",
                "signed": "RdBu",
                "contour": "white",
                "boundary": "magenta",
            },
        ),
        (
            "cividis",
            {
                "key": "cividis",
                "label": "Colorblind-friendly (Cividis)",
                "short": "cividis",
                "description": (
                    "Density/shell use Cividis (blue -> yellow, monotonic lightness); Re/Im use PuOr "
                    "diverging purple-orange, readable for red-green and blue-yellow deficiencies."
                ),
                "cvd": "Readable for red-green and blue-yellow deficiencies; recommended general scheme",
                "magnitude": False,
                "density": "cividis",
                "shell": "cividis",
                "real": "PuOr",
                "imag": "PuOr",
                "signed": "PuOr",
                "contour": "white",
                "boundary": "magenta",
            },
        ),
        (
            "deuteranopia",
            {
                "key": "deuteranopia",
                "label": "Green-blind friendly (Deuteranopia)",
                "short": "deuteranopia",
                "description": (
                    "Deuteranopia scheme: density uses Cividis; Re/Im use PuOr "
                    "purple-orange diverging and avoid red/green opposition."
                ),
                "cvd": "Deuteranopia (green-blind)",
                "magnitude": False,
                "density": "cividis",
                "shell": "cividis",
                "real": "PuOr",
                "imag": "PuOr",
                "signed": "PuOr",
                "contour": "white",
                "boundary": "magenta",
            },
        ),
        (
            "protanopia",
            {
                "key": "protanopia",
                "label": "Red-blind friendly (Protanopia)",
                "short": "protanopia",
                "description": (
                    "Protanopia scheme: density uses Cividis; Re/Im use PuOr purple-orange, "
                    "relying on lightness and blue-orange contrast."
                ),
                "cvd": "Protanopia (red-blind)",
                "magnitude": False,
                "density": "cividis",
                "shell": "cividis",
                "real": "PuOr",
                "imag": "PuOr",
                "signed": "PuOr",
                "contour": "white",
                "boundary": "magenta",
            },
        ),
        (
            "tritanopia",
            {
                "key": "tritanopia",
                "label": "Blue-blind friendly (Tritanopia)",
                "short": "tritanopia",
                "description": (
                    "Tritanopia scheme: density uses Magma; Re/Im use PiYG "
                    "pink-green diverging and avoid blue/yellow opposition."
                ),
                "cvd": "Tritanopia (blue-blind)",
                "magnitude": False,
                "density": "magma",
                "shell": "magma",
                "real": "PiYG",
                "imag": "PiYG",
                "signed": "PiYG",
                "contour": "white",
                "boundary": "red",
            },
        ),
        (
            "mono",
            {
                "key": "mono",
                "label": "Achromatopsia / grayscale",
                "short": "mono",
                "description": (
                    "Pure grayscale: density/shell/Re/Im all use Greys; panels are distinguished "
                    "by lightness and position. Suitable for achromatopsia and black-and-white printing."
                ),
                "cvd": "Achromatopsia / black-and-white printing",
                "magnitude": False,
                "density": "Greys",
                "shell": "Greys",
                "real": "Greys",
                "imag": "Greys",
                "signed": "Greys",
                "contour": "white",
                "boundary": "black",
            },
        ),
        (
            "turbo",
            {
                "key": "turbo",
                "label": "High-saturation Turbo",
                "short": "turbo",
                "description": (
                    "High-saturation Turbo colormap; Re/Im show magnitude. Strong visual impact, "
                    "but no colorblind adaptation; suitable for projected demonstrations."
                ),
                "cvd": "Not recommended for colorblind users",
                "magnitude": True,
                "density": "turbo",
                "shell": "turbo",
                "real": "turbo",
                "imag": "turbo",
                "signed": "turbo",
                "contour": "white",
                "boundary": "red",
            },
        ),
    ]
)


SCHEME_KEYS = list(SCHEMES.keys())


# ---------------------------------------------------------------------------
# Scheme resolution
# ---------------------------------------------------------------------------
def resolve_scheme(scheme: Scheme) -> dict:
    """Resolve a scheme-name string to a scheme dictionary; dictionaries are returned unchanged."""
    if isinstance(scheme, dict):
        merged = {
            "key": scheme.get("key", "custom"),
            "label": scheme.get("label", "Custom scheme"),
            "short": scheme.get("short", "custom"),
            "description": scheme.get("description", ""),
            "cvd": scheme.get("cvd", "custom"),
            "magnitude": bool(scheme.get("magnitude", False)),
            "density": scheme.get("density", "viridis"),
            "shell": scheme.get("shell", "inferno"),
            "real": scheme.get("real", scheme.get("signed", "PuOr")),
            "imag": scheme.get("imag", scheme.get("signed", "PuOr")),
            "signed": scheme.get("signed", "PuOr"),
            "contour": scheme.get("contour", "white"),
            "boundary": scheme.get("boundary", "magenta"),
        }
        return merged
    try:
        return SCHEMES[str(scheme)]
    except KeyError as exc:
        # Allow a matplotlib colormap name to be used directly, e.g. scheme='cividis'
        name = str(scheme)
        try:
            get_cmap(name)
        except Exception:
            raise KeyError(
                f"Unknown color scheme: {scheme!r}; available: {', '.join(SCHEME_KEYS)}"
            ) from exc
        return make_custom_scheme(
            density=name, shell=name, real="PuOr", imag="PuOr",
            signed="PuOr", label=f"Single-colormap scheme ({name})",
        )


def is_magnitude(scheme: Scheme) -> bool:
    """Whether Re/Im use magnitude + sequential colormaps."""
    return bool(resolve_scheme(scheme).get("magnitude", False))


def get_cmap(name) -> Colormap:
    """Get a matplotlib colormap by name, including project-defined colormaps."""
    if isinstance(name, Colormap):
        return name
    key = str(name)
    if key in _CUSTOM_CMAPS:
        return _CUSTOM_CMAPS[key]
    try:
        return matplotlib.colormaps[key]
    except KeyError:
        # Try case-insensitive lookup so rdbu / viridis / Cividis all work
        lower = {k.lower(): k for k in matplotlib.colormaps}
        if key.lower() in lower:
            return matplotlib.colormaps[lower[key.lower()]]
        raise KeyError(
            f"Unknown matplotlib colormap: {key!r}. See "
            f"https://matplotlib.org/stable/users/explain/colors/colormaps.html"
        ) from None


def cmap_for(scheme: Scheme, role: str) -> Colormap:
    """
    Get a colormap for a scheme and role.

    role:
        'density'  probability density |psi|^2
        'shell'    shell density 4*pi*r^2*|psi|^2
        'real'     real-part panel
        'imag'     imaginary-part panel
        'signed'   signed panel (chosen automatically from magnitude)
    """
    s = resolve_scheme(scheme)
    if role in ("real", "imag") and not s.get("magnitude", False):
        return get_cmap(s["signed"])
    if role == "signed":
        return get_cmap(s["signed"])
    if role not in s:
        raise KeyError(f"Unknown role: {role!r}")
    return get_cmap(s[role])


def plotly_scale(scheme_or_name, n_samples: int = 33):
    """
    Convert a matplotlib colormap to a Plotly colorscale.

    Everything is converted to [[0, '#xxxxxx'], ...] so custom and matplotlib
    colormaps both work, without relying on built-in Plotly names.
    """
    cmap = get_cmap(scheme_or_name)
    if n_samples < 2:
        n_samples = 2
    return [
        [i / (n_samples - 1), to_hex(cmap(i / (n_samples - 1)))]
        for i in range(n_samples)
    ]


def describe_scheme(scheme: Scheme) -> str:
    """Return a readable description for the CLI."""
    s = resolve_scheme(scheme)
    mode = "magnitude mode" if s.get("magnitude") else "signed mode"
    return (
        f"[{s['key']}] {s['label']}\n"
        f"  Colormaps: density={s['density']}, shell={s['shell']}, "
        f"real={s['real']}, imag={s['imag']}, signed={s['signed']}\n"
        f"  Re/Im: {mode}\n"
        f"  Color vision: {s['cvd']}\n"
        f"  Notes: {s['description']}"
    )


def scheme_table() -> str:
    """Scheme list for the CLI menu."""
    lines = ["Available color schemes:"]
    for i, key in enumerate(SCHEME_KEYS, 1):
        s = SCHEMES[key]
        lines.append(f"  {i}. {key:<13} {s['label']}")
        lines.append(f"       {s['cvd']}")
    lines.append(f"  {len(SCHEME_KEYS) + 1}. custom        custom matplotlib colormap")
    return "\n".join(lines)


def make_custom_scheme(
    density: str = "viridis",
    real: str = "PuOr",
    imag: str = "PuOr",
    signed: str | None = None,
    magnitude: bool = False,
    shell: str | None = None,
    label: str = "Custom scheme",
) -> dict:
    """Build a custom scheme dictionary (validates colormap names)."""
    signed = signed or real
    shell = shell or density
    # Validate early to avoid failures halfway through a plot
    for name in (density, shell, real, imag, signed):
        get_cmap(name)
    return {
        "key": "custom",
        "label": label,
        "short": "custom",
        "description": (
            f"density={density}, shell={shell}, real={real}, imag={imag}, "
            f"signed={signed}, {'magnitude' if magnitude else 'signed'} mode"
        ),
        "cvd": "custom (check color-vision readability yourself)",
        "magnitude": bool(magnitude),
        "density": density,
        "shell": shell,
        "real": real,
        "imag": imag,
        "signed": signed,
        "contour": "white",
        "boundary": "magenta",
    }


def role_cmap_names(scheme: Scheme) -> Iterable[str]:
    """Return all colormap names used by a scheme (for preview/validation)."""
    s = resolve_scheme(scheme)
    return (s["density"], s["shell"], s["real"], s["imag"], s["signed"])


if __name__ == "__main__":
    # Print the scheme table when this file is run directly
    print(scheme_table())
    print()
    for key in SCHEME_KEYS:
        print(describe_scheme(key))
        print()
