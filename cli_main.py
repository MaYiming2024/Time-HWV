"""
cli_main.py — Time_Hydrogen_v2 command-line interface
==========================================

Six plot types plus one standalone static point-cloud tool. Every plot asks for a color scheme,
and a global default is available for batch work.

Plot matrix (New ICS model)
----------------------------
                 probability |psi|^2      shell density 4*pi*r^2*|psi|^2
    2D cross-section plot1                    plot2
    3D surface (height field)  plot3           plot4
    3D volume rendering        plot5 (Plotly)  plot6 (Plotly)

    standalone     pointcloud_static.py (static 3D point cloud)

Run
----
    python cli_main.py                 # interactive
    python cli_main.py --plot 1 --scheme cividis --save out.png
    python cli_main.py --plot 5 --mode d --component both --no-show

Plot functions do not show/save automatically; this file handles save -> show -> close.
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt

from core import HydrogenWavefunction
from palettes import (
    SCHEMES,
    SCHEME_KEYS,
    describe_scheme,
    make_custom_scheme,
    resolve_scheme,
    scheme_table,
)
import plot1
import plot2
import plot3
import plot4
import plot5
import plot6
import pointcloud_static

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["Times New Roman", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ===========================================================================
# Input helpers
# ===========================================================================
def ask_int(prompt, default=None, minimum=None, maximum=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return int(default)
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter an integer.")
            continue
        if minimum is not None and value < minimum:
            print(f"  Must be >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"  Must be <= {maximum}.")
            continue
        return value


def ask_float(prompt, default=None, minimum=None, maximum=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if minimum is not None and value < minimum:
            print(f"  Must be >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"  Must be <= {maximum}.")
            continue
        return value


def ask_yes_no(prompt, default=False):
    hint = "y" if default else "n"
    while True:
        raw = input(f"{prompt} (y/n, default {hint}): ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")


def ask_choice(prompt, choices, default):
    choices = [str(c) for c in choices]
    while True:
        raw = input(f"{prompt} ({'/'.join(choices)}, default {default}): ").strip().lower()
        if raw == "":
            return default
        if raw in choices:
            return raw
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        print(f"  Choices: {', '.join(choices)}")


# ===========================================================================
# Color schemes
# ===========================================================================
def ask_custom_scheme():
    print("\nCustom colors: enter matplotlib colormap names (viridis / cividis / RdBu / PuOr / Greys ...)")
    while True:
        density = input("  Density colormap [viridis]: ").strip() or "viridis"
        real = input("  Real-part colormap [PuOr]: ").strip() or "PuOr"
        imag = input("  Imaginary-part colormap [PuOr]: ").strip() or "PuOr"
        signed = input(f"  Diverging colormap [{real}]: ").strip() or real
        magnitude = ask_yes_no("  Use magnitude mode for Re/Im (|Re|, |Im|)?", False)
        try:
            scheme = make_custom_scheme(
                density=density, real=real, imag=imag, signed=signed,
                magnitude=magnitude,
            )
            print("  Custom scheme enabled.")
            return scheme
        except KeyError as exc:
            print(f"  {exc}")


def ask_scheme(default="classic"):
    default_key = resolve_scheme(default)["key"]
    if isinstance(default, dict):
        default_key = "custom"
    keys = ", ".join(SCHEME_KEYS)
    print(f"\nColor schemes: {keys}")
    while True:
        raw = input(
            f"Choose a scheme [default {default_key}] (Enter keeps it, ? shows help, "
            f"custom for a custom scheme): "
        ).strip().lower()
        if raw == "":
            return default
        if raw in ("?", "h", "help"):
            print()
            print(scheme_table())
            for key in SCHEME_KEYS:
                print()
                print(describe_scheme(key))
            print()
            continue
        if raw == "custom":
            return ask_custom_scheme()
        if raw in SCHEMES:
            print("  Selected: " + describe_scheme(raw).replace("\n", "\n  "))
            return raw
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(SCHEME_KEYS):
                return SCHEME_KEYS[idx]
        print("  Unknown scheme; try again.")


def resolve_cli_scheme(value):
    """Resolve --scheme to a scheme name or a custom scheme dictionary."""
    if value in SCHEMES:
        return value
    # Common aliases for the point-cloud tool
    aliases = {
        "colorblind": "cividis",
        "gray": "Greys",
        "grey": "Greys",
    }
    cmap_name = aliases.get(str(value).lower(), value)
    try:
        return make_custom_scheme(
            density=cmap_name, shell=cmap_name,
            real="PuOr", imag="PuOr", signed="PuOr",
            label=f"custom:{cmap_name}",
        )
    except KeyError as exc:
        raise SystemExit(f"Invalid --scheme: {exc}") from exc


# ===========================================================================
# Quantum numbers
# ===========================================================================
def input_quantum_numbers():
    print("\n--- Quantum numbers and grid ---")
    n = ask_int("Principal quantum number n (positive integer): ", 3, minimum=1)
    l = ask_int(f"Angular quantum number l (0 <= l < {n}): ", min(2, n - 1), minimum=0, maximum=n - 1)
    m = ask_int(f"Magnetic quantum number m (-{l} <= m <= {l}): ", 0, minimum=-l, maximum=l)
    grid = ask_int("2D grid size (100-2000): ", 400, minimum=100, maximum=2000)
    orbital = ask_choice("Orbital type: complex / real", ["complex", "real"], "complex")
    real_component = "cos"
    if orbital == "real":
        real_component = ask_choice("Real-orbital component: cos / sin", ["cos", "sin"], "cos")

    calc = HydrogenWavefunction()
    scale = calc.scale_for(n, l)
    r_min, r_max = calc.classical_turning_points(n, l)
    print(
        f"\nSet: n={n}, l={l}, m={m}, grid={grid}, "
        f"orbital={orbital}"
        + (f"/{real_component}" if orbital == "real" else "")
    )
    print(
        f"Classical turning points: r_min={r_min / calc.a0:.2f} a0, "
        f"r_max={r_max / calc.a0:.2f} a0; display half-width={scale / calc.a0:.2f} a0"
    )
    return {
        "n": n, "l": l, "m": m, "grid": grid,
        "orbital": orbital, "real_component": real_component,
        "calc": calc, "scale": scale,
    }


# ===========================================================================
# Save / show
# ===========================================================================
def save_figure(fig, default_name, no_show=False):
    if ask_yes_no("Save the image?", False):
        fname = input(f"  File name (default {default_name}): ").strip() or default_name
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"  Saved: {fname}")
    if not no_show:
        plt.show()
    plt.close(fig)


def save_animation(ani, fig, default_name, interval, no_show=False):
    if ask_yes_no("Save the GIF?", False):
        fname = input(f"  File name (default {default_name}): ").strip() or default_name
        ani.save(fname, writer="pillow", fps=max(1.0, 1000.0 / interval))
        print(f"  Saved: {fname}")
    if not no_show:
        plt.show()
    plt.close(fig)


def save_plotly(fig, default_name, no_show=False):
    if ask_yes_no("Save as HTML?", False):
        fname = input(f"  File name (default {default_name}): ").strip() or default_name
        fig.write_html(fname)
        print(f"  Saved: {fname}")
    if not no_show:
        fig.show()


# ===========================================================================
# Menu
# ===========================================================================
MENU = """
==================== Time-Dependent Hydrogen v2 ====================
  1  2D cross-section - probability |psi|^2
  2  2D cross-section - shell density 4*pi*r^2*|psi|^2
  3  3D surface (height field) - probability |psi|^2
  4  3D surface (height field) - shell density 4*pi*r^2*|psi|^2
  5  3D volume rendering - probability |psi|^2        (Plotly)
  6  3D volume rendering - shell density 4*pi*r^2*|psi|^2    (Plotly)
  7  Standalone static 3D point cloud
  s  Set the default color scheme
  r  Re-enter quantum numbers
  q  Quit
============================================================
"""


def plot_default_name(choice, mode, n, l, m, kind):
    suffix = "dynamic.gif" if mode == "d" else "png"
    return f"plot{choice}_{kind}_n{n}_l{l}_m{m}.{suffix}"


# ===========================================================================
# Plot branches
# ===========================================================================
def run_plot_1(state, default_scheme, no_show):
    mode = ask_choice("Mode: s=static / d=dynamic", ["s", "d"], "s")
    plane = ask_choice("Plane: xy / xz / yz", ["xy", "xz", "yz"], "xz")
    scheme = ask_scheme(default_scheme)
    bg = "white" if mode == "s" else ask_choice(
        "Background: black / white", ["black", "white"], "black")
    three = False
    if mode == "d":
        three = ask_yes_no("Use the 3x3 three-plane view?", False)

    if mode == "s":
        fig = plot1.plot1_density_2d(
            state["calc"], state["n"], state["l"], state["m"],
            plane=plane, scale=state["scale"], grid_size=state["grid"],
            scheme=scheme, background=bg,
            orbital_type=state["orbital"],
            real_component=state["real_component"],
        )
        save_figure(fig, plot_default_name(1, "s", state["n"], state["l"], state["m"], "2d_density"),
                    no_show)
    else:
        dt = ask_float("Time step dt (default 2.0): ", 2.0)
        frames = ask_int("Frames (default 160): ", 160, minimum=2, maximum=2000)
        interval = ask_int("Frame interval ms (default 40): ", 40, minimum=10, maximum=1000)
        if three:
            ani, fig = plot1.plot1_density_dynamic_three_planes(
                state["calc"], state["n"], state["l"], state["m"],
                scale=state["scale"], grid_size=state["grid"],
                scheme=scheme, dt=dt, frames=frames, interval=interval,
                background=bg, orbital_type=state["orbital"],
                real_component=state["real_component"],
            )
        else:
            ani, fig = plot1.plot1_density_dynamic(
                state["calc"], state["n"], state["l"], state["m"],
                plane=plane, scale=state["scale"], grid_size=state["grid"],
                scheme=scheme, dt=dt, frames=frames, interval=interval,
                background=bg, orbital_type=state["orbital"],
                real_component=state["real_component"],
            )
        save_animation(
            ani, fig,
            plot_default_name(
                1, "d", state["n"], state["l"], state["m"],
                "2d_density_3planes" if three else "2d_density",
            ),
            interval, no_show,
        )


def run_plot_2(state, default_scheme, no_show):
    mode = ask_choice("Mode: s=static / d=dynamic", ["s", "d"], "s")
    plane = ask_choice("Plane: xy / xz / yz", ["xy", "xz", "yz"], "xz")
    scheme = ask_scheme(default_scheme)
    bg = "white" if mode == "s" else ask_choice(
        "Background: black / white", ["black", "white"], "black")

    if mode == "s":
        fig = plot2.plot2_shell_2d(
            state["calc"], state["n"], state["l"], state["m"],
            plane=plane, scale=state["scale"], grid_size=state["grid"],
            scheme=scheme, background=bg,
            orbital_type=state["orbital"],
            real_component=state["real_component"],
        )
        save_figure(fig, plot_default_name(2, "s", state["n"], state["l"], state["m"], "2d_shell"),
                    no_show)
    else:
        dt = ask_float("Time step dt (default 2.0): ", 2.0)
        frames = ask_int("Frames (default 160): ", 160, minimum=2, maximum=2000)
        interval = ask_int("Frame interval ms (default 40): ", 40, minimum=10, maximum=1000)
        ani, fig = plot2.plot2_shell_dynamic(
            state["calc"], state["n"], state["l"], state["m"],
            plane=plane, scale=state["scale"], grid_size=state["grid"],
            scheme=scheme, dt=dt, frames=frames, interval=interval,
            background=bg, orbital_type=state["orbital"],
            real_component=state["real_component"],
        )
        save_animation(
            ani, fig,
            plot_default_name(2, "d", state["n"], state["l"], state["m"], "2d_shell"),
            interval, no_show,
        )


def run_plot_3(state, default_scheme, no_show):
    mode = ask_choice("Mode: s=static / d=dynamic", ["s", "d"], "s")
    plane = ask_choice("Plane: xy / xz / yz", ["xy", "xz", "yz"], "xz")
    scheme = ask_scheme(default_scheme)
    bg = "white" if mode == "s" else ask_choice(
        "Background: black / white", ["black", "white"], "black")

    if mode == "s":
        fig = plot3.plot3_surface_density(
            state["calc"], state["n"], state["l"], state["m"],
            plane=plane, scale=state["scale"], grid_size=state["grid"],
            scheme=scheme, background=bg,
            orbital_type=state["orbital"],
            real_component=state["real_component"],
        )
        save_figure(fig, plot_default_name(3, "s", state["n"], state["l"], state["m"], "3d_surface_density"),
                    no_show)
    else:
        dt = ask_float("Time step dt (default 1.0): ", 1.0)
        frames = ask_int("Frames (default 80): ", 80, minimum=2, maximum=1000)
        interval = ask_int("Frame interval ms (default 60): ", 60, minimum=10, maximum=1000)
        res_factor = ask_float("Sampling factor 0.05-1.0 (larger is finer, default 0.2): ", 0.2,
                               minimum=0.02, maximum=1.0)
        ani, fig = plot3.plot3_surface_density_dynamic(
            state["calc"], state["n"], state["l"], state["m"],
            plane=plane, scale=state["scale"], grid_size=state["grid"],
            scheme=scheme, dt=dt, frames=frames, interval=interval,
            background=bg, res_factor=res_factor,
            orbital_type=state["orbital"],
            real_component=state["real_component"],
        )
        save_animation(
            ani, fig,
            plot_default_name(3, "d", state["n"], state["l"], state["m"], "3d_surface_density"),
            interval, no_show,
        )


def run_plot_4(state, default_scheme, no_show):
    mode = ask_choice("Mode: s=static / d=dynamic", ["s", "d"], "s")
    plane = ask_choice("Plane: xy / xz / yz", ["xy", "xz", "yz"], "xz")
    scheme = ask_scheme(default_scheme)
    bg = "white" if mode == "s" else ask_choice(
        "Background: black / white", ["black", "white"], "black")

    if mode == "s":
        fig = plot4.plot4_surface_shell(
            state["calc"], state["n"], state["l"], state["m"],
            plane=plane, scale=state["scale"], grid_size=state["grid"],
            scheme=scheme, background=bg,
            orbital_type=state["orbital"],
            real_component=state["real_component"],
        )
        save_figure(fig, plot_default_name(4, "s", state["n"], state["l"], state["m"], "3d_surface_shell"),
                    no_show)
    else:
        dt = ask_float("Time step dt (default 1.0): ", 1.0)
        frames = ask_int("Frames (default 80): ", 80, minimum=2, maximum=1000)
        interval = ask_int("Frame interval ms (default 60): ", 60, minimum=10, maximum=1000)
        res_factor = ask_float("Sampling factor 0.05-1.0 (larger is finer, default 0.2): ", 0.2,
                               minimum=0.02, maximum=1.0)
        ani, fig = plot4.plot4_surface_shell_dynamic(
            state["calc"], state["n"], state["l"], state["m"],
            plane=plane, scale=state["scale"], grid_size=state["grid"],
            scheme=scheme, dt=dt, frames=frames, interval=interval,
            background=bg, res_factor=res_factor,
            orbital_type=state["orbital"],
            real_component=state["real_component"],
        )
        save_animation(
            ani, fig,
            plot_default_name(4, "d", state["n"], state["l"], state["m"], "3d_surface_shell"),
            interval, no_show,
        )


def _volume_common_prompts():
    resolution = ask_int("Volume resolution (40-70 recommended, default 50): ", 50,
                         minimum=20, maximum=120)
    background = ask_choice("Background: black / white", ["black", "white"], "black")
    smooth = ask_float("Gaussian sigma (default 1.0): ", 1.0, minimum=0.0, maximum=5.0)
    surface_count = ask_int("Isosurfaces (default=resolution): ", resolution,
                            minimum=5, maximum=120)
    opacity = ask_float("Base opacity (default 0.10): ", 0.10,
                        minimum=0.01, maximum=1.0)
    return resolution, background, smooth, surface_count, opacity


def run_plot_5(state, default_scheme, no_show):
    mode = ask_choice("Mode: s=static / d=dynamic slider", ["s", "d"], "s")
    scheme = ask_scheme(default_scheme)
    resolution, background, smooth, surface_count, opacity = _volume_common_prompts()

    if mode == "s":
        fig = plot5.plot5_volume_density(
            state["calc"], state["n"], state["l"], state["m"],
            state["scale"], resolution=resolution, scheme=scheme,
            background=background, smooth_sigma=smooth,
            surface_count=surface_count, opacity=opacity,
            orbital_type=state["orbital"],
            real_component=state["real_component"],
        )
        save_plotly(fig, plot_default_name(5, "s", state["n"], state["l"], state["m"], "volume_density"),
                    no_show)
    else:
        component = ask_choice("Component: real / imag / both", ["real", "imag", "both"], "both")
        dt = ask_float("Time step dt (default 0.5): ", 0.5)
        frames = ask_int("Frames (default 40): ", 40, minimum=2, maximum=500)
        fig = plot5.plot5_volume_density_dynamic(
            state["calc"], state["n"], state["l"], state["m"],
            state["scale"], resolution=resolution, dt=dt, frames=frames,
            mode=component, scheme=scheme, background=background,
            smooth_sigma=smooth, surface_count=surface_count,
            opacity=opacity, orbital_type=state["orbital"],
            real_component=state["real_component"],
        )
        save_plotly(fig, plot_default_name(5, "d", state["n"], state["l"], state["m"], "volume_density"),
                    no_show)


def run_plot_6(state, default_scheme, no_show):
    mode = ask_choice("Mode: s=static / d=dynamic slider", ["s", "d"], "s")
    scheme = ask_scheme(default_scheme)
    resolution, background, smooth, surface_count, opacity = _volume_common_prompts()

    if mode == "s":
        fig = plot6.plot6_volume_shell(
            state["calc"], state["n"], state["l"], state["m"],
            state["scale"], resolution=resolution, scheme=scheme,
            background=background, smooth_sigma=smooth,
            surface_count=surface_count, opacity=opacity,
            orbital_type=state["orbital"],
            real_component=state["real_component"],
        )
        save_plotly(fig, plot_default_name(6, "s", state["n"], state["l"], state["m"], "volume_shell"),
                    no_show)
    else:
        component = ask_choice("Component: real / imag / both", ["real", "imag", "both"], "both")
        dt = ask_float("Time step dt (default 0.5): ", 0.5)
        frames = ask_int("Frames (default 40): ", 40, minimum=2, maximum=500)
        fig = plot6.plot6_volume_shell_dynamic(
            state["calc"], state["n"], state["l"], state["m"],
            state["scale"], resolution=resolution, dt=dt, frames=frames,
            mode=component, scheme=scheme, background=background,
            smooth_sigma=smooth, surface_count=surface_count,
            opacity=opacity, orbital_type=state["orbital"],
            real_component=state["real_component"],
        )
        save_plotly(fig, plot_default_name(6, "d", state["n"], state["l"], state["m"], "volume_shell"),
                    no_show)


def run_plot_7(state, default_scheme, no_show):
    print("\nStandalone static 3D point cloud (pointcloud_static.py)")
    scheme = ask_scheme(default_scheme)
    scheme_d = resolve_scheme(scheme)
    # The point cloud uses one colormap; prefer the scheme's shell / density map
    cmap = scheme_d["shell"] if scheme_d["shell"] != scheme_d["density"] else scheme_d["density"]
    if scheme_d["key"] == "classic":
        cmap = "inferno"
    if scheme_d["key"] == "signed":
        cmap = "turbo"
    resolution = ask_int("Point-cloud resolution (50-80 recommended, default 60): ", 60,
                         minimum=20, maximum=120)
    cut = ask_choice(
        "Cut: none/half_x/half_y/half_z/quadrant/octant/hexadecant",
        ["none", "half_x", "half_y", "half_z", "quadrant", "octant", "hexadecant"],
        "none",
    )
    background = ask_choice("Background: black / white", ["black", "white"], "black")
    color_by = ask_choice("Color by: shell=shell density / density=probability density",
                          ["shell", "density"], "shell")
    save = None
    if ask_yes_no("Save PNG?", True):
        default = f"pointcloud_n{state['n']}_l{state['l']}_m{state['m']}.png"
        save = input(f"  File name (default {default}): ").strip() or default
    pointcloud_static.render_pointcloud(
        n=state["n"], l=state["l"], m=state["m"],
        resolution=resolution, cmap=cmap, cut_mode=cut,
        background=background, color_by=color_by,
        orbital_type=state["orbital"],
        real_component=state["real_component"],
        save=save, show=not no_show,
    )


# ===========================================================================
# Interactive main loop
# ===========================================================================
def interactive(no_show=False, default_scheme="classic"):
    print("=== Time-Dependent Hydrogen Wavefunction Visualizer v2 ===")
    print("Hint: each plot asks for a scheme separately; enter ? to list all schemes.")
    state = input_quantum_numbers()
    default_scheme = resolve_cli_scheme(default_scheme)
    print("\nCurrent default scheme: " + describe_scheme(default_scheme).replace("\n", "\n  "))

    handlers = {
        "1": run_plot_1, "2": run_plot_2, "3": run_plot_3,
        "4": run_plot_4, "5": run_plot_5, "6": run_plot_6,
        "7": run_plot_7,
    }

    while True:
        print(MENU)
        choice = input("Select: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Exiting.")
            return
        if choice in ("r", "reset"):
            state = input_quantum_numbers()
            continue
        if choice in ("s", "scheme"):
            default_scheme = ask_scheme(default_scheme)
            print("Default scheme updated: " + describe_scheme(default_scheme).replace("\n", "\n  "))
            continue
        if choice in handlers:
            try:
                handlers[choice](state, default_scheme, no_show)
            except KeyboardInterrupt:
                print("\nCurrent plot cancelled.")
                plt.close("all")
            except Exception as exc:
                print(f"\n[error] Plot generation failed: {exc}")
                plt.close("all")
            continue
        print("Invalid option; try again.")


# ===========================================================================
# Non-interactive quick mode (for scripts / tests)
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="Time-Dependent Hydrogen v2 (interactive or --plot quick mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--plot", type=int, choices=list(range(1, 8)),
                   help="non-interactive mode: generate the specified plot")
    p.add_argument("--mode", choices=["s", "d"], default="s", help="static/dynamic")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--l", type=int, default=2)
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--grid", type=int, default=400, help="2D grid size")
    p.add_argument("--plane", choices=["xy", "xz", "yz"], default="xz")
    p.add_argument("--three-planes", action="store_true",
                   help="plot1 dynamic: 3x3 three-plane view (XY/XZ/YZ)")
    p.add_argument("--scheme", default="classic",
                   help="scheme name or any matplotlib colormap name")
    p.add_argument("--orbital", choices=["complex", "real"], default="complex")
    p.add_argument("--real-component", choices=["cos", "sin"], default="cos")
    p.add_argument("--background", choices=["black", "white"], default=None)
    p.add_argument("--resolution", type=int, default=50, help="3D/volume resolution")
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--res-factor", type=float, default=0.2)
    p.add_argument("--smooth", type=float, default=1.0)
    p.add_argument("--surface-count", type=int, default=None)
    p.add_argument("--opacity", type=float, default=0.1)
    p.add_argument("--component", choices=["real", "imag", "both"], default="both")
    p.add_argument("--cut", default="none",
                   choices=["none", "half_x", "half_y", "half_z",
                            "quadrant", "octant", "hexadecant"])
    p.add_argument("--color-by", choices=["shell", "density"], default="shell")
    p.add_argument("--save", default=None,
                   help="output file; PNG/GIF/HTML depends on the plot type")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    return p


def _write_plotly(fig, out):
    """Save a Plotly figure; PNG/PDF needs kaleido and falls back to HTML on failure."""
    if out.lower().endswith(".html"):
        fig.write_html(out)
        return
    try:
        fig.write_image(out)
    except Exception as exc:
        fallback = out + ".html"
        print(f"[hint] Export to {out} failed (kaleido may be missing): {exc}")
        print(f"       Saved HTML instead: {fallback}")
        fig.write_html(fallback)


def quick_mode(args):
    if not (args.n > 0 and 0 <= args.l < args.n and -args.l <= args.m <= args.l):
        raise SystemExit("Invalid quantum numbers.")
    calc = HydrogenWavefunction()
    state = {
        "n": args.n, "l": args.l, "m": args.m, "grid": args.grid,
        "orbital": args.orbital, "real_component": args.real_component,
        "calc": calc, "scale": calc.scale_for(args.n, args.l),
    }
    scheme = resolve_cli_scheme(args.scheme)
    scheme_d = resolve_scheme(scheme)
    bg = args.background or ("white" if args.mode == "s" else "black")
    out = args.save

    if args.plot == 1:
        if args.mode == "s":
            fig = plot1.plot1_density_2d(
                calc, args.n, args.l, args.m, plane=args.plane,
                scale=state["scale"], grid_size=args.grid, scheme=scheme,
                background=bg, orbital_type=args.orbital,
                real_component=args.real_component,
            )
            if out:
                fig.savefig(out, dpi=300, bbox_inches="tight")
            if not args.no_show:
                plt.show()
        else:
            if args.three_planes:
                ani, fig = plot1.plot1_density_dynamic_three_planes(
                    calc, args.n, args.l, args.m,
                    scale=state["scale"], grid_size=args.grid, scheme=scheme,
                    dt=args.dt, frames=args.frames, interval=args.interval,
                    background=bg, orbital_type=args.orbital,
                    real_component=args.real_component,
                )
            else:
                ani, fig = plot1.plot1_density_dynamic(
                    calc, args.n, args.l, args.m, plane=args.plane,
                    scale=state["scale"], grid_size=args.grid, scheme=scheme,
                    dt=args.dt, frames=args.frames, interval=args.interval,
                    background=bg, orbital_type=args.orbital,
                    real_component=args.real_component,
                )
            if out:
                ani.save(out, writer="pillow",
                         fps=max(1.0, 1000.0 / args.interval))
            if not args.no_show:
                plt.show()

    elif args.plot == 2:
        if args.mode == "s":
            fig = plot2.plot2_shell_2d(
                calc, args.n, args.l, args.m, plane=args.plane,
                scale=state["scale"], grid_size=args.grid, scheme=scheme,
                background=bg, orbital_type=args.orbital,
                real_component=args.real_component,
            )
            if out:
                fig.savefig(out, dpi=300, bbox_inches="tight")
            if not args.no_show:
                plt.show()
        else:
            ani, fig = plot2.plot2_shell_dynamic(
                calc, args.n, args.l, args.m, plane=args.plane,
                scale=state["scale"], grid_size=args.grid, scheme=scheme,
                dt=args.dt, frames=args.frames, interval=args.interval,
                background=bg, orbital_type=args.orbital,
                real_component=args.real_component,
            )
            if out:
                ani.save(out, writer="pillow",
                         fps=max(1.0, 1000.0 / args.interval))
            if not args.no_show:
                plt.show()

    elif args.plot == 3:
        if args.mode == "s":
            fig = plot3.plot3_surface_density(
                calc, args.n, args.l, args.m, plane=args.plane,
                scale=state["scale"], grid_size=args.grid, scheme=scheme,
                background=bg, orbital_type=args.orbital,
                real_component=args.real_component,
            )
            if out:
                fig.savefig(out, dpi=300, bbox_inches="tight")
            if not args.no_show:
                plt.show()
        else:
            ani, fig = plot3.plot3_surface_density_dynamic(
                calc, args.n, args.l, args.m, plane=args.plane,
                scale=state["scale"], grid_size=args.grid, scheme=scheme,
                dt=args.dt, frames=args.frames, interval=args.interval,
                background=bg, res_factor=args.res_factor,
                orbital_type=args.orbital,
                real_component=args.real_component,
            )
            if out:
                ani.save(out, writer="pillow",
                         fps=max(1.0, 1000.0 / args.interval))
            if not args.no_show:
                plt.show()

    elif args.plot == 4:
        if args.mode == "s":
            fig = plot4.plot4_surface_shell(
                calc, args.n, args.l, args.m, plane=args.plane,
                scale=state["scale"], grid_size=args.grid, scheme=scheme,
                background=bg, orbital_type=args.orbital,
                real_component=args.real_component,
            )
            if out:
                fig.savefig(out, dpi=300, bbox_inches="tight")
            if not args.no_show:
                plt.show()
        else:
            ani, fig = plot4.plot4_surface_shell_dynamic(
                calc, args.n, args.l, args.m, plane=args.plane,
                scale=state["scale"], grid_size=args.grid, scheme=scheme,
                dt=args.dt, frames=args.frames, interval=args.interval,
                background=bg, res_factor=args.res_factor,
                orbital_type=args.orbital,
                real_component=args.real_component,
            )
            if out:
                ani.save(out, writer="pillow",
                         fps=max(1.0, 1000.0 / args.interval))
            if not args.no_show:
                plt.show()

    elif args.plot == 5:
        if args.mode == "s":
            fig = plot5.plot5_volume_density(
                calc, args.n, args.l, args.m, state["scale"],
                resolution=args.resolution, scheme=scheme,
                background=bg, smooth_sigma=args.smooth,
                surface_count=args.surface_count, opacity=args.opacity,
                orbital_type=args.orbital,
                real_component=args.real_component,
            )
        else:
            fig = plot5.plot5_volume_density_dynamic(
                calc, args.n, args.l, args.m, state["scale"],
                resolution=args.resolution, dt=args.dt, frames=args.frames,
                mode=args.component, scheme=scheme, background=bg,
                smooth_sigma=args.smooth,
                surface_count=args.surface_count, opacity=args.opacity,
                orbital_type=args.orbital,
                real_component=args.real_component,
            )
        if out:
            _write_plotly(fig, out)
        if not args.no_show:
            fig.show()

    elif args.plot == 6:
        if args.mode == "s":
            fig = plot6.plot6_volume_shell(
                calc, args.n, args.l, args.m, state["scale"],
                resolution=args.resolution, scheme=scheme,
                background=bg, smooth_sigma=args.smooth,
                surface_count=args.surface_count, opacity=args.opacity,
                orbital_type=args.orbital,
                real_component=args.real_component,
            )
        else:
            fig = plot6.plot6_volume_shell_dynamic(
                calc, args.n, args.l, args.m, state["scale"],
                resolution=args.resolution, dt=args.dt, frames=args.frames,
                mode=args.component, scheme=scheme, background=bg,
                smooth_sigma=args.smooth,
                surface_count=args.surface_count, opacity=args.opacity,
                orbital_type=args.orbital,
                real_component=args.real_component,
            )
        if out:
            _write_plotly(fig, out)
        if not args.no_show:
            fig.show()

    elif args.plot == 7:
        cmap = scheme_d["shell"]
        if scheme_d["key"] == "classic":
            cmap = "inferno"
        if scheme_d["key"] == "signed":
            cmap = "turbo"
        pointcloud_static.render_pointcloud(
            n=args.n, l=args.l, m=args.m, resolution=args.resolution,
            cmap=cmap, cut_mode=args.cut,
            background=(args.background or "black"),
            color_by=args.color_by, orbital_type=args.orbital,
            real_component=args.real_component,
            save=out, show=not args.no_show,
        )


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.plot is not None:
        quick_mode(args)
    else:
        interactive(no_show=args.no_show, default_scheme=args.scheme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
