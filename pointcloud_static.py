#!/usr/bin/env python3
"""
pointcloud_static.py — standalone static 3D point-cloud renderer
=======================================================

This is a **standalone static** visualization extracted from the original
Time_Hydrogen/plot4.py (plot4_3d_realistic), packaged as a single self-contained file:

- does not import this project's core / palettes / plot_common;
- includes hydrogen radial functions, spherical harmonics, and point-cloud cuts;
- uses only numpy + scipy + matplotlib;
- can be run from the command line or imported as `from pointcloud_static import render_pointcloud`.

Purpose
----
Uniform-grid sampling plus density-dependent point size/alpha approximates volume rendering,
producing a diffuse electron cloud. Only a static version exists (the original project
had no time-dependent counterpart), so it is kept as a single static tool.

Examples
----
    python pointcloud_static.py -n 3 -l 2 -m 1 --resolution 60 \
        --scheme colorblind --cut octant --save n3l2m1.png

    python pointcloud_static.py -n 4 -l 0 -m 0 --resolution 80 \
        --cmap inferno --background black --color-by shell --no-show

Common colormaps
--------
inferno, magma, plasma, viridis, cividis, turbo, Greys, RdBu, PuOr,
PiYG, coolwarm, nipy_spectral, ... (any matplotlib colormap name)

Colorblind-friendly suggestions: ``--scheme colorblind`` (cividis) or ``--cmap cividis``;
achromatopsia / black-and-white printing: ``--cmap Greys``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.constants as const
import scipy.special as spe
from matplotlib.colors import PowerNorm
from scipy.special import sph_harm_y


# ===========================================================================
# 1. Physics: hydrogen radial wavefunction (log-space stabilization)
# ===========================================================================
class HydrogenWavefunction:
    """Minimal hydrogen wavefunction calculator (for this single file only)."""

    def __init__(self):
        self.a0 = const.value("Bohr radius")

    def classical_turning_points(self, n, l):
        """r_min/max = n^2 a0 [1 +/- sqrt(1 - l(l+1)/n^2)]."""
        eps_sq = l * (l + 1) / (n * n)
        eps_sq = min(eps_sq, 1.0 - 1e-15)
        root = np.sqrt(1.0 - eps_sq)
        return (
            n * n * self.a0 * (1.0 - root),
            n * n * self.a0 * (1.0 + root),
        )

    def log_factorial(self, n):
        return spe.loggamma(n + 1)

    @staticmethod
    def _log_assoc_laguerre(rho, k, alpha):
        """Log-space recurrence for L_k^(alpha)(rho); returns (sign, log_abs)."""
        rho = np.asarray(rho, dtype=float)
        if k == 0:
            return np.ones_like(rho), np.zeros_like(rho)

        s_prev = np.ones_like(rho)
        l_prev = np.zeros_like(rho)

        v1 = 1.0 + alpha - rho
        s_curr = np.sign(v1)
        l_curr = np.log(np.abs(v1) + 1e-300)
        if k == 1:
            return s_curr, l_curr

        for m in range(1, k):
            coef1 = 2.0 * m + alpha + 1.0 - rho
            coef2 = float(m) + alpha

            log_a = np.log(np.abs(coef1) + 1e-300) + l_curr
            sign_a = np.sign(coef1) * s_curr
            log_b = np.log(coef2 + 1e-300) + l_prev
            sign_b = s_prev

            big_a = log_a >= log_b
            log_big = np.where(big_a, log_a, log_b)
            log_small = np.where(big_a, log_b, log_a)
            eff_sign_big = np.where(big_a, sign_a, -sign_b)
            eff_sign_small = np.where(big_a, -sign_b, sign_a)

            ratio = np.exp(log_small - log_big)
            factor = eff_sign_big + eff_sign_small * ratio
            s_next = np.sign(factor)
            l_next = log_big + np.log(np.abs(factor) + 1e-300) - np.log(m + 1.0)

            s_prev, l_prev = s_curr, l_curr
            s_curr, l_curr = s_next, l_next
        return s_curr, l_curr

    def R_nl(self, r, n, l):
        """Radial wavefunction R_nl(r); supports arrays of any shape."""
        r_arr = np.asarray(r, dtype=float)
        shape = r_arr.shape
        r_flat = r_arr.ravel()
        r_safe = np.where(r_flat < 1e-100 * self.a0, 1e-100 * self.a0, r_flat)

        log_coeff = 0.5 * (
            3.0 * np.log(2.0 / (self.a0 * n))
            + self.log_factorial(n - l - 1)
            - np.log(2.0 * n)
            - self.log_factorial(n + l)
        )
        rho = 2.0 * r_safe / (self.a0 * n)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rho = np.log(np.where(rho > 0.0, rho, 1.0))
        log_exp = -rho / 2.0 + l * log_rho

        k = n - l - 1
        alpha = 2 * l + 1
        with np.errstate(over="ignore", invalid="ignore"):
            lag = spe.assoc_laguerre(rho, k, alpha)
        bad = ~np.isfinite(lag)

        if not np.any(bad):
            sign_l = np.sign(lag)
            log_l = np.log(np.abs(lag) + 1e-300)
        else:
            sign_l = np.sign(np.where(bad, 1.0, lag))
            log_l = np.log(np.abs(np.where(bad, 1.0, lag)) + 1e-300)
            s_bad, l_bad = self._log_assoc_laguerre(rho[bad], k, alpha)
            sign_l[bad] = s_bad
            log_l[bad] = l_bad

        result = sign_l * np.exp(np.clip(log_coeff + log_exp + log_l, -700, 700))
        if l != 0:
            result = np.where(r_flat < 1e-100 * self.a0, 0.0, result)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result.reshape(shape)

    def angular(self, l, m, phi, theta,
                orbital_type="complex", real_component="cos"):
        """Complex or real spherical harmonic (cos/sin type)."""
        if orbital_type == "complex":
            return sph_harm_y(l, m, theta, phi)

        m_abs = abs(m)
        if m_abs == 0:
            return np.real(sph_harm_y(l, 0, theta, phi))
        yp = sph_harm_y(l, m_abs, theta, phi)
        yn = sph_harm_y(l, -m_abs, theta, phi)
        sign = -1.0 if (m_abs % 2) else 1.0
        if real_component == "cos":
            result = (yn + sign * yp) / np.sqrt(2.0)
        else:
            result = 1j * (yn - sign * yp) / np.sqrt(2.0)
        result = np.real(result)
        return np.where(np.abs(result) < 1e-10, 0.0, result)


# ===========================================================================
# 2. Point-cloud cuts
# ===========================================================================
def get_cut_mask(points, mode="none", which_parts=None):
    """3D point-cloud cut; mode can be none/half_x/half_y/half_z/quadrant/octant/hexadecant."""
    points = np.asarray(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if mode == "none":
        return np.ones(len(points), dtype=bool)
    if mode == "half_x":
        return x >= 0
    if mode == "half_y":
        return y >= 0
    if mode == "half_z":
        return z >= 0
    if mode == "quadrant":
        quad = np.zeros(len(points), dtype=int)
        quad[(x >= 0) & (y >= 0)] = 0
        quad[(x < 0) & (y >= 0)] = 1
        quad[(x < 0) & (y < 0)] = 2
        quad[(x >= 0) & (y < 0)] = 3
        keep = which_parts if which_parts is not None else [0, 1, 2, 3]
        return np.isin(quad, keep)
    if mode == "octant":
        octant = (
            ((x >= 0).astype(int) << 2)
            + ((y >= 0).astype(int) << 1)
            + (z >= 0).astype(int)
        )
        keep = which_parts if which_parts is not None else list(range(8))
        return np.isin(octant, keep)
    if mode == "hexadecant":
        phi = np.arctan2(y, x)
        norms = np.linalg.norm(points, axis=1)
        theta = np.arccos(np.clip(z / (norms + 1e-100), -1.0, 1.0))
        phi_bin = ((phi + np.pi) // (np.pi / 2)).astype(int) % 4
        theta_bin = np.clip((theta // (np.pi / 4)).astype(int), 0, 3)
        idx = phi_bin * 4 + theta_bin
        keep = which_parts if which_parts is not None else list(range(16))
        return np.isin(idx, keep)
    return np.ones(len(points), dtype=bool)


# ===========================================================================
# 3. Colormap resolution
# ===========================================================================
SCHEME_ALIASES = {
    "classic": "inferno",
    "signed": "turbo",
    "colorblind": "cividis",
    "cividis": "cividis",
    "mono": "Greys",
    "gray": "Greys",
    "grey": "Greys",
    "turbo": "turbo",
    "viridis": "viridis",
    "inferno": "inferno",
    "magma": "magma",
    "plasma": "plasma",
}


def resolve_cmap(name):
    """Resolve a scheme alias or matplotlib colormap name to a Colormap."""
    import matplotlib

    key = str(name)
    key = SCHEME_ALIASES.get(key.lower(), key)
    try:
        return matplotlib.colormaps[key]
    except KeyError:
        lower = {k.lower(): k for k in matplotlib.colormaps}
        if key.lower() in lower:
            return matplotlib.colormaps[lower[key.lower()]]
        raise SystemExit(
            f"Unknown colormap: {name!r}. Try inferno / cividis / viridis / "
            f"turbo / Greys / PuOr / PiYG."
        )


# ===========================================================================
# 4. Rendering
# ===========================================================================
def render_pointcloud(
    n=3,
    l=2,
    m=1,
    resolution=60,
    scale=None,
    cmap="inferno",
    cut_mode="none",
    which_parts=None,
    background="black",
    zoom=1.0,
    point_size_scale=1.5,
    alpha_scale=1.2,
    color_by="shell",
    orbital_type="complex",
    real_component="cos",
    jitter=0.35,
    seed=0,
    show_labels=True,
    save=None,
    dpi=300,
    show=True,
    verbose=True,
):
    """
    Static 3D point-cloud rendering.

    Parameters
    ----
    resolution : grid points per dimension (50-80 recommended; larger is slower)
    cmap       : any matplotlib colormap name or scheme alias
    cut_mode   : none / half_x / half_y / half_z / quadrant / octant / hexadecant
    color_by   : 'shell' (4*pi*r^2*|psi|^2, default) or 'density' (|psi|^2)
    save       : PNG output path; None means do not save
    show       : whether to call plt.show()
    """
    import matplotlib.pyplot as plt

    t0 = time.time()
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if not (0 <= l < n):
        raise ValueError(f"l must satisfy 0 <= l < {n}")
    if not (-l <= m <= l):
        raise ValueError(f"m must satisfy -{l} <= m <= {l}")
    if resolution < 10:
        raise ValueError("resolution must be at least 10")

    calc = HydrogenWavefunction()
    a0 = calc.a0
    _, r_max = calc.classical_turning_points(n, l)
    if scale is None:
        scale = 1.8 * r_max

    if verbose:
        print(
            f"[point cloud] n={n}, l={l}, m={m}, resolution={resolution}^3, "
            f"color_by={color_by}, cut={cut_mode}"
        )

    # ---- uniform grid + in-sphere cropping ----
    x = np.linspace(-scale, scale, resolution)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    r_points = np.linalg.norm(points, axis=1)
    sphere = r_points <= scale
    points = points[sphere]
    n_points = len(points)
    if n_points == 0:
        raise RuntimeError("No sample points inside the sphere; increase scale or resolution")
    if verbose:
        print(
            f"[point cloud] {n_points} points inside the sphere "
            f"({100.0 * n_points / resolution**3:.1f}% of cube)"
        )

    # ---- small jitter to break up moire patterns from the uniform grid ----
    if jitter and jitter > 0:
        cell = 2.0 * scale / max(resolution - 1, 1)
        rng = np.random.default_rng(seed)
        points = points + rng.uniform(
            -jitter, jitter, size=points.shape
        ) * cell

    # ---- wavefunction ----
    r = np.linalg.norm(points, axis=1)
    theta = np.arccos(np.clip(points[:, 2] / (r + 1e-100), -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])
    radial = calc.R_nl(r, n, l)
    angular = calc.angular(
        l, m, phi, theta,
        orbital_type=orbital_type, real_component=real_component,
    )
    psi = radial * angular
    dens = np.abs(psi) ** 2
    max_dens = float(np.max(dens)) if dens.size else 0.0
    if max_dens <= 0:
        raise RuntimeError("Wavefunction is zero everywhere; check the quantum numbers")
    dens_norm = dens / max_dens

    # ---- shell weight ----
    r_safe = np.where(r < 1e-100 * a0, 1e-100 * a0, r)
    shell = 4.0 * np.pi * r_safe**2 * dens_norm
    max_shell = float(np.max(shell))
    shell_norm = shell / max_shell if max_shell > 0 else shell

    # ---- cuts ----
    mask = get_cut_mask(points, cut_mode, which_parts)
    points = points[mask]
    dens_norm = dens_norm[mask]
    shell_norm = shell_norm[mask]
    if len(points) == 0:
        raise RuntimeError("No points remain after the cut; choose another cut_mode")

    color_values = shell_norm if color_by == "shell" else dens_norm

    # ---- point size / alpha ----
    sizes = (1.0 + 9.0 * shell_norm) * float(point_size_scale)
    alpha = 0.2 + 0.75 * dens_norm
    alpha = np.clip(alpha * float(alpha_scale), 0.02, 1.0)

    # ---- plotting ----
    # Physics uses meters; plot in Bohr radii to avoid 1e-9 axis scales
    points_plot = points / a0
    scale_plot = scale / a0
    bg = "black" if str(background).lower() == "black" else "white"
    fg = "white" if bg == "black" else "black"
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor(bg)
    ax = fig.add_subplot(111, projection="3d", facecolor=bg)

    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    scatter = ax.scatter(
        points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
        c=color_values, cmap=resolve_cmap(cmap), norm=norm,
        s=sizes, alpha=alpha, linewidth=0, rasterized=True,
    )

    ax.set_box_aspect([1, 1, 1])
    lim = scale_plot * float(zoom)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    if show_labels:
        ax.set_title(
            f"Diffuse Point-Cloud of Hydrogen — n={n}, l={l}, m={m}\n"
            f"resolution={resolution}^3, color_by={color_by}, cut={cut_mode} "
            f"(coordinates in Bohr radii a0)",
            color=fg, fontsize=13,
        )
        ax.set_xlabel("X (a0)", color=fg)
        ax.set_ylabel("Y (a0)", color=fg)
        ax.set_zlabel("Z (a0)", color=fg)
    ax.tick_params(colors=fg, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0, 0, 0, 1) if bg == "black" else (1, 1, 1, 1))
        axis.pane.set_edgecolor(fg)
        axis.line.set_color(fg)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10, pad=0.08)
    cbar.set_label(
        r"$4\pi r^2|\psi|^2$ (norm)" if color_by == "shell"
        else r"$|\psi|^2$ (norm)"
    )
    cbar.ax.tick_params(colors=fg)

    fig.tight_layout()
    if verbose:
        print(f"[point cloud] rendering finished in {time.time() - t0:.2f} s.")

    if save:
        folder = os.path.dirname(os.path.abspath(save))
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        if verbose:
            print(f"[point cloud] saved: {save}")

    if show:
        plt.show()
    return fig


# ===========================================================================
# 5. Command line
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Standalone static 3D point-cloud renderer (hydrogen electron cloud). "
            "Includes its own physics; no other project files are required."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-n", "--n", type=int, default=3, help="principal quantum number n")
    p.add_argument("-l", "--l", type=int, default=2, help="angular quantum number l")
    p.add_argument("-m", "--m", type=int, default=1, help="magnetic quantum number m")
    p.add_argument("--resolution", type=int, default=60, help="samples per dimension")
    p.add_argument("--scale", type=float, default=None,
                   help="display half-width (m); default is 1.8 x classical turning point")
    p.add_argument("--cmap", default="inferno",
                   help="matplotlib colormap name or scheme alias (classic/colorblind/mono/...)")
    p.add_argument("--scheme", default=None,
                   help="scheme alias (classic/signed/colorblind/mono/turbo) "
                        "or any matplotlib colormap name; overrides --cmap")
    p.add_argument("--cut", default="none",
                   choices=["none", "half_x", "half_y", "half_z",
                            "quadrant", "octant", "hexadecant"],
                   help="point-cloud cut mode")
    p.add_argument("--background", default="black",
                   choices=["black", "white"], help="background color")
    p.add_argument("--zoom", type=float, default=1.0, help="zoom factor")
    p.add_argument("--point-size", type=float, default=1.5, help="point-size scale")
    p.add_argument("--alpha", type=float, default=1.2, help="alpha scale")
    p.add_argument("--color-by", default="shell",
                   choices=["shell", "density"],
                   help="color by shell density or probability density")
    p.add_argument("--jitter", type=float, default=0.35,
                   help="point-position jitter in grid steps; 0 disables")
    p.add_argument("--seed", type=int, default=0, help="jitter random seed")
    p.add_argument("--orbital", default="complex",
                   choices=["complex", "real"], help="complex or real orbital")
    p.add_argument("--real-component", default="cos",
                   choices=["cos", "sin"], help="cos/sin type for a real orbital")
    p.add_argument("--save", default=None, help="PNG output path")
    p.add_argument("--dpi", type=int, default=300, help="output DPI")
    p.add_argument("--no-show", action="store_true", help="save only; do not open a window")
    p.add_argument("--no-labels", action="store_true", help="hide titles and axis labels")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cmap = args.scheme or args.cmap
    try:
        render_pointcloud(
            n=args.n, l=args.l, m=args.m,
            resolution=args.resolution, scale=args.scale,
            cmap=cmap, cut_mode=args.cut,
            background=args.background, zoom=args.zoom,
            point_size_scale=args.point_size, alpha_scale=args.alpha,
            color_by=args.color_by,
            orbital_type=args.orbital, real_component=args.real_component,
            jitter=args.jitter, seed=args.seed,
            show_labels=not args.no_labels,
            save=args.save, dpi=args.dpi,
            show=not args.no_show,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
