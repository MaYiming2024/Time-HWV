#!/usr/bin/env python3
"""
selftest.py — Time_Hydrogen_v2 smoke test
=========================================

In headless (Agg) mode, run the six plot types plus the standalone point cloud and check:
- every color scheme resolves to a matplotlib colormap;
- static figures save to PNG;
- animations save to GIF;
- Plotly figures save to HTML;
- the standalone point-cloud file saves to PNG.

Run:
    python selftest.py
Output directory:
    _selftest_out/
"""

from __future__ import annotations

import os
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from core import HydrogenWavefunction, build_plane_data, build_volume_data
from palettes import (
    SCHEME_KEYS,
    cmap_for,
    describe_scheme,
    get_cmap,
    plotly_scale,
    resolve_scheme,
)
import plot1
import plot2
import plot3
import plot4
import plot5
import plot6
import pointcloud_static


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_selftest_out")
os.makedirs(OUT, exist_ok=True)


def check(condition, message):
    if condition:
        print(f"  [PASS] {message}")
        return True
    print(f"  [FAIL] {message}")
    return False


def run(name, func):
    print(f"\n=== {name} ===")
    t0 = time.time()
    try:
        result = func()
        dt = time.time() - t0
        print(f"  [OK] {name} in {dt:.2f} s")
        return True, result
    except Exception:
        print(f"  [ERROR] {name}")
        traceback.print_exc()
        return False, None


def main():
    ok = True
    calc = HydrogenWavefunction()
    n, l, m = 3, 2, 1
    scale = calc.scale_for(n, l)
    print(f"Bohr a0 = {calc.a0:.6e} m")
    print(f"n={n}, l={l}, m={m}, scale = {scale:.6e} m")

    # ------------------------------------------------------------------
    # 1. Color-scheme integrity
    # ------------------------------------------------------------------
    def test_palettes():
        for key in SCHEME_KEYS:
            s = resolve_scheme(key)
            assert s["label"]
            for role in ("density", "shell", "real", "imag", "signed"):
                get_cmap(s[role])
            scale_list = plotly_scale(cmap_for(key, "density"))
            assert len(scale_list) >= 2
            assert all(0.0 <= pos <= 1.0 for pos, _ in scale_list)
            print("  -", describe_scheme(key).splitlines()[0])
        return True

    good, _ = run("color-scheme resolution", test_palettes)
    ok &= good

    # ------------------------------------------------------------------
    # 2. Physics / data construction
    # ------------------------------------------------------------------
    def test_data():
        plane = build_plane_data(calc, n, l, m, "xz", scale, grid_size=120)
        assert plane["psi"].shape == (120, 120)
        assert abs(np.max(plane["density"]) - 1.0) < 1e-9
        X, Y, Z, r, mask, psi, density, shell = build_volume_data(
            calc, n, l, m, scale, resolution=24)
        assert X.shape == (24, 24, 24)
        assert psi.dtype == complex
        assert density.max() <= 1.0 + 1e-12
        assert shell.max() <= 1.0 + 1e-12
        # Real orbital
        plane_real = build_plane_data(
            calc, n, l, 0, "xy", scale, grid_size=80,
            orbital_type="real", real_component="cos")
        assert np.max(np.abs(np.imag(plane_real["psi"]))) < 1e-10
        # Nodal-plane detection: the XY plane for n=4,l=2,m=1 is nodal; XZ is not
        nodal = build_plane_data(
            calc, 4, 2, 1, "xy", calc.scale_for(4, 2), grid_size=100)
        assert nodal["nodal_plane"] is True
        assert np.max(nodal["density"]) == 0.0
        assert nodal["relative_amplitude"] < 1e-9
        non_nodal = build_plane_data(
            calc, 4, 2, 1, "xz", calc.scale_for(4, 2), grid_size=100)
        assert non_nodal["nodal_plane"] is False
        assert abs(np.max(non_nodal["density"]) - 1.0) < 1e-9
        # The XY plane for l=1,m=1 is not nodal
        not_nodal_xy = build_plane_data(
            calc, 2, 1, 1, "xy", calc.scale_for(2, 1), grid_size=100)
        assert not_nodal_xy["nodal_plane"] is False
        return True

    good, _ = run("wavefunction and grid data", test_data)
    ok &= good

    # ------------------------------------------------------------------
    # 3. Static 2D
    # ------------------------------------------------------------------
    for idx, func, scheme, plane in [
        (1, plot1.plot1_density_2d, "classic", "xz"),
        (2, plot2.plot2_shell_2d, "cividis", "xz"),
    ]:
        def make(func=func, scheme=scheme, plane=plane, idx=idx):
            fig = func(
                calc, n, l, m, plane=plane, scale=scale, grid_size=180,
                scheme=scheme,
            )
            path = os.path.join(OUT, f"plot{idx}_static_{scheme}.png")
            fig.savefig(path, dpi=90, bbox_inches="tight")
            plt.close(fig)
            assert os.path.getsize(path) > 1000
            return path
        good, path = run(f"plot{idx} static 2D", make)
        ok &= good
        if path:
            print("  output:", path)

    def make_nodal_2d():
        fig = plot1.plot1_density_2d(
            calc, 4, 2, 1, plane="xy", scale=calc.scale_for(4, 2),
            grid_size=120, scheme="classic",
        )
        path = os.path.join(OUT, "plot1_nodal_xy.png")
        fig.savefig(path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        assert os.path.getsize(path) > 1000
        return path

    good, path = run("plot1 nodal XY", make_nodal_2d)
    ok &= good
    if path:
        print("  output:", path)

    # ------------------------------------------------------------------
    # 4. Static 3D surface
    # ------------------------------------------------------------------
    for idx, func, scheme in [
        (3, plot3.plot3_surface_density, "signed"),
        (4, plot4.plot4_surface_shell, "deuteranopia"),
    ]:
        def make(func=func, scheme=scheme, idx=idx):
            fig = func(
                calc, n, l, m, plane="xz", scale=scale, grid_size=140,
                scheme=scheme,
            )
            path = os.path.join(OUT, f"plot{idx}_static_{scheme}.png")
            fig.savefig(path, dpi=90, bbox_inches="tight")
            plt.close(fig)
            assert os.path.getsize(path) > 1000
            return path
        good, path = run(f"plot{idx} static 3D surface", make)
        ok &= good
        if path:
            print("  output:", path)

    # ------------------------------------------------------------------
    # 5. Dynamic 2D (GIF)
    # ------------------------------------------------------------------
    for idx, func, scheme in [
        (1, plot1.plot1_density_dynamic, "classic"),
        (2, plot2.plot2_shell_dynamic, "cividis"),
    ]:
        def make(func=func, scheme=scheme, idx=idx):
            ani, fig = func(
                calc, n, l, m, plane="xz", scale=scale, grid_size=120,
                scheme=scheme, dt=2.0, frames=4, interval=80,
            )
            path = os.path.join(OUT, f"plot{idx}_dynamic_{scheme}.gif")
            ani.save(path, writer="pillow", fps=12)
            plt.close(fig)
            assert os.path.getsize(path) > 1000
            return path
        good, path = run(f"plot{idx} dynamic 2D", make)
        ok &= good
        if path:
            print("  output:", path)

    # ------------------------------------------------------------------
    # 5b. Verify that the animation actually changes (compare GIF frames 0 and 1)
    # ------------------------------------------------------------------
    def test_gif_changes():
        path = os.path.join(OUT, "plot1_dynamic_classic.gif")
        assert os.path.exists(path)
        from PIL import Image
        im = Image.open(path)
        assert getattr(im, "n_frames", 1) >= 2
        im.seek(0)
        a = np.asarray(im.convert("L"), dtype=float)
        im.seek(1)
        b = np.asarray(im.convert("L"), dtype=float)
        diff = float(np.mean(np.abs(a - b)))
        print(f"  - mean grayscale difference frame0/frame1 = {diff:.3f}")
        assert diff > 0.05
        return True

    good, _ = run("animation frame-change check", test_gif_changes)
    ok &= good

    # ------------------------------------------------------------------
    # 6. Dynamic 3D surface (GIF, few frames)
    # ------------------------------------------------------------------
    for idx, func, scheme in [
        (3, plot3.plot3_surface_density_dynamic, "signed"),
        (4, plot4.plot4_surface_shell_dynamic, "tritanopia"),
    ]:
        def make(func=func, scheme=scheme, idx=idx):
            ani, fig = func(
                calc, n, l, m, plane="xz", scale=scale, grid_size=110,
                scheme=scheme, dt=1.0, frames=3, interval=80,
                res_factor=0.2,
            )
            path = os.path.join(OUT, f"plot{idx}_dynamic_{scheme}.gif")
            ani.save(path, writer="pillow", fps=10)
            plt.close(fig)
            assert os.path.getsize(path) > 1000
            return path
        good, path = run(f"plot{idx} dynamic 3D surface", make)
        ok &= good
        if path:
            print("  output:", path)

    # ------------------------------------------------------------------
    # 7. Plotly volume rendering
    # ------------------------------------------------------------------
    def make_plot5_static():
        fig = plot5.plot5_volume_density(
            calc, n, l, m, scale, resolution=26, scheme="cividis",
            surface_count=26, smooth_sigma=1.0,
        )
        path = os.path.join(OUT, "plot5_static.html")
        fig.write_html(path)
        assert os.path.getsize(path) > 1000
        return path

    good, path = run("plot5 static volume", make_plot5_static)
    ok &= good
    if path:
        print("  output:", path)

    def make_plot6_static():
        fig = plot6.plot6_volume_shell(
            calc, n, l, m, scale, resolution=26, scheme="magma",
            surface_count=26, smooth_sigma=1.0,
        )
        path = os.path.join(OUT, "plot6_static.html")
        fig.write_html(path)
        assert os.path.getsize(path) > 1000
        return path

    good, path = run("plot6 static volume", make_plot6_static)
    ok &= good
    if path:
        print("  output:", path)

    def make_plot5_dynamic():
        fig = plot5.plot5_volume_density_dynamic(
            calc, n, l, m, scale, resolution=22, dt=0.5, frames=4,
            mode="both", scheme="signed", surface_count=22,
        )
        path = os.path.join(OUT, "plot5_dynamic.html")
        fig.write_html(path)
        assert os.path.getsize(path) > 1000
        return path

    good, path = run("plot5 dynamic volume", make_plot5_dynamic)
    ok &= good
    if path:
        print("  output:", path)

    def make_plot6_dynamic():
        fig = plot6.plot6_volume_shell_dynamic(
            calc, n, l, m, scale, resolution=22, dt=0.5, frames=4,
            mode="both", scheme="deuteranopia", surface_count=22,
        )
        path = os.path.join(OUT, "plot6_dynamic.html")
        fig.write_html(path)
        assert os.path.getsize(path) > 1000
        return path

    good, path = run("plot6 dynamic volume", make_plot6_dynamic)
    ok &= good
    if path:
        print("  output:", path)

    # ------------------------------------------------------------------
    # 8. Standalone point-cloud file
    # ------------------------------------------------------------------
    def make_pointcloud():
        path = os.path.join(OUT, "pointcloud.png")
        fig = pointcloud_static.render_pointcloud(
            n=n, l=l, m=m, resolution=32, cmap="cividis",
            cut_mode="octant", background="black",
            save=path, show=False, dpi=80, verbose=False,
        )
        plt.close(fig)
        assert os.path.getsize(path) > 1000
        return path

    good, path = run("standalone static 3D point cloud", make_pointcloud)
    ok &= good
    if path:
        print("  output:", path)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All checks passed" if ok else "Some checks failed; see [FAIL]/[ERROR] above")
    print("Output directory:", OUT)
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
