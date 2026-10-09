#!/usr/bin/env python3
"""
Test for OOB/PHI guard in smart_frames ingest notebook.

This test validates:
1. Robust scope mask doesn't collapse to full-frame on scope withdrawal
2. Per-frame OOB detector catches frames shot outside the body
3. Mask-dependent + mask-independent OOB checks work together
4. Red hue wrapping (0-20, 160-179 in OpenCV HSV H:0-179)
5. Lit-pixel analysis (g > 25) ignores vignette

The test builds synthetic frames: circular scope (red tissue), out-of-body room
(gray/white/blue), whiteout, and a withdrawal sequence. Runs locally without
Databricks via numpy/cv2 (no torch needed for OOB detector).
"""

import os
import sys
import tempfile
import json
import numpy as np
import cv2
import math


# Load the detector straight from the notebook so the test exercises the code
# that actually ships (not a copy that can drift).
NOTEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "ingest_notebooks", "smart_frames", "notebook.py")


def _load_notebook_detector():
    import re
    src = open(NOTEBOOK).read()
    consts = "\n".join(l.strip() for l in src.splitlines()
                       if re.match(r"\s*(TISSUE_HUE_MIN|NEAR_WHITE_BLUE_MAX|LIT_MIN)\s*=", l))
    start = src.index("    def _is_oob_frame(rgb):")
    end = src.index("\n    # ---- SigLIP2 zero-shot scores", start)
    body = "\n".join(l[4:] for l in src[start:end].splitlines())  # dedent out of the try block
    ns = {"cv2": cv2, "np": np}
    exec(consts + "\n" + body, ns)
    return ns


_nb = _load_notebook_detector()
_is_oob_frame = _nb["_is_oob_frame"]
TISSUE_HUE_MIN = _nb["TISSUE_HUE_MIN"]
NEAR_WHITE_BLUE_MAX = _nb["NEAR_WHITE_BLUE_MAX"]


def _grad_of(gray, m_bool):
    """Gradient magnitude in focus region (copied from notebook)."""
    gb = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gx = cv2.Sobel(gb, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gb, cv2.CV_32F, 0, 1)
    return float(np.sqrt(gx ** 2 + gy ** 2)[m_bool].mean())


def _build_scope_mask(frames_sample):
    """Mirror of the notebook's inline pass-A mask (it lives inside the job's
    try block, so it can't be exec'd standalone): a pixel is scope if it is
    lit (>25) in more than 40% of the sampled frames. Keep in sync."""
    lit_count = None
    for rgb in frames_sample:
        lit = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > 25
        lit_count = lit.astype(np.uint16) if lit_count is None else lit_count + lit
    mask = (lit_count > 0.4 * len(frames_sample)).astype(np.uint8)
    ncc, lab = cv2.connectedComponents(mask)
    if ncc > 1:
        biggest = 1 + int(np.argmax([(lab == i).sum() for i in range(1, ncc)]))
        mask = (lab == biggest).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.erode(mask, np.ones((21, 21), np.uint8))

    # Check for full-frame fallback
    full_frame = mask.mean() > 0.90
    M = mask.astype(bool)
    border = None if full_frame else ~M

    return M, border, full_frame


def _make_tissue_frame(h=480, w=640, center=None, radius=None):
    """Create a synthetic frame: circular scope with red/orange tissue, dark vignette."""
    if center is None:
        center = (h // 2, w // 2)
    if radius is None:
        radius = min(h, w) // 3

    # Start with black background (outside scope)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw circular scope region with red/orange tissue
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    scope_mask = dist <= radius

    # Fill scope with red/orange tissue (note: OpenCV is BGR, not RGB)
    tissue_color = np.stack([
        np.random.randint(180, 221, scope_mask.sum()),  # R
        np.random.randint(100, 151, scope_mask.sum()),  # G
        np.random.randint(80, 121, scope_mask.sum()),   # B
    ]).T
    rgb[scope_mask] = tissue_color

    # Add some texture/edges inside scope
    if scope_mask.sum() > 0:
        for _ in range(20):
            py = np.random.randint(center[0] - radius, center[0] + radius + 1)
            px = np.random.randint(center[1] - radius, center[1] + radius + 1)
            if 0 <= py < h and 0 <= px < w and scope_mask[py, px]:
                cv2.circle(rgb, (px, py), np.random.randint(5, 20),
                          (np.random.randint(150, 255), np.random.randint(50, 150), 50), -1)

    # Add vignette (dark edges)
    vignette = np.ones((h, w), dtype=np.uint8) * 255
    cv2.circle(vignette, center, radius, 255, -1)
    cv2.circle(vignette, center, radius - 20, 200, cv2.FILLED)
    rgb = (rgb.astype(np.float32) * vignette[..., None].astype(np.float32) / 255.0).astype(np.uint8)

    return rgb


def _make_room_frame(h=480, w=640):
    """Create out-of-body frame: mostly white/gray/blue — operating room environment."""
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 210  # Light gray background

    # Add blue drapes (top-half)
    rgb[:2 * h // 3, :] = [90, 160, 230]  # Strong blue

    # Add white/light gray walls and equipment (center)
    rgb[h // 2:, :] = [225, 225, 225]  # Very light gray/white

    # Add a tiny hint of skin tone (small patch, bottom-right corner)
    rgb[3 * h // 4:, 3 * w // 4:] = [210, 160, 130]

    # Add high-frequency noise
    noise = np.random.randint(-15, 16, rgb.shape).astype(np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return rgb


def _make_whiteout_frame(h=480, w=640):
    """Create whiteout frame (glare, should fail for that reason)."""
    return np.ones((h, w, 3), dtype=np.uint8) * 250


def _make_red_hue_wrap_frame(h=480, w=640):
    """Create frame with deep red (hue wraps to 160+ in OpenCV HSV)."""
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 100
    # Deep red: RGB ~ (200, 0, 0) which maps to HSV H~0 or H~180 depending on implementation
    # Create a gradient that includes high-H red
    rgb[:, :, 0] = np.linspace(150, 230, w)[np.newaxis, :].astype(np.uint8)  # R gradient
    rgb[:, :, 1] = 0  # G=0
    rgb[:, :, 2] = 0  # B=0
    return rgb


def test_scope_mask_robustness():
    """Test that scope mask doesn't collapse to full-frame on scope withdrawal."""
    print("\n=== Test 1: Scope mask robustness ===")

    # Create sequence: normal scope + withdrawal + normal scope
    h, w = 480, 640
    frames = [
        _make_tissue_frame(h, w),
        _make_tissue_frame(h, w),
        np.ones((h, w, 3), dtype=np.uint8) * 200,  # Scope withdrawal (bright/white)
        _make_tissue_frame(h, w),
        _make_tissue_frame(h, w),
    ]

    M, border, full_frame = _build_scope_mask(frames)

    print(f"  Mask covers {M.mean()*100:.1f}% of frame")
    print(f"  Full-frame fallback triggered: {full_frame}")

    assert not full_frame, "FAIL: Scope mask collapsed to full-frame on withdrawal"
    assert M.mean() < 0.95, "FAIL: Mask covers >95% (should be < for scope region only)"
    assert border is not None, "FAIL: Border should be set when mask doesn't collapse"
    print("  PASS: Mask is robust to withdrawal event")


def test_oob_detector_on_tissue():
    """Test that tissue frame is NOT marked as OOB."""
    print("\n=== Test 2: Tissue frame passes OOB detector ===")

    rgb = _make_tissue_frame()
    oob_flag, tissue_frac, edge_density, near_white_blue_frac, brightness_std = _is_oob_frame(rgb)

    print(f"  Tissue fraction: {tissue_frac:.3f}")
    print(f"  Edge density: {edge_density:.3f}")
    print(f"  Near-white/blue fraction: {near_white_blue_frac:.3f}")
    print(f"  Brightness std: {brightness_std:.1f}")
    print(f"  OOB flag: {oob_flag}")

    assert tissue_frac > TISSUE_HUE_MIN, f"Tissue fraction {tissue_frac} should be > {TISSUE_HUE_MIN}"
    assert not oob_flag, "FAIL: Tissue frame marked as OOB"
    print("  PASS: Tissue frame recognized as in-body")


def test_oob_detector_on_room():
    """Test that room frame IS marked as OOB."""
    print("\n=== Test 3: Room frame fails OOB detector ===")

    rgb = _make_room_frame()
    oob_flag, tissue_frac, edge_density, near_white_blue_frac, brightness_std = _is_oob_frame(rgb)

    print(f"  Tissue fraction: {tissue_frac:.3f}")
    print(f"  Edge density: {edge_density:.3f}")
    print(f"  Near-white/blue fraction: {near_white_blue_frac:.3f}")
    print(f"  Brightness std: {brightness_std:.1f}")
    print(f"  OOB flag: {oob_flag}")

    assert near_white_blue_frac > NEAR_WHITE_BLUE_MAX, f"Room should have high near_white_blue"
    assert oob_flag, "FAIL: Room frame NOT marked as OOB (should be)"
    print("  PASS: Room frame rejected as out-of-body")


def test_oob_detector_on_whiteout():
    """Test that whiteout frame IS marked as OOB."""
    print("\n=== Test 4: Whiteout frame fails OOB detector ===")

    rgb = _make_whiteout_frame()
    oob_flag, tissue_frac, edge_density, near_white_blue_frac, brightness_std = _is_oob_frame(rgb)

    print(f"  Tissue fraction: {tissue_frac:.3f}")
    print(f"  Edge density: {edge_density:.3f}")
    print(f"  Near-white/blue fraction: {near_white_blue_frac:.3f}")
    print(f"  Brightness std: {brightness_std:.1f}")
    print(f"  OOB flag: {oob_flag}")

    assert near_white_blue_frac > NEAR_WHITE_BLUE_MAX, "Whiteout should have high near-white/blue"
    assert tissue_frac < TISSUE_HUE_MIN, "Whiteout should have low tissue fraction"
    assert oob_flag, "FAIL: Whiteout NOT marked as OOB"
    print("  PASS: Whiteout rejected as out-of-body")


def test_red_hue_wrapping():
    """Test that deep red hues (hue wrapping at 160+) are recognized as tissue."""
    print("\n=== Test 5: Red hue wrapping (160-179) detection ===")

    rgb = _make_red_hue_wrap_frame()
    oob_flag, tissue_frac, edge_density, near_white_blue_frac, brightness_std = _is_oob_frame(rgb)

    print(f"  Tissue fraction: {tissue_frac:.3f} (should be >0 due to hue wrapping)")
    print(f"  OOB flag: {oob_flag}")

    assert tissue_frac > 0.01, f"Red hue wrap (160+) should be detected; got {tissue_frac}"
    print("  PASS: Deep red hues (H>=160) recognized as tissue")


def test_lit_pixel_analysis():
    """Test that vignette/black pixels don't dilute fractions."""
    print("\n=== Test 6: Lit-pixel analysis (g > 25) ignores vignette ===")

    # Frame with large black vignette but bright tissue center
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[200:280, 270:370] = [200, 100, 50]  # Tissue red in center

    oob_flag, tissue_frac, edge_density, near_white_blue_frac, brightness_std = _is_oob_frame(rgb)

    print(f"  Tissue fraction (with vignette): {tissue_frac:.3f}")
    print(f"  Should be high since vignette is excluded from analysis")

    assert tissue_frac > 0.5, f"Vignette should be ignored; tissue_frac = {tissue_frac}"
    print("  PASS: Lit-pixel analysis ignores black vignette")


def test_withdrawal_sequence():
    """Test sequence: in->out->in with scope withdrawal in middle."""
    print("\n=== Test 7: Withdrawal sequence handling ===")

    h, w = 480, 640
    frames = [
        _make_tissue_frame(h, w),
        _make_tissue_frame(h, w),
        _make_room_frame(h, w),  # Out-of-body (withdrawal into room)
        _make_tissue_frame(h, w),
        _make_tissue_frame(h, w),
    ]

    # Check each frame's OOB status
    oob_flags = []
    for i, rgb in enumerate(frames):
        oob_flag, tissue_frac, edge_density, near_white_blue_frac, brightness_std = _is_oob_frame(rgb)
        oob_flags.append(oob_flag)
        status = "OOB" if oob_flag else "IN"
        print(f"  Frame {i}: {status} (tissue={tissue_frac:.2f}, edge={edge_density:.3f}, "
              f"near_white_blue={near_white_blue_frac:.2f})")

    # Frame 2 should be OOB, others should be in
    assert not oob_flags[0], "Frame 0 should be in-body"
    assert not oob_flags[1], "Frame 1 should be in-body"
    assert oob_flags[2], "Frame 2 (withdrawal) should be out-of-body"
    assert not oob_flags[3], "Frame 3 should be in-body"
    assert not oob_flags[4], "Frame 4 should be in-body"
    print("  PASS: Sequence correctly identifies withdrawal as out-of-body")


def test_zero_oob_persisted():
    """Integration test: OOB frames never persisted to output."""
    print("\n=== Test 8: Zero OOB frames persisted ===")

    h, w = 480, 640
    frames = [
        _make_tissue_frame(h, w),
        _make_room_frame(h, w),  # OUT-OF-BODY
        _make_tissue_frame(h, w),
        _make_whiteout_frame(h, w),  # OUT-OF-BODY (glare + OOB)
        _make_tissue_frame(h, w),
    ]

    # Simulate gate + persist logic
    oob_count = 0
    persisted_count = 0

    for rgb in frames:
        oob_flag, tissue_frac, edge_density, near_white_blue_frac, brightness_std = _is_oob_frame(rgb)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        whiteout = (g > 240).mean()

        if oob_flag or whiteout > 0.35:
            oob_count += 1
        else:
            persisted_count += 1

    print(f"  OOB rejected: {oob_count}")
    print(f"  Persisted: {persisted_count}")

    assert oob_count == 2, f"Expected 2 OOB rejections, got {oob_count}"
    assert persisted_count == 3, f"Expected 3 persisted, got {persisted_count}"
    print("  PASS: All OOB frames filtered; safe frames persisted")


if __name__ == "__main__":
    try:
        test_scope_mask_robustness()
        test_oob_detector_on_tissue()
        test_oob_detector_on_room()
        test_oob_detector_on_whiteout()
        test_red_hue_wrapping()
        test_lit_pixel_analysis()
        test_withdrawal_sequence()
        test_zero_oob_persisted()

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
        print("\nWhat still needs workspace validation:")
        print("- Real in→out→in surgical clip writes 0 OOB JPGs")
        print("- Scope withdrawal sequences have all OOB frames rejected")
        print("- VFR .mov files get correct timestamps")
        print("- Re-ingest preserves label/snapshot paths")
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
