"""Regression tests for SCAN-family alpha switching polynomials."""

import ast
from pathlib import Path

import numpy as np

from atom.xc.meta_scan import rSCAN


def test_meta_scan_has_no_discarded_alpha_polynomial_continuations():
    source_path = Path(__file__).parents[1] / "atom" / "xc" / "meta_scan.py"
    source = source_path.read_text()
    tree = ast.parse(source)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        segment = ast.get_source_segment(source, node) or ""
        stripped = segment.lstrip()
        if stripped.startswith("+") and "alpha0To25" in stripped:
            offenders.append((node.lineno, stripped.splitlines()[0]))

    assert offenders == []


def test_rscan_exchange_uses_full_alpha_switching_polynomial():
    rho = np.array([1.0])
    s = np.array([1.0])
    alpha = np.array([1.0])
    zero = np.array([0.0])

    actual_exchange, *_ = rSCAN._rSCAN_exchange(
        rho,
        s,
        alpha,
        zero,
        zero,
        zero,
        zero,
        zero,
    )

    k1 = 0.065
    muak = 10 / 81
    b2 = np.sqrt(5913 / 405000)
    b1 = 511 / 13500 / (2 * b2)
    b3 = 0.5
    b4 = muak * muak / k1 - 1606 / 18225 - b1 * b1

    x = (muak * s**2) * (
        1 + b4 * (s**2) / muak * np.exp(-np.abs(b4) * (s**2) / muak)
    ) + (b1 * s**2 + b2 * (1 - alpha) * np.exp(-b3 * (1 - alpha) ** 2)) ** 2
    hx1 = 1 + k1 - k1 / (1 + x / k1)
    hx0 = 1.174
    fx_full = (
        1
        + (-0.667) * alpha
        + (-0.4445555) * alpha**2
        + (-0.663086601049) * alpha**3
        + 1.451297044490 * alpha**4
        + (-0.887998041597) * alpha**5
        + 0.234528941479 * alpha**6
        + (-0.023185843322) * alpha**7
    )
    gx = 1 - np.exp(-4.9479 * s ** (-0.5))
    expected_exchange = (
        -3 / (4 * np.pi) * (3 * (np.pi**2) * rho) ** (1 / 3)
    ) * ((hx1 + fx_full * (hx0 - hx1)) * gx)

    np.testing.assert_allclose(actual_exchange, expected_exchange, atol=1e-14)
