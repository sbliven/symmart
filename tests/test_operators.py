#!/usr/bin/env python

"""Tests for `symmart` package."""

import sympy as sp

from symmart.operators import MatrixOperator

half = sp.sympify(1) / 2

# fundamental operators
𝜏1 = sp.Matrix([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
𝜏i = sp.Matrix([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
𝜌2 = sp.Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
𝜎x = sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
𝛾y = sp.Matrix([[-1, 0, 0], [0, 1, half], [0, 0, 1]])


def test_fixed_points():
    assert MatrixOperator(𝜏1 * 𝜌2).fixed_point() == [half, 0]
    assert MatrixOperator(𝜏1 * 𝜌2).stable_line() is None
    assert MatrixOperator(𝜎x).fixed_point() is None
    assert MatrixOperator(𝜎x).stable_line() == ([0, 0], [1, 0], [0, 0])
    assert MatrixOperator(𝛾y).fixed_point() is None
    assert MatrixOperator(𝛾y).stable_line() == ([0, 0], [0, 1], [0, half])
    assert MatrixOperator(𝜏1 * 𝛾y).fixed_point() is None
    assert MatrixOperator(𝜏1 * 𝛾y).stable_line() == ([half, 0], [0, 1], [0, half])
    assert MatrixOperator(𝜏1).fixed_point() is None
    assert MatrixOperator(𝜏1).stable_line() is None
    assert MatrixOperator(sp.eye(3)).fixed_point() is None
    assert MatrixOperator(sp.eye(3)).stable_line() is None
