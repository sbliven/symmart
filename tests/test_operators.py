#!/usr/bin/env python

"""Tests for `symmart` package."""

import pytest
import sympy as sp

from symmart.operators import MatrixOperator, complete_group
from symmart.wallpaper_groups import wallpaper_generators as wg
from symmart.wallpaper_groups import 𝛾y, 𝜌2, 𝜌4, 𝜌6, 𝜎x, 𝜎x_hex, 𝜏1, 𝜏i


def test_fixed_points():
    half = sp.sympify(1) / 2

    assert (𝜏1 * 𝜌2).fixed_point() == [half, 0]
    assert (𝜏1 * 𝜌2).stable_line() is None
    assert (𝜎x).fixed_point() is None
    assert (𝜎x).stable_line() == ([0, 0], [1, 0], [0, 0])
    assert (𝛾y).fixed_point() is None
    assert (𝛾y).stable_line() == ([0, 0], [0, 1], [0, half])
    assert (𝜏1 * 𝛾y).fixed_point() is None
    assert (𝜏1 * 𝛾y).stable_line() == ([half, 0], [0, 1], [0, half])
    assert (𝜏1).fixed_point() is None
    assert (𝜏1).stable_line() is None
    assert MatrixOperator(sp.eye(3)).fixed_point() is None
    assert MatrixOperator(sp.eye(3)).stable_line() is None


def test_eq():
    assert (𝜏1.inv() * 𝜏1) != sp.eye(3)
    assert MatrixOperator(
        [
            [
                1,
                0,
            ]
        ]
    )

    # glide reflection across x = 1/2 by (0, 1/2)
    # glide reflection across x = 1/2 by (0, -1/2)
    assert MatrixOperator([[-1, 0, 1], [-1, 1, 1], [0, 0, 1]]) != MatrixOperator(
        [[-1, 0, 1], [-1, 1, 0], [0, 0, 1]]
    )


def test_description():
    assert (𝜏1 * 𝜏i).description() == "translation by (1, 1)"
    assert (𝛾y).description() == "glide reflection across x = 0 by (0, 1/2)"
    assert (𝜏1 * 𝜌2).description() == "2-fold rotation around (1/2, 0)"
    assert (𝜏i * 𝜎x).description() == "reflection across y = 1/2"
    oblique = MatrixOperator(sp.Matrix([[12, 16, 14], [16, -12, -3], [0, 0, 20]]) / 20)
    assert (
        oblique.description() == "glide reflection across x - 2⋅y = 1/2 by (1/2, 1/4)"
    )
    assert 𝜌6.description() == "6-fold rotation around (0, 0)"
    assert (𝜎x_hex * 𝜌6).description() == "reflection across x = -y"

    with pytest.raises(ValueError):
        (𝜎x * 𝜌6).description()


def test_in_unit_cell():
    # Translational
    assert 𝜏1.in_unit_cell()
    assert 𝜏i.in_unit_cell()
    assert not (𝜏1 ** -1).in_unit_cell()
    assert not (𝜏1 * 𝜏i).in_unit_cell()

    # identity
    assert not MatrixOperator(sp.eye(3)).in_unit_cell()

    # Point symm
    assert (𝜏1 * 𝜌2).in_unit_cell()  # 2-fold rotation around (1/2, 0)
    assert not (𝜏1 ** 2 * 𝜌2).in_unit_cell()  # 2-fold rotation around (1, 0)
    assert not (𝜌2 * 𝜏1).in_unit_cell()  # around (-1/2,0)

    # Glides
    assert (𝛾y).in_unit_cell()
    assert (𝛾y ** 2).in_unit_cell()  # 𝜏i
    assert not (𝛾y ** 3).in_unit_cell()

    # glide reflection across x - y = 1/2 by (1/2, 1/2)
    assert MatrixOperator([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).in_unit_cell()
    # glide reflection across x - y = -1/2 by (1/2, 1/2)
    assert not MatrixOperator([[0, 1, 0], [1, 0, 1], [0, 0, 1]]).in_unit_cell()

    # glide reflection across x - 2⋅y = 1/2 by (1, 1/2)
    assert MatrixOperator([[1, 0, 1], [1, -1, 0], [0, 0, 1]]).in_unit_cell()
    # glide reflection across x - 2⋅y = -1/2 by (1, 1/2)
    assert not MatrixOperator([[1, 0, 1], [1, -1, 1], [0, 0, 1]]).in_unit_cell()

    # glide reflection across x - 2⋅y = -5/2 by (1, 1/2)
    assert not MatrixOperator([[1, 0, 1], [1, -1, 3], [0, 0, 1]]).in_unit_cell()

    # glide reflection across x = 1/2 by (0, 1/2)
    assert MatrixOperator([[-1, 0, 1], [-1, 1, 1], [0, 0, 1]]).in_unit_cell()
    # glide reflection across x = 1/2 by (0, -1/2)
    assert not MatrixOperator([[-1, 0, 1], [-1, 1, 0], [0, 0, 1]]).in_unit_cell()

    # reflections
    assert 𝜎x.in_unit_cell(), 𝜎x.description()
    # reflection across x - y = -1
    assert not MatrixOperator([[0, 1, -1], [2, -1, 2], [0, 0, 1]]).in_unit_cell()
    # reflection across x - y = 1
    assert not MatrixOperator([[0, 1, 1], [1, 0, -1], [0, 0, 1]]).in_unit_cell()
    # reflection across x = y
    assert MatrixOperator([[0, 1, 0], [1, 0, 0], [0, 0, 1]]).in_unit_cell()
    # glide reflection across y = 0 by (1, 0)
    assert not (𝜎x * 𝜏1).in_unit_cell(), (𝜎x * 𝜏1).description()
    # glide reflection across y = 0 by (2, 0)
    assert not (𝜎x * 𝜏1 * 𝜏1).in_unit_cell(), (𝜎x * 𝜏1 * 𝜏1).description()


def test_wg_in_cell():
    for grp, ops in wg.items():
        for op in ops:
            assert op.in_unit_cell(), f"{grp} operator {op} not in unit cell"


def test_symm_group():
    assert (𝜏1).symm_group() == ("𝜏", None)
    assert (𝜏1 * 𝜏i).symm_group() == ("𝜏", None)
    assert (𝜏1 * 𝜌2).symm_group() == ("𝜌", 2)
    assert (𝜎x).symm_group() == ("𝜎", 2)
    assert (𝜏1 * 𝛾y).symm_group() == ("𝛾", None)


def test_complete_group():
    p2_gen = wg["p2"]
    assert len(p2_gen) == 3
    p2 = complete_group(p2_gen)
    assert set(p2) == {𝜏1, 𝜏i, 𝜌2, 𝜏1 * 𝜌2, 𝜏i * 𝜌2, 𝜏1 * 𝜏i * 𝜌2}

    pmm_gen = wg["pmm"]
    pmm = complete_group(pmm_gen)
    assert len(pmm_gen) == 4
    assert set(pmm) == {*p2, 𝜎x, 𝜎x * 𝜌2, 𝜏i * 𝜎x, 𝜏1 * 𝜎x * 𝜌2}, ",".join(
        [g.name for g in pmm]
    )

    pm_gen = wg["pm"]
    pm = complete_group(pm_gen)
    assert len(pm_gen) == 3
    𝜎y = 𝜌2 * 𝜎x
    assert set(pm) == {𝜏1, 𝜏i, 𝜎y, 𝜏1 * 𝜎y}, ",".join([g.name for g in pm])

    p4m_gen = wg["p4m"]
    p4m = complete_group(p4m_gen)
    assert len(p4m_gen) == 3
    assert set(p4m) == (
        {𝜏1, 𝜏i}
        | {𝜌4, 𝜌2, 𝜌4 ** 3}  # 𝜌4, including powers
        | {𝜏1 * 𝜌4, 𝜏1 * 𝜏i * 𝜌2, 𝜏i * 𝜌4 ** 3}  # 𝜌4
        | {𝜏1 * 𝜌2, 𝜏i * 𝜌2}  # 𝜌2
        | {𝜎x, 𝜏i * 𝜎x, 𝜎x * 𝜌2, 𝜏1 * 𝜎x * 𝜌2}  # 𝜎
        | {𝜎x * 𝜌4, 𝜌4 * 𝜎x, 𝜏1 * 𝜌4 * 𝜎x, 𝜏1 * 𝜎x * 𝜌4}  # 𝛾
    ), ",".join([g.name for g in pm])

    p6m_gen = wg["p6m"]
    p6m = complete_group(p6m_gen)
    assert len(p6m_gen) == 3
    expected = (
        {𝜏1, 𝜏i}
        | {𝜌6 ** n for n in range(1, 6)}  # 𝜌6
        | {𝜏1 * 𝜌6 ** 2, 𝜏1 * 𝜏i * 𝜌6 ** 4, 𝜏i * 𝜌6 ** 4, 𝜏1 * 𝜏i * 𝜌6 ** 2}  # 𝜌3
        | {𝜏1 * 𝜌2, 𝜏i * 𝜌2, 𝜏1 * 𝜏i * 𝜌2}  # 𝜌2
        | {𝜌6 ** n * 𝜎x_hex for n in range(6)}  # 𝜎
        | {
            𝜏1 * 𝜎x_hex * 𝜌6,
            𝜏1 * 𝜌6 * 𝜎x_hex,
            𝜏1 * 𝜏i * 𝜎x_hex,
            𝜏1 * 𝜌6 ** 2 * 𝜎x_hex,
            𝜏1 * 𝜏i * 𝜌2 * 𝜎x_hex,
            𝜏1 * 𝜏i * 𝜎x_hex * 𝜌6 ** 2,
        }
    )
    assert set(p6m) == expected, ",".join([g.name for g in pm])
