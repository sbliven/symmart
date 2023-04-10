import sympy as sp

from .operators import MatrixOperator, complete_group

_half = sp.sympify(1) / 2

# Basic operations
𝜏1 = MatrixOperator([[1, 0, 1], [0, 1, 0], [0, 0, 1]], "𝜏1")
𝜏i = MatrixOperator([[1, 0, 0], [0, 1, 1], [0, 0, 1]], "𝜏i")
# 𝜏𝜔 = MatrixOperator([[1, 0, _half], [0, 1, _half], [0, 0, 1]], "𝜏𝜔")
𝜌2 = MatrixOperator([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], "𝜌2")
𝜎x = MatrixOperator([[1, 0, 0], [0, -1, 0], [0, 0, 1]], "𝜎x")  # rectangular lattices
𝜎y = MatrixOperator(𝜌2 * 𝜎x, "𝜎y")
𝛾y = MatrixOperator([[-1, 0, 0], [0, 1, _half], [0, 0, 1]], "𝛾y")
𝜌4 = MatrixOperator([[0, -1, 0], [1, 0, 0], [0, 0, 1]], "𝜌4")
𝜎c = MatrixOperator([[0, 1, 0], [1, 0, 0], [0, 0, 1]], "𝜎c")
𝜎s = MatrixOperator([[0, 1, _half], [1, 0, -_half], [0, 0, 1]], "𝜎s")
𝛾q = MatrixOperator([[-1, 0, _half], [0, 1, _half], [0, 0, 1]], "𝛾q")
# hex operations
𝜌3 = MatrixOperator([[0, -1, 0], [1, -1, 0], [0, 0, 1]], "𝜌3")
𝜎x_hex = MatrixOperator([[1, -1, 0], [0, -1, 0], [0, 0, 1]], "𝜎x")
𝜎y_hex = MatrixOperator([[-1, 1, 0], [0, 1, 0], [0, 0, 1]], "𝜎y")
𝜌6 = MatrixOperator(𝜌3 ** 2 * 𝜌2, "𝜌6")


wallpaper_lattices = {
    "monoclinic": ["p1", "p2"],
    "rhombic": ["cm", "cmm"],
    "rectangular": ["pm", "pg", "pmm", "pmg", "pgg"],
    "square": ["p4", "p4m", "p4g"],
    "hexagonal": ["p3", "p31m", "p3m1", "p6", "p6m"],
}

wallpaper_generators = {
    # General
    "p1": [𝜏1, 𝜏i],
    "p2": [𝜏1, 𝜏i, 𝜌2],
    # Rhombic
    "cm": [𝜏1, 𝜎c],
    "cmm": [𝜏1, 𝜎c, 𝜌2],
    # "cm": [𝜏𝜔, 𝜎x],
    # "cmm": [𝜏𝜔, 𝜎x, 𝜌2],
    # Rectangular
    "pm": [𝜏1, 𝜏i, 𝜎y],
    "pg": [𝜏1, 𝛾y],
    "pmm": [𝜏1, 𝜏i, 𝜌2, 𝜎x],
    "pmg": [𝜏1, 𝜏i, 𝜌2, 𝛾y],
    "pgg": [𝜏1, 𝜏i, 𝜌2, 𝛾q],
    # Square
    "p4": [𝜏1, 𝜌4],
    "p4m": [𝜏1, 𝜌4, 𝜎c],
    "p4g": [𝜏1, 𝜌4, 𝜎s],
    # Hexagonal
    "p3": [𝜏1, 𝜌3],
    "p31m": [𝜏1, 𝜌3, 𝜎x_hex],
    "p3m1": [𝜏1, 𝜌3, 𝜎y_hex],
    "p6": [𝜏1, 𝜌6],
    "p6m": [𝜏1, 𝜌6, 𝜎x_hex],
}
