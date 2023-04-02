import sympy as sp

from .operators import MatrixOperator, complete_group


def _make_wp_generators():
    half = sp.sympify(1) / 2

    # Basic operations
    𝜏1 = sp.Matrix([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    𝜏i = sp.Matrix([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    𝜌2 = sp.Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    𝜎x = sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    𝛾y = sp.Matrix([[-1, 0, 0], [0, 1, half], [0, 0, 1]])
    𝜌4 = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    𝜎c = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    𝜌3 = sp.Matrix([[0, -1, 0], [1, -1, 0], [0, 0, 1]])

    return {
        "p2": [
            MatrixOperator(𝜏1, "𝜏1"),
            MatrixOperator(𝜏i, "𝜏i"),
            MatrixOperator(𝜌2, "𝜌2"),
        ],
        "pm": [
            MatrixOperator(𝜏1, "𝜏1"),
            MatrixOperator(𝜏i, "𝜏i"),
            MatrixOperator(𝜌2 * 𝜎x, "𝜎y"),
        ],
        "pmm": [
            MatrixOperator(𝜏1, "𝜏1"),
            MatrixOperator(𝜏i, "𝜏i"),
            MatrixOperator(𝜌2, "𝜌2"),
            MatrixOperator(𝜎x, "𝜎x"),
        ],
        "p4": [MatrixOperator(𝜏1, "𝜏1"), MatrixOperator(𝜌4, "𝜌4")],
        "p4m": [
            MatrixOperator(𝜏1, "𝜏1"),
            MatrixOperator(𝜌4, "𝜌4"),
            MatrixOperator(𝜎c, "𝜎c"),
        ],
        "p3": [MatrixOperator(𝜏1, "𝜏1"), MatrixOperator(𝜌3, "𝜌3")],
        "p31m": [
            MatrixOperator(𝜏1, "𝜏1"),
            MatrixOperator(𝜌3, "𝜌3"),
            MatrixOperator(sp.Matrix([[1, -1, 0], [0, -1, 0], [0, 0, 1]]), "𝜎x"),
        ],
        "p3m1": [
            MatrixOperator(𝜏1, "𝜏1"),
            MatrixOperator(𝜌3, "𝜌3"),
            MatrixOperator(sp.Matrix([[-1, 1, 0], [0, 1, 0], [0, 0, 1]]), "𝜎y"),
        ],
    }


wallpaper_generators = _make_wp_generators()
wallpaper_groups = {
    grp: complete_group(gens) for grp, gens in wallpaper_generators.items()
}
