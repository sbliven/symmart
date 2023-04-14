from symmart.operators import MatrixOperator
from symmart.unitcell import expand_group
from symmart.wallpaper_groups import 𝛾y, 𝜌2, 𝜌4, 𝜌6, 𝜎x, 𝜎x_hex, 𝜏1, 𝜏i


def test_expand_group():
    assert set(expand_group([𝜌2])) == {
        𝜌2,
        𝜏1 ** 2 * 𝜌2,
        𝜏i ** 2 * 𝜌2,
        𝜏1 ** 2 * 𝜏i ** 2 * 𝜌2,
    }, 𝜌2.description()
    assert list(expand_group([𝜏1 * 𝜏i * 𝜌2])) == [𝜏1 * 𝜏i * 𝜌2]
    assert set(expand_group([𝜏1 * 𝜌2])) == {𝜏1 * 𝜌2, 𝜏1 * 𝜏i ** 2 * 𝜌2}

    assert set(expand_group([𝜎x])) == {𝜎x, 𝜏i ** 2 * 𝜎x}, 𝜎x.description()

    # glide reflection across x - y = 1/2 by (1/2, 1/2)
    𝜎d_plus = MatrixOperator([[0, 1, 1], [1, 0, 0], [0, 0, 1]])
    # glide reflection across x - y = -1/2 by (1/2, 1/2)
    𝜎d_minus = MatrixOperator([[0, 1, 0], [1, 0, 1], [0, 0, 1]])

    assert set(expand_group([𝜎d_plus])) == {𝜎d_plus, 𝜎d_minus}, 𝜎d_plus.description()

    𝜎y = MatrixOperator([[-1, 0, 0], [-1, 1, 0], [0, 0, 1]], "𝜎y")
    𝜎y1 = MatrixOperator([[-1, 0, 2], [-1, 1, 1], [0, 0, 1]], "𝜎y1")
    assert set(expand_group([𝜎y])) == {𝜎y, 𝜎y1}, 𝜎y.description()
