from functools import partial

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme

from . import colorwheels as cw
from . import plane_fns as pf
from .plane_fns import matrix_to_src, plane_fn_src

lattices = [
    "hexagonal",
    "generic",
    # "rhombic",
    "rectangular",
    # "square"
]


def get_lattice_fn(lattice: str, a, b):
    if lattice == "generic":
        fn = pf.wallpaper_generic
    # elif lattice == "rhombic":
    #     fn = pf.wallpaper_rhombic
    elif lattice == "rectangular":
        fn = pf.wallpaper_rectangular
    # elif lattice == "square":
    #     fn = pf.wallpaper_square
    elif lattice == "hexagonal":
        fn = pf.wallpaper_hexagonal
    else:
        raise ValueError(f"invalid lattice {lattice}")

    return fn


def balance_cell(lattice, a, b, a_changed):
    "Preserve cell constraints after updating a lattice vector"
    if lattice.lower() == "generic":
        # No constraints on generic cell
        pass
    elif lattice.lower() == "hexagonal":
        # b = 𝜔3* a
        𝜔3 = np.exp(2j * np.pi / 3)
        if a_changed:
            b = 𝜔3 * a
        else:
            a = b / 𝜔3
    elif lattice.lower() == "square":
        # b = i * a
        if a_changed:
            b = 1j * a
        else:
            a = -1j * b
    elif lattice.lower() == "rectangular":
        # b/|b| = i * a / |a|
        if a_changed:
            b = 1j * a * abs(b) / abs(a)
        else:
            a = -1j * b * abs(a) / abs(b)
    elif lattice.lower() == "rhombic":
        # |a| = |b|
        if a_changed:
            b = b * abs(a) / abs(b)
        else:
            a = a * abs(b) / abs(a)
    else:
        raise ValueError(f"Unknown lattice {lattice}")
    return a, b


def lattice_tab(app, lattices):
    @app.callback(
        Output("fourier-label", "children"), Input("lattice-dropdown", "value")
    )
    def update_lattice(value):
        return f"Fourier coefficients ({value.lower()} lattice)"

    @app.callback(
        Output("cell-a-real", "value"),
        Output("cell-a-imag", "value"),
        Output("cell-b-real", "value"),
        Output("cell-b-imag", "value"),
        Input("cell-a-real", "value"),
        Input("cell-a-imag", "value"),
        Input("cell-b-real", "value"),
        Input("cell-b-imag", "value"),
        Input("lattice-dropdown", "value"),
    )
    def balance_cell_callback(a_real, a_imag, b_real, b_imag, lattice):
        "Preserve cell constraints after updating a lattice vector"
        a = a_real + a_imag * 1j
        b = b_real + b_imag * 1j
        a_changed = ctx.triggered_id is None or ctx.triggered_id.startswith("cell-a")
        a, b = balance_cell(lattice, a, b, a_changed)
        return a.real, a.imag, b.real, b.imag

    return dcc.Tab(
        label="Lattice",
        value="tab-lattice",
        children=[
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Lattice"),
                                ],
                                width=1,
                            ),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        [l.capitalize() for l in lattices],
                                        id="lattice-dropdown",
                                        value=lattices[0].capitalize(),
                                    ),
                                ]
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col([html.Label("a")], width=1),
                            dbc.Col(
                                [
                                    dcc.Input(
                                        id="cell-a-real",
                                        value=1,
                                        type="number",
                                        style={"width": "4em"},
                                    ),
                                    html.Span(" + "),
                                    dcc.Input(
                                        id="cell-a-imag",
                                        value=0,
                                        type="number",
                                        style={"width": "4em"},
                                    ),
                                    html.Em("i"),
                                ]
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col([html.Label("b")], width=1),
                            dbc.Col(
                                [
                                    dcc.Input(
                                        id="cell-b-real",
                                        value=1,
                                        type="number",
                                        style={"width": "4em"},
                                    ),
                                    html.Span(" + "),
                                    dcc.Input(
                                        id="cell-b-imag",
                                        value=0,
                                        type="number",
                                        style={"width": "4em"},
                                    ),
                                    html.Em("i"),
                                ]
                            ),
                        ]
                    ),
                ]
            )
        ],
    )


def wheel_tab(app, wheels):
    @app.callback(
        Output("img-wheel", "src"),
        Input("colorwheel-dropdown", "value"),
        Input("colorwheel-x-min", "value"),
        Input("colorwheel-x-max", "value"),
        Input("colorwheel-y-min", "value"),
        Input("colorwheel-y-max", "value"),
    )
    def update_colorwheel(value, xmin, xmax, ymin, ymax):
        print(f"Updating to {value}")
        wheel = wheels[value]
        limits = (xmin + ymin * 1j, xmax + ymax * 1j)
        return plane_fn_src(
            wheel=wheel,
            limits=limits,
            width=400,
            height=400,
        )

    return dcc.Tab(
        label="Color Wheel",
        value="tab-wheel",
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                list(wheels.keys()),
                                id="colorwheel-dropdown",
                                value=next(iter(wheels.keys())),
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                html.Img(
                                    id="img-wheel",
                                    className="img-fluid",
                                    style={
                                        "width": "100%",
                                        "minWidth": "50px",
                                        "minHeight": "50px",
                                    },
                                    src=matrix_to_src(
                                        np.zeros((1, 1, 3), dtype=np.uint8)
                                    ),
                                )
                            ),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="colorwheel-x-min",
                                        type="number",
                                        value=-2,
                                        style={"width": "4em"},
                                    ),
                                    html.Label("X"),
                                    dcc.Input(
                                        id="colorwheel-x-max",
                                        type="number",
                                        value=2,
                                        style={"width": "4em"},
                                    ),
                                ],
                                className="d-flex justify-content-between",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="colorwheel-y-min",
                                                type="number",
                                                value=-2,
                                                style={"width": "4em"},
                                            ),
                                            html.Em("i"),
                                        ]
                                    ),
                                    html.Label("Y"),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="colorwheel-y-max",
                                                type="number",
                                                value=2,
                                                style={"width": "4em"},
                                            ),
                                            html.Em("i"),
                                        ]
                                    ),
                                ],
                                className="d-flex justify-content-between",
                            ),
                        ],
                        width=6,
                    ),
                ]
            )
        ],
    )


def preview_tab(app, wheels, lattices):
    @app.callback(
        Output("img-preview", "src"),
        Input("colorwheel-dropdown", "value"),
        Input("preview-x-min", "value"),
        Input("preview-x-max", "value"),
        Input("preview-y-min", "value"),
        Input("preview-y-max", "value"),
        Input("fourier-table", "data"),
        Input("fourier-table", "columns"),
        Input("lattice-dropdown", "value"),
    )
    def update_colorwheel(value, xmin, xmax, ymin, ymax, rows, columns, lattice):
        df = pd.DataFrame(rows, columns=[c["name"] for c in columns])
        print(f"Got coefficients {df}")
        wheel = wheels[value]
        limits = (xmin + ymin * 1j, xmax + ymax * 1j)
        lattice_fn = get_lattice_fn(lattice.lower(), None, None)
        return plane_fn_src(
            lattice_fn(df.a.astype(float), df.n.astype(int), df.m.astype(int)),
            wheel=wheel,
            limits=limits,
            width=400,
            height=400,
        )

    @app.callback(
        Output("fourier-table", "data"),
        Input("btn-add-row", "n_clicks"),
        State("fourier-table", "data"),
    )
    def add_row(n_clicks, rows):
        if n_clicks > 0:
            rows.append(dict(a=0, n=0, m=0))
        return rows

    return dcc.Tab(
        label="Preview",
        value="tab-preview",
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Fourier coefficients", id="fourier-label"),
                            dash_table.DataTable(
                                id="fourier-table",
                                data=[dict(a=1, n=1, m=0)],
                                columns=[
                                    dict(
                                        name="a",
                                        id="a",
                                        type="numeric",
                                        format=Format(precision=1, scheme=Scheme.fixed),
                                    ),
                                    dict(
                                        name="n",
                                        id="n",
                                        type="numeric",
                                        format=Format(
                                            precision=0, scheme=Scheme.decimal_integer
                                        ),
                                    ),
                                    dict(
                                        name="m",
                                        id="m",
                                        type="numeric",
                                        format=Format(
                                            precision=0, scheme=Scheme.decimal_integer
                                        ),
                                    ),
                                ],
                                editable=True,
                                row_deletable=True,
                            ),
                            html.Button("Add row", id="btn-add-row", n_clicks=0),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                html.Img(
                                    id="img-preview",
                                    className="img-fluid",
                                    style={
                                        "width": "100%",
                                        "minWidth": "50px",
                                        "minHeight": "50px",
                                    },
                                    src=matrix_to_src(
                                        np.zeros((1, 1, 3), dtype=np.uint8)
                                    ),
                                )
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="preview-x-min",
                                                type="number",
                                                value=-2,
                                                style={"width": "4em"},
                                            ),
                                            html.Em(
                                                "",
                                                style={
                                                    "width": ".8em",
                                                    "textAlign": "right",
                                                    "display": "inline-block",
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Label("X"),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="preview-x-max",
                                                type="number",
                                                value=2,
                                                style={"width": "4em"},
                                            ),
                                            html.Em(
                                                "",
                                                style={
                                                    "width": ".8em",
                                                    "textAlign": "right",
                                                    "display": "inline-block",
                                                },
                                            ),
                                        ],
                                    ),
                                ],
                                className="d-flex justify-content-between",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="preview-y-min",
                                                type="number",
                                                value=-2,
                                                style={"width": "4em"},
                                            ),
                                            html.Em(
                                                "i",
                                                style={
                                                    "width": ".8em",
                                                    "textAlign": "right",
                                                    "display": "inline-block",
                                                },
                                            ),
                                        ]
                                    ),
                                    html.Label("Y"),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="preview-y-max",
                                                type="number",
                                                value=2,
                                                style={"width": "4em"},
                                            ),
                                            html.Em(
                                                "i",
                                                style={
                                                    "width": ".8em",
                                                    "textAlign": "right",
                                                    "display": "inline-block",
                                                },
                                            ),
                                        ]
                                    ),
                                ],
                                className="d-flex justify-content-between",
                            ),
                        ],
                        width=6,
                    ),
                ]
            )
        ],
    )


def export_tab(app):
    return dcc.Tab(
        label="Export", value="tab-export", children=[html.Div([html.H3("export")])]
    )


def make_app(app=None):
    if app is None:
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    wheels = {
        "4-fold stepped": partial(cw.wheel_stepped, hue_steps=4),
        "6-fold wheel": cw.wheel_6,
    }

    tabs = [
        lattice_tab(app, lattices),
        wheel_tab(app, wheels),
        preview_tab(app, wheels, lattices),
        export_tab(app),
    ]

    app.layout = dbc.Container(
        [
            dcc.Tabs(
                id="tabs",
                value="tab-preview",
                children=tabs,
            ),
            html.Div(id="tab-content"),
        ]
    )

    # @callback(Output("tab-content", "children"), Input("tabs", "value"))
    # def render_content(tab):
    #     if tab == "tab-lattice":
    #     elif tab == "tab-wheel":
    #     elif tab == "tab-preview":

    #     elif tab == "tab-export":

    return app


if __name__ == "__main__":
    make_app().run_server(debug=True)
