import base64
from functools import partial
from io import BytesIO

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
from PIL import Image

from . import colorwheels as cw
from . import plane_fns as pf
from .plane_fns import matrix_to_src, plane_fn_src
from .unitcell import DashCellDiagram
from .util import metricunit
from .wallpaper_groups import wallpaper_generators, wallpaper_lattices

lattices = list(wallpaper_lattices.keys())

wheels = {
    "Hue-saturation-lightness": cw.hsl_wheel,
    "4-fold stepped": partial(cw.wheel_stepped, hue_steps=4, l_steps=4),
    "5-fold stepped": partial(cw.wheel_stepped, hue_steps=5, l_steps=4),
    "6-fold stepped": partial(cw.wheel_stepped, hue_steps=6, l_steps=4),
    "4-fold gradient": partial(cw.wheel_gradiant, hue_steps=4),
    "5-fold gradient": partial(cw.wheel_gradiant, hue_steps=5),
    "6-fold gradient": partial(cw.wheel_gradiant, hue_steps=6),
    "6-fold wheel": cw.wheel_6,
    "Image: Härtzlisee": cw.LazyWheel(
        "haertzlisee.jpeg", background=(45, 118, 116), fix_aspect=True
    ),
}


def get_colorwheel(tab, builtin, upload_content):
    if tab == "tab-wheel-builtin":
        return wheels[builtin]
    elif tab == "tab-wheel-image":
        if upload_content is None:
            raise PreventUpdate
        img, _ = parse_upload(upload_content)
        arr = np.asarray(img.convert("RGB"))
        return cw.raster_wheel(arr)
    else:
        raise ValueError(f"Unknown tab {tab}")


def get_lattice_fn(lattice: str, a, b):
    lattice = lattice.lower()
    if lattice == "monoclinic" or lattice == "generic":
        fn = partial(pf.wallpaper_generic, b / a)
    elif lattice == "rhombic":
        𝜔 = b / a
        height = 𝜔.imag / 2 / (1 + 𝜔.real)
        fn = partial(pf.wallpaper_rhombic, height)
    elif lattice == "rectangular":
        fn = partial(pf.wallpaper_rectangular, np.abs(b / a))
    elif lattice == "square":
        fn = pf.wallpaper_square
    elif lattice == "hexagonal":
        fn = pf.wallpaper_hexagonal
    else:
        raise ValueError(f"invalid lattice {lattice}")

    return fn


def balance_cell(
    lattice: str, a: complex, b: complex, a_changed: bool
) -> tuple[complex, complex]:
    "Preserve cell constraints after updating a lattice vector"
    lattice = lattice.lower()
    if lattice == "generic" or lattice == "monoclinic":
        # No constraints on generic cell
        pass
    elif lattice == "hexagonal":
        # b = 𝜔3* a
        𝜔3 = np.exp(2j * np.pi / 3)
        if a_changed:
            b = 𝜔3 * a
        else:
            a = b / 𝜔3
    elif lattice == "square":
        # b = i * a
        if a_changed:
            b = 1j * a
        else:
            a = -1j * b
    elif lattice == "rectangular":
        # b/|b| = i * a / |a|
        if a_changed:
            b = 1j * a * abs(b) / abs(a)
        else:
            a = -1j * b * abs(a) / abs(b)
    elif lattice == "rhombic":
        # |a| = |b|
        if a_changed:
            b = b * abs(a) / abs(b)
        else:
            a = a * abs(b) / abs(a)
    else:
        raise ValueError(f"Unknown lattice {lattice}")

    return a, b


def default_cell_params(lattice):
    lattice = lattice.lower()
    if lattice == "generic" or lattice == "monoclinic":
        return 1, 0.5 + 1j
    elif lattice == "hexagonal":
        return 1, np.exp(2j * np.pi / 3)
    elif lattice == "square":
        return 1, 1j
    elif lattice == "rectangular":
        return 1, 0.5j
    elif lattice == "rhombic":
        return 0.5 + 0.5j, 0.5 - 0.5j
    else:
        raise ValueError(f"Unknown lattice {lattice}")


def parse_upload(contents):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)

    fp = BytesIO(decoded)
    return Image.open(fp), len(decoded)


def lattice_tab(app, lattices):
    @app.callback(
        Output("fourier-label", "children"), Input("spacegroup-dropdown", "value")
    )
    def update_lattice(value):
        return f"Fourier coefficients ({value})"

    @app.callback(
        Output("spacegroup-dropdown", "options"),
        Output("spacegroup-dropdown", "value"),
        Input("lattice-dropdown", "value"),
    )
    def update_spacegroup_dropdown(lattice):
        spacegroups = wallpaper_lattices[lattice.lower()]
        # print(f"New sg {spacegroups[0]}")
        return spacegroups, spacegroups[0]

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
        try:
            a = a_real + a_imag * 1j
            b = b_real + b_imag * 1j
        except TypeError:
            raise PreventUpdate
        changed = ctx.triggered_id
        if changed is None or changed == "lattice-dropdown":
            a, b = default_cell_params(lattice)
        else:
            a_changed = changed.startswith("cell-a")
            a, b = balance_cell(lattice, a, b, a_changed)
        # print(f"New cell {a}, {b}")
        return a.real, a.imag, b.real, b.imag

    @app.callback(
        Output("div-celldiagram", "children"),
        Input("cell-a-real", "value"),
        Input("cell-a-imag", "value"),
        Input("cell-b-real", "value"),
        Input("cell-b-imag", "value"),
        Input("spacegroup-dropdown", "value"),
        State("lattice-dropdown", "value"),
    )
    def draw_cell_diagram(a_real, a_imag, b_real, b_imag, spacegroup, lattice):
        try:
            a = a_real + a_imag * 1j
            b = b_real + b_imag * 1j
        except TypeError:
            raise PreventUpdate
        dia = DashCellDiagram(100 * a, 100 * b)
        dia.draw_cell()
        if spacegroup is not None:
            gen = wallpaper_generators[spacegroup.lower()]
            dia.draw_ops(gen)
        # print(f"Updating cell diagram for {spacegroup}")
        svg = dia.draw()
        if type(svg).__name__ != "Svg":
            type(svg)

        return [svg]

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
                                    dbc.Row(
                                        [
                                            dbc.Col([html.Label("Lattice")], width=1),
                                            dbc.Col(
                                                [
                                                    dcc.Dropdown(
                                                        [
                                                            l.capitalize()
                                                            for l in lattices
                                                        ],
                                                        id="lattice-dropdown",
                                                        value="Hexagonal",
                                                        clearable=False,
                                                    )
                                                ]
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [html.Label("Space Group")], width=1
                                            ),
                                            dbc.Col(
                                                [
                                                    dcc.Dropdown(
                                                        ["p1"],
                                                        id="spacegroup-dropdown",
                                                        value="p1",
                                                        clearable=False,
                                                    )
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
                                                        debounce=True,
                                                    ),
                                                    html.Span(" + "),
                                                    dcc.Input(
                                                        id="cell-a-imag",
                                                        value=0,
                                                        type="number",
                                                        style={"width": "4em"},
                                                        debounce=True,
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
                                                        value=0,
                                                        type="number",
                                                        style={"width": "4em"},
                                                        debounce=True,
                                                    ),
                                                    html.Span(" + "),
                                                    dcc.Input(
                                                        id="cell-b-imag",
                                                        value=1,
                                                        type="number",
                                                        style={"width": "4em"},
                                                        debounce=True,
                                                    ),
                                                    html.Em("i"),
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        id="div-celldiagram",
                                        style={
                                            "width": "100%",
                                            "minWidth": 50,
                                            "minHeight": 50,
                                        },
                                    )
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
        Input("tabs-wheels", "value"),
        # Built-ins
        Input("colorwheel-dropdown", "value"),
        # Image
        Input("wheel-upload", "contents"),
        # Scaling
        Input("colorwheel-x-min", "value"),
        Input("colorwheel-x-max", "value"),
        Input("colorwheel-y-min", "value"),
        Input("colorwheel-y-max", "value"),
    )
    def update_colorwheel(
        wheel_tab, wheel_dropdown, wheel_upload, xmin, xmax, ymin, ymax
    ):
        limits = (xmin + ymin * 1j, xmax + ymax * 1j)
        wheel = get_colorwheel(wheel_tab, wheel_dropdown, wheel_upload)
        return plane_fn_src(
            wheel=wheel,
            limits=limits,
            width=400,
            height=400,
        )

    @app.callback(
        Output("upload-info", "children"),
        Input("wheel-upload", "contents"),
        State("wheel-upload", "filename"),
    )
    def update_output(contents, filename):
        if contents is not None:
            img, size = parse_upload(contents)
            return [
                html.P(f"Filename: {filename}"),
                html.P(f"Size: {metricunit(size, base=1024, sigfigs=3)}iB"),
                html.P(f"Resolution: {img.width} x {img.height}"),
            ]
        else:
            return []

    tabs = [
        dcc.Tab(
            value="tab-wheel-builtin",
            label="Built-in",
            children=[
                dcc.Dropdown(
                    list(wheels.keys()),
                    id="colorwheel-dropdown",
                    value="Image: Härtzlisee",
                    clearable=False,
                )
            ],
        ),
        dcc.Tab(
            value="tab-wheel-image",
            label="Image",
            children=[
                dcc.Upload(
                    id="wheel-upload",
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                    multiple=False,
                    accept="image/*",
                ),
                html.Div(id="upload-info"),
            ],
        ),
    ]

    @app.callback(Output("h3-active", "children"), Input("tabs-wheels", "value"))
    def tabswitch(tab):
        return f"Active: {tab}"

    return dcc.Tab(
        label="Color Wheel",
        value="tab-wheel",
        children=[
            html.H3("Active: ", id="h3-active"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Tabs(
                                id="tabs-wheels",
                                value="tab-wheel-builtin",
                                children=tabs,
                                vertical=True,
                                # parent_style={"width": "100%"},
                                # content_style={"width": "100%"},
                                parent_className="container-fluid",
                                content_className="col-9 ps-3",
                                className="col-3",
                            )
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
            ),
        ],
    )


def preview_tab(app, wheels, lattices):
    @app.callback(
        Output("img-preview", "src"),
        # Color wheel
        Input("tabs-wheels", "value"),
        Input("colorwheel-dropdown", "value"),
        Input("wheel-upload", "contents"),
        # Scaling
        Input("preview-x-min", "value"),
        Input("preview-x-max", "value"),
        Input("preview-y-min", "value"),
        Input("preview-y-max", "value"),
        # Function
        Input("fourier-table", "data"),
        Input("fourier-table", "columns"),
        # Cell
        Input("cell-a-real", "value"),
        Input("cell-a-imag", "value"),
        Input("cell-b-real", "value"),
        Input("cell-b-imag", "value"),
        Input("lattice-dropdown", "value"),
    )
    def update_colorwheel(
        wheel_tab,
        wheel_dropdown,
        wheel_upload,
        xmin,
        xmax,
        ymin,
        ymax,
        rows,
        columns,
        a_real,
        a_imag,
        b_real,
        b_imag,
        lattice,
    ):
        try:
            a = a_real + a_imag * 1j
            b = b_real + b_imag * 1j
            lattice_fn = get_lattice_fn(lattice, a, b)
        except TypeError:
            raise PreventUpdate

        df = pd.DataFrame(rows, columns=[c["name"] for c in columns])
        wheel = get_colorwheel(wheel_tab, wheel_dropdown, wheel_upload)
        limits = (xmin + ymin * 1j, xmax + ymax * 1j)
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
                                sort_action="native",
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
            )
        ]
    )

    return app


if __name__ == "__main__":
    make_app().run_server(debug=True)
