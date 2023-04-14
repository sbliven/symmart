import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_curve(f, n=500, start=0, end=2 * np.pi, colors=px.colors.cyclical.Phase):
    "plot a complex parametric curve"
    t = np.linspace(start, end, n)

    z = f(t)
    # repeat points to avoid discontinuities from discrete colors
    fig = px.line(
        x=z.real.repeat(2)[1:-1],
        y=z.imag.repeat(2)[1:-1],
        color=np.floor(t.repeat(2)[:-2] * len(colors) / (2 * np.pi)),
        color_discrete_sequence=colors,
    )
    fig.update_layout(showlegend=False)
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def plot_curves(
    *fns,
    n=500,
    start=0,
    end=2 * np.pi,
    rows=1,
    cols=None,
    colors=px.colors.cyclical.Phase
):
    "plot several complex parametric curves as subplots"
    t = np.linspace(start, end, n)
    if rows is None:
        rows = int(np.ceil(np.sqrt(len(fns))))
    if cols is None:
        cols = int(np.ceil(len(fns) / rows))
    fig = make_subplots(rows=rows, cols=cols)

    pts_per_color = int(np.ceil(n / len(colors)))

    for i, f in enumerate(fns):
        z = f(t)
        # repeat points to avoid discontinuities from discrete colors
        row = i // cols + 1
        col = (i % cols) + 1
        for j in range(len(colors)):
            # Add each color as a trace
            fig.add_trace(
                go.Scatter(
                    x=z.real[j * pts_per_color : (j + 1) * pts_per_color + 1],
                    y=z.imag[j * pts_per_color : (j + 1) * pts_per_color + 1],
                    line_color=colors[j],
                    mode="lines",
                ),
                row=row,
                col=col,
            )
        sp = fig.get_subplot(row, col)
        sp.yaxis.scaleanchor = sp.yaxis.anchor
    fig.update_layout(colorscale_sequential=px.colors.cyclical.Phase)
    fig.update_layout(showlegend=False)
    return fig
