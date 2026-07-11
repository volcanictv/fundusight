"""Phase 9: DR severity probability chart.

Severity is ORDINAL (swapping the class order would change its meaning),
not nominal categorical identity -- per the dataviz skill's color-formula.md
this takes a single one-hue ramp with monotone lightness, not the
categorical eight-hue palette. The ramp below is this app's own primary
accent teal (--vdx-teal, #0E7C86 -- see theme.py's module docstring for why
teal replaced the old flat blue in the glass redesign) stepped light->dark,
same single-hue/monotone-lightness shape as the ramp it replaces.

A single "series" (the probability distribution) needs no legend box per
the same reference -- identity here comes from the y-axis category labels,
not a swatch legend. The exact same numbers are also shown in the
"Severity Probabilities" table section report/content.py builds (rendered
right after this chart in both the PDF and the in-app preview), which is
this chart's table-view fallback.
"""

import plotly.graph_objects as go

from src.detection.model import SEVERITY_LABELS

_ORDINAL_RAMP = ["#8FD0D6", "#5CB8C0", "#2C9CA6", "#0E7C86", "#0A5960"]

_PRIMARY_INK = "#1A1D23"
_MUTED_INK = "#5F6570"
_GRIDLINE = "#DCE0E7"


def probability_bar_chart(detection: dict) -> go.Figure:
    """Horizontal bar chart of the 5-class DR severity probability
    distribution. `detection` is detection/infer.predict()'s return dict
    (pipeline.run_pipeline()'s "detection" key, when not None).
    """
    labels = [SEVERITY_LABELS[i] for i in range(5)]
    values = [p * 100 for p in detection["probabilities"]]
    predicted_idx = detection["class_idx"]

    # Selective direct labels: every bar gets its value (there are only
    # five, and the distribution across all of them IS the story), but
    # only the model's top pick is bolded in primary ink -- the others
    # stay in muted/secondary ink so the emphasis reads as "this one",
    # never by recoloring a bar itself (that would break the ordinal
    # severity-ramp meaning).
    text_labels = [f"<b>{v:.1f}%</b>" if i == predicted_idx else f"{v:.1f}%" for i, v in enumerate(values)]
    text_colors = [_PRIMARY_INK if i == predicted_idx else _MUTED_INK for i in range(5)]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=_ORDINAL_RAMP, cornerradius=4),
            text=text_labels,
            textposition="outside",
            textfont=dict(color=text_colors),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=280,
        margin=dict(l=10, r=40, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, 'Segoe UI', sans-serif", color=_PRIMARY_INK),
        bargap=0.45,  # air around each bar -- never fill the category slot
        xaxis=dict(
            range=[0, 112],
            showgrid=True,
            gridcolor=_GRIDLINE,
            ticksuffix="%",
            zeroline=False,
            showline=False,
            tickfont=dict(color=_MUTED_INK),
        ),
        yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(color=_PRIMARY_INK)),
        showlegend=False,
    )
    return figure
