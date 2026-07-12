"""Phase 9: DR severity probability chart.

Severity is ORDINAL (swapping the class order would change its meaning),
not nominal categorical identity -- per the dataviz skill's color-formula.md
this takes a single one-hue ramp with monotone lightness, not the
categorical eight-hue palette. The ramp below is this app's own primary
accent indigo (--vdx-primary, #3525CD -- see theme.py's module docstring
for the "Clinical Liquid Glass" reference this re-hue matches) stepped
light->dark, same single-hue/monotone-lightness shape as the ramp it
replaces.

A single "series" (the probability distribution) needs no legend box per
the same reference -- identity here comes from the y-axis category labels,
not a swatch legend. The exact same numbers are also shown in the
"Severity Probabilities" table section report/content.py builds (rendered
right after this chart in both the PDF and the in-app preview), which is
this chart's table-view fallback.
"""

import plotly.graph_objects as go

from src.detection.model import SEVERITY_LABELS

_ORDINAL_RAMP = ["#C3C0FF", "#8F86F5", "#6355E8", "#3525CD", "#241A8F"]

_PRIMARY_INK = "#191C1E"
_MUTED_INK = "#464555"
_GRIDLINE = "#C7C4D8"

# A genuine 0.0% bar renders with literally no visible mark -- confirmed
# live, a true zero was indistinguishable at a glance from a missing/broken
# data row (the label just floats at the left margin with no bar next to
# it). This is a DISPLAY-only floor on the drawn bar length; the printed
# percentage and the hover tooltip both still read the true value via
# `customdata`, so a confirmed-zero category still says "0.0%" -- it just
# now also gets a thin visible tick instead of nothing.
_MIN_VISIBLE_BAR = 1.5


def probability_bar_chart(detection: dict) -> go.Figure:
    """Horizontal bar chart of the 5-class DR severity probability
    distribution. `detection` is detection/infer.predict()'s return dict
    (pipeline.run_pipeline()'s "detection" key, when not None).
    """
    labels = [SEVERITY_LABELS[i] for i in range(5)]
    true_values = [p * 100 for p in detection["probabilities"]]
    display_values = [max(v, _MIN_VISIBLE_BAR) for v in true_values]
    predicted_idx = detection["class_idx"]

    # Selective direct labels: every bar gets its value (there are only
    # five, and the distribution across all of them IS the story), but
    # only the model's top pick is bolded in primary ink -- the others
    # stay in muted/secondary ink so the emphasis reads as "this one",
    # never by recoloring a bar itself (that would break the ordinal
    # severity-ramp meaning).
    text_labels = [f"<b>{v:.1f}%</b>" if i == predicted_idx else f"{v:.1f}%" for i, v in enumerate(true_values)]
    text_colors = [_PRIMARY_INK if i == predicted_idx else _MUTED_INK for i in range(5)]

    figure = go.Figure(
        go.Bar(
            x=display_values,
            y=labels,
            orientation="h",
            marker=dict(color=_ORDINAL_RAMP, cornerradius=4),
            text=text_labels,
            textposition="outside",
            textfont=dict(color=text_colors),
            customdata=true_values,
            cliponaxis=False,
            hovertemplate="%{y}: %{customdata:.1f}%<extra></extra>",
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


# Binary classifiers (glaucoma, AMD) are NOMINAL-with-semantic-meaning, not
# ordinal like DR severity above -- "present" isn't a step further along a
# scale from "absent", it's the other one of two states. So this uses the
# app's own two semantic colors directly (indigo = normal/absent, tertiary
# orange = attention/present, same mapping as render_pill()'s "normal"/
# "attention" variants in components.py) rather than a single-hue ordinal
# ramp.
_PRIMARY = "#3525CD"
_TERTIARY = "#A44100"


def binary_probability_chart(detection: dict, labels: dict) -> go.Figure:
    """Horizontal bar chart for a 2-class (absent/present) classifier.
    `detection` is glaucoma_infer.predict()/amd_infer.predict()'s return
    dict; `labels` is that model's LABELS dict (class_idx -> display text).
    Gives glaucoma/AMD the same probability-breakdown view DR's
    probability_bar_chart() already provides, instead of that breakdown
    only existing in the PDF-mirror report preview.
    """
    display_labels = [labels[i] for i in range(2)]
    true_values = [p * 100 for p in detection["probabilities"]]
    display_values = [max(v, _MIN_VISIBLE_BAR) for v in true_values]
    predicted_idx = detection["class_idx"]
    bar_colors = [_PRIMARY, _TERTIARY]

    text_labels = [f"<b>{v:.1f}%</b>" if i == predicted_idx else f"{v:.1f}%" for i, v in enumerate(true_values)]
    text_colors = [_PRIMARY_INK if i == predicted_idx else _MUTED_INK for i in range(2)]

    figure = go.Figure(
        go.Bar(
            x=display_values,
            y=display_labels,
            orientation="h",
            marker=dict(color=bar_colors, cornerradius=4),
            text=text_labels,
            textposition="outside",
            textfont=dict(color=text_colors),
            customdata=true_values,
            cliponaxis=False,
            hovertemplate="%{y}: %{customdata:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=140,
        margin=dict(l=10, r=40, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, 'Segoe UI', sans-serif", color=_PRIMARY_INK),
        bargap=0.5,
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
