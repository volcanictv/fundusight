"""Headless smoke test for src/app/main.py using Streamlit's own AppTest
runner -- actually executes the page script (widget state, reruns,
st.stop()) rather than importing individual functions, without needing a
real browser. Not a substitute for eyeballing the rendered UI (layout,
CSS, chart legibility) -- see the manual verification steps in the
project's Phase 8/9 plan for that -- but it does catch the class of bug a
browser click-through would (a widget key typo, an exception raised
partway down the script, a missing session_state key).
"""

import sys

import pytest
from streamlit.testing.v1 import AppTest

_APP_PATH = "src/app/main.py"
# The full pipeline (detection + Grad-CAM + hybrid vessel/optic-disc U-Nets)
# on a real fundus photo takes well past AppTest's default few-second
# timeout on CPU.
_RUN_TIMEOUT = 180


@pytest.fixture(autouse=True)
def _fresh_theme_module():
    """theme.py registers its ambient-cursor CCv2 component as a
    module-level side effect at import time -- the pattern Streamlit's own
    docs recommend (register once, not per-rerun), and it's correct for a
    real server process where the component registry is equally long-lived.
    AppTest instead builds a fresh mock registry per instance; Python's
    module cache means a second AppTest in the same pytest process reuses
    the FIRST instance's now-stale registration (theme.py doesn't re-import,
    so it never re-registers), which raises "Component ... is not
    registered" even though the same app works fine for real (confirmed
    live: the failing test passes on its own, only fails after another
    AppTest ran first in the same process). Evicting src.app.theme from
    sys.modules before each test forces it to re-import -- and re-register
    -- against that test's own registry.
    """
    for name in list(sys.modules):
        if name == "src.app.theme":
            del sys.modules[name]
    yield


def test_app_shows_intake_screen_with_no_image_selected():
    # Third redesign pass ("Clinical Liquid Glass", see main.py's module
    # docstring): before an image is available, the app shows a centered
    # intake panel (render_intake_screen()) instead of the old plain
    # st.info() prompt -- assert its headline and gated "Initialize
    # analysis" button render instead of the retired message.
    at = AppTest.from_file(_APP_PATH).run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Patient intake" in markdown_text
    initialize_buttons = [b for b in at.button if b.key == "initialize_btn"]
    assert len(initialize_buttons) == 1
    assert initialize_buttons[0].disabled  # no image chosen yet


def test_demo_mode_runs_full_pipeline_without_exceptions():
    at = AppTest.from_file(_APP_PATH).run(timeout=_RUN_TIMEOUT)
    # Demo mode / patient ID / the image source all live in the intake
    # panel (main area) on first launch now, not the sidebar -- see
    # render_intake_screen(). Toggling demo mode makes a sample image
    # available, which un-disables "Initialize analysis"; clicking it sets
    # the "_vdx_started" session-state flag the rest of the app gates on.
    at.toggle(key="demo_mode").set_value(True)
    at = at.run(timeout=_RUN_TIMEOUT)
    at.button(key="initialize_btn").click()
    at = at.run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    # Every major dashboard row header should have rendered -- a stage
    # silently failing to produce output would shrink this list. Redesign:
    # the per-disease/quality/vessel/optic-disc labels are no longer
    # st.header/st.subheader roles (they're compact tile titles rendered as
    # raw HTML via st.markdown, see app/main.py's _tile_label()/
    # render_stat_tile()), so those are checked via markdown content below
    # instead of the header-role collection.
    #
    # "Report Preview" is gone (second redesign pass, see app/main.py's
    # module docstring): that section used to duplicate everything already
    # shown above it. The Recommendation card (checked via markdown content
    # below) is the one part of it that survived, since it's not shown
    # anywhere else on the page.
    headers = {h.value for h in at.header} | {h.value for h in at.subheader}
    assert {
        "Results",
        "Overview",
        "Disease Screening",
        "Biomarkers",
        "Image Comparison",
    } <= headers

    markdown_text = " ".join(m.value for m in at.markdown)
    for expected in (
        "Image Quality",
        "Preprocessing",
        "Diabetic Retinopathy",
        "Glaucoma",
        "AMD",
        "Vessel Biomarkers",
        "Optic Disc",
        "Recommendation",
    ):
        assert expected in markdown_text, f"expected tile label {expected!r} not found"
