"""Headless smoke test for src/app/main.py using Streamlit's own AppTest
runner -- actually executes the page script (widget state, reruns,
st.stop()) rather than importing individual functions, without needing a
real browser. Not a substitute for eyeballing the rendered UI (layout,
CSS, chart legibility) -- see the manual verification steps in the
project's Phase 8/9 plan for that -- but it does catch the class of bug a
browser click-through would (a widget key typo, an exception raised
partway down the script, a missing session_state key).
"""

from streamlit.testing.v1 import AppTest

_APP_PATH = "src/app/main.py"
# The full pipeline (detection + Grad-CAM + hybrid vessel/optic-disc U-Nets)
# on a real fundus photo takes well past AppTest's default few-second
# timeout on CPU.
_RUN_TIMEOUT = 180


def test_app_shows_upload_prompt_with_no_image_selected():
    at = AppTest.from_file(_APP_PATH).run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    assert any("Upload a fundus photo" in info.value for info in at.info)


def test_demo_mode_runs_full_pipeline_without_exceptions():
    at = AppTest.from_file(_APP_PATH).run(timeout=_RUN_TIMEOUT)
    at.sidebar.toggle(key="demo_mode").set_value(True)
    at = at.run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    # Every major dashboard row header should have rendered -- a stage
    # silently failing to produce output would shrink this list. Redesign:
    # the per-disease/quality/vessel/optic-disc labels are no longer
    # st.header/st.subheader roles (they're compact tile titles rendered as
    # raw HTML via st.markdown, see app/main.py's _tile_label()/
    # render_stat_tile()), so those are checked via markdown content below
    # instead of the header-role collection.
    headers = {h.value for h in at.header} | {h.value for h in at.subheader}
    assert {
        "Results",
        "Overview",
        "Disease Screening",
        "Biomarkers",
        "Image Comparison",
        "Report Preview",
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
    ):
        assert expected in markdown_text, f"expected tile label {expected!r} not found"
