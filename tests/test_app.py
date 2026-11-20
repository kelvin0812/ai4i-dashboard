"""End-to-end app tests with Streamlit's AppTest (no browser needed).

Every page must render without errors, and every sidebar filter must change
what every exploratory page shows.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
EXPLORE_PAGES = [
    "app_pages/1_overview.py",
    "app_pages/2_distributions.py",
    "app_pages/3_relationships.py",
    "app_pages/4_failure_modes.py",
]

# (description, function that applies the filter to a running AppTest)
FILTERS = {
    "type=H": lambda at: at.multiselect(key="filter_types").set_value(["H"]),
    "failure only": lambda at: at.radio(key="filter_status").set_value("Failure only"),
    "no failure only": lambda at: at.radio(key="filter_status").set_value("No failure only"),
    "air temp": lambda at: at.slider(key="filter_range_air_temperature_k").set_value((298.0, 301.0)),
    "process temp": lambda at: at.slider(key="filter_range_process_temperature_k").set_value((308.0, 311.0)),
    "speed": lambda at: at.slider(key="filter_range_rotational_speed_rpm").set_value((1168, 1500)),
    "torque": lambda at: at.slider(key="filter_range_torque_nm").set_value((40.0, 76.6)),
    "tool wear": lambda at: at.slider(key="filter_range_tool_wear_min").set_value((100, 253)),
}


def _new_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _snapshot(at: AppTest) -> tuple:
    """Everything filter-dependent on the page: chart specs + metric values."""
    charts = tuple(el.proto.spec for el in at.get("plotly_chart"))
    metrics = tuple(m.value for m in at.metric)
    return charts, metrics


@pytest.mark.parametrize("page", EXPLORE_PAGES + ["app_pages/5_prediction.py"])
def test_page_renders(page):
    at = _new_app()
    at.switch_page(page).run()
    assert not at.exception, at.exception
    if page != "app_pages/5_prediction.py":
        assert len(at.get("plotly_chart")) >= 1


def test_overview_kpis_unfiltered():
    at = _new_app()
    values = [m.value for m in at.metric]
    assert values[:3] == ["10,000", "339", "3.39%"]


# Set the filter on the landing page, then switch: this also proves the
# selection carries over to the other pages through session_state.
def _page_snapshot(page, filter_name=None):
    at = _new_app()
    if filter_name:
        FILTERS[filter_name](at)
        at.run()
        assert not at.exception, at.exception
    at.switch_page(page).run()
    assert not at.exception, at.exception
    return _snapshot(at)


@pytest.mark.parametrize("page", EXPLORE_PAGES)
@pytest.mark.parametrize("filter_name", list(FILTERS))
def test_every_filter_changes_every_page(page, filter_name):
    baseline = _page_snapshot(page)
    after = _page_snapshot(page, filter_name)
    assert after[0], "page has no charts after filtering"
    assert after != baseline, f"{filter_name} did not change {page}"
    # Every chart on the page must change, not just one.
    changed = [a != b for a, b in zip(after[0], baseline[0])]
    assert all(changed), f"{filter_name}: unchanged chart(s) on {page}: {changed}"


@pytest.mark.parametrize("page", EXPLORE_PAGES)
def test_empty_filter_shows_no_data(page):
    at = _new_app()
    at.multiselect(key="filter_types").set_value([]).run()
    at.switch_page(page).run()
    assert not at.exception, at.exception
    assert any("No data" in w.value for w in at.warning)
    assert len(at.get("plotly_chart")) == 0


def test_filters_persist_across_pages():
    at = _new_app()
    at.multiselect(key="filter_types").set_value(["M"]).run()
    at.switch_page("app_pages/3_relationships.py").run()
    assert at.session_state["filter_types"] == ["M"]
    assert any("2,997 of 10,000" in c.value for c in at.caption)
    at.switch_page("app_pages/1_overview.py").run()
    assert [m.value for m in at.metric][:2] == ["2,997", "83"]


def test_reset_button():
    at = _new_app()
    at.radio(key="filter_status").set_value("Failure only").run()
    at.button[0].click().run()
    assert at.radio(key="filter_status").value == "All records"
    assert [m.value for m in at.metric][:2] == ["10,000", "339"]

