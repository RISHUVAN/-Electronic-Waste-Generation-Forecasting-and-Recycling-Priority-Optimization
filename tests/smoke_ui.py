"""UI smoke test WITHOUT a browser: runs app.py and every page against the real dataset using a stand-in for Streamlit
(and a stand-in for Plotly if it is not installed). It verifies that page code executes and builds figures; it does
not verify visual rendering. Run:  python tests/smoke_ui.py"""
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import plotly  # noqa: F401
    REAL_PLOTLY = True
except ImportError:
    REAL_PLOTLY = False
    for m in ["plotly", "plotly.express", "plotly.graph_objects", "plotly.io"]:
        sys.modules[m] = MagicMock()
    sys.modules["plotly"].express, sys.modules["plotly"].graph_objects = sys.modules["plotly.express"], sys.modules["plotly.graph_objects"]
    sys.modules["plotly"].io = sys.modules["plotly.io"]


class Stop(Exception):
    pass


class FakeSt:
    def __init__(self, overrides):
        self.o, self.calls, self.errors, self.session_state, self.sidebar = overrides, {}, [], {}, self
        self.column_config = MagicMock()

    def _rec(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def __getattr__(self, name):
        def f(*a, **k):
            self._rec(name)
            if name == "error":
                self.errors.append(a[0] if a else "")
        return f

    def _get(self, label, k, default):
        return self.o.get(k.get("key") or label, default)

    def plotly_chart(self, fig, **k):
        self._rec("plotly_chart")
        if REAL_PLOTLY:
            fig.to_json()

    def stop(self): raise Stop()
    def cache_data(self, *a, **k): return a[0] if a and callable(a[0]) else (lambda f: f)
    def tabs(self, labels): return [self] * len(labels)
    def columns(self, spec, **k): return [self] * (spec if isinstance(spec, int) else len(spec))
    def expander(self, *a, **k): return self
    def container(self, *a, **k): return self
    def selectbox(self, label, options, index=0, **k):
        options = list(options)
        return self._get(label, k, options[index] if options else None)
    def radio(self, label, options, index=0, **k): return self._get(label, k, list(options)[index])
    def multiselect(self, label, options, default=None, **k): return self._get(label, k, list(default or []))
    def slider(self, label, min_value=None, max_value=None, value=None, **k): return self._get(label, k, value if value is not None else min_value)
    def toggle(self, label, value=False, **k): return self._get(label, k, value)
    def text_input(self, label, value="", **k): return self._get(label, k, value)


def run(overrides, name):
    fake = FakeSt(overrides)
    mod = ModuleType("streamlit")
    mod.__dict__.update({n: getattr(fake, n) for n in dir(FakeSt) if not n.startswith("__")})
    mod.__getattr__ = lambda n: getattr(fake, n)
    mod.sidebar, mod.session_state, mod.column_config = fake, fake.session_state, fake.column_config
    sys.modules["streamlit"] = mod
    for m in [k for k in sys.modules if k.startswith("src.")]:
        if m.startswith("src.ui") or m == "src.viz.charts":
            del sys.modules[m]
    try:
        runpy.run_path(str(ROOT / "app.py"), run_name="app")
    except Stop:
        pass
    print(f"{name:55s} charts={fake.calls.get('plotly_chart', 0):2d} tables={fake.calls.get('dataframe', 0):2d} "
          f"downloads={fake.calls.get('download_button', 0)} errors={fake.errors}")
    return fake


PAGES = ["Executive Overview", "Forecast Studio", "Recycling Gap Analysis", "City Priority Ranking", "City Comparison",
         "Data Explorer", "Methodology & Model Performance"]

if __name__ == "__main__":
    print("plotly:", "real" if REAL_PLOTLY else "stand-in")
    for p in PAGES:
        f = run({"nav": p}, p)
        assert not f.errors, f.errors
    extra = {
        "Gap: recycling, generated basis, selected year": {"nav": "Recycling Gap Analysis", "gap_kind": "Estimated recycling gap", "gap_basis": "Share of generated waste", "gap_view": "Selected year"},
        "Priority: recycling gap + fallback": {"nav": "City Priority Ranking", "pr_kind": "Estimated recycling gap", "pr_fallback": True},
        "Priority: forecast-only, zero infra weight": {"nav": "City Priority Ranking", "w_infrastructure_deficiency": 0},
        "Forecast: Polynomial model, horizon 5": {"nav": "Forecast Studio", "f_model": "Polynomial", "fs_horizon": 5},
        "Filter: Delhi only": {"nav": "Executive Overview", "f_states": ["Delhi"]},
        "Filter: no match -> empty state": {"nav": "Forecast Studio", "f_states": ["Goa"], "f_years": (2015, 2015)},
        "Comparison: Delhi + Mumbai": {"nav": "City Comparison", "Cities to compare (2-6)": ["Delhi", "Mumbai"]},
    }
    for name, ov in extra.items():
        f = run(ov, name)
        assert not f.errors, f.errors
    print("SMOKE TEST PASSED")
