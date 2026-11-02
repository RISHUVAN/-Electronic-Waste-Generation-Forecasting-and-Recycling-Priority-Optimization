"""Small reusable UI building blocks."""
import html
import inspect
import streamlit as st


def _new_width(fn):
    """True if this Streamlit version's `width` parameter accepts 'stretch' (newer releases)."""
    try:
        p = inspect.signature(fn).parameters.get("width")
        return p is not None and ("stretch" in str(p.annotation) or "Width" in str(p.annotation))
    except (TypeError, ValueError):
        return False


def _stretch(fn):
    return {"width": "stretch"} if _new_width(fn) else {"use_container_width": True}


def header(title, subtitle):
    st.markdown(f'<div class="page-head"><div class="eyebrow">India E-Waste Intelligence</div><h1>{html.escape(title)}</h1>'
                f'<p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def kpis(items):
    """items: iterable of (label, value, sub)."""
    cards = "".join(f'<div class="kpi"><div class="kpi-label">{html.escape(str(l))}</div><div class="kpi-value">{html.escape(str(v))}</div>'
                    f'<div class="kpi-sub">{html.escape(str(s))}</div></div>' for l, v, s in items)
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def callout(kind, text_html):
    """kind: info | warn | danger | ok. text_html may contain <b> tags (caller supplies trusted text)."""
    cls = "" if kind == "info" else f" {kind}"
    st.markdown(f'<div class="callout{cls}">{text_html}</div>', unsafe_allow_html=True)


def empty(title, hint=""):
    st.markdown(f'<div class="empty"><b>{html.escape(title)}</b>{html.escape(hint)}</div>', unsafe_allow_html=True)


def label(text):
    st.markdown(f'<div class="side-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def show(fig, key=None):
    cfg = {"displayModeBar": False, "responsive": True}
    st.plotly_chart(fig, theme=None, config=cfg, key=key, **_stretch(st.plotly_chart))


def table(df, **kw):
    st.dataframe(df, hide_index=True, **_stretch(st.dataframe), **kw)


def download(df, label_, filename, key):
    st.download_button(label_, df.to_csv(index=False).encode("utf-8"), filename, "text/csv", key=key)


def fmt_t(v):
    if v is None or v != v:
        return "n/a"
    return f"{v / 1000:,.1f} kt" if abs(v) >= 1000 else f"{v:,.0f} t"
