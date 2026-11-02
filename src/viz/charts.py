"""Plotly figure builders. Each returns a go.Figure styled with the 'ewaste' dark template."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.viz import theme as T

T.register_plotly()


def _style(fig, title=None, height=420):
    fig.update_layout(template="ewaste", height=height, margin=dict(l=10, r=12, t=64 if title else 28, b=10),
                      title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)) if title else None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None))
    return fig


def india_map(df, color, color_label, size="ewaste_generated_tonnes", title=None, height=560):
    d = df.dropna(subset=[color, size, "latitude", "longitude"])
    fig = px.scatter_geo(d, lat="latitude", lon="longitude", color=color, size=size, size_max=30,
                         hover_name="city", color_continuous_scale=T.SCALE,
                         custom_data=["state", size, color, "year"], labels={color: color_label})
    fig.update_traces(marker=dict(line=dict(width=0.6, color="#0b1220"), opacity=0.9),
                      hovertemplate="<b>%{hovertext}</b>, %{customdata[0]}<br>Latest observation: %{customdata[3]}<br>"
                                    f"{color_label}: " + "%{customdata[2]:,.1f}<br>Generation: %{customdata[1]:,.0f} t/yr<extra></extra>")
    fig.update_geos(scope="asia", lataxis_range=[5.5, 37.5], lonaxis_range=[67, 98.5], projection_type="mercator",
                    showcountries=True, countrycolor="#2a3f63", showland=True, landcolor="#0f1a2e",
                    showocean=True, oceancolor="#0a101c", showcoastlines=True, coastlinecolor="#2a3f63",
                    showframe=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(coloraxis_colorbar=dict(title=dict(text=color_label, font=dict(color=T.MUTED, size=11)), len=0.7))
    return _style(fig, title, height)


def trend_lines(df, y, ylabel, title=None, height=430):
    d = df.sort_values(["city", "year"])
    fig = px.line(d, x="year", y=y, color="city", markers=True, custom_data=["state"])
    fig.update_traces(line=dict(width=2), marker=dict(size=8),
                      hovertemplate="Year %{x}<br>" + ylabel + ": %{y:,.1f}<br>%{customdata[0]}<extra>%{fullData.name}</extra>")
    fig.update_xaxes(dtick=1, title=None)
    fig.update_yaxes(title=ylabel)
    return _style(fig, title, height)


def state_bar(df, value, label, title=None):
    d = df.sort_values(value)
    fig = go.Figure(go.Bar(x=d[value], y=d["state"], orientation="h", marker=dict(color=d[value], colorscale=T.SCALE),
                           text=[f"n={n}" for n in d["cities"]], textposition="outside", cliponaxis=False,
                           customdata=np.stack([d["cities"]], axis=-1),
                           hovertemplate="<b>%{y}</b><br>" + label + ": %{x:.1f}<br>Cities: %{customdata[0]}<extra></extra>"))
    fig.update_xaxes(title=label)
    return _style(fig, title, max(360, 22 * len(d) + 90))


def coverage_by_year_chart(cov):
    fig = go.Figure(go.Bar(x=cov["year"], y=cov["cities"], marker=dict(color=np.where(cov["partial"], T.AMBER, T.TEAL)),
                           hovertemplate="Year %{x}<br>Cities observed: %{y}<extra></extra>"))
    fig.update_xaxes(dtick=1, title=None)
    fig.update_yaxes(title="Cities with an E-waste observation")
    return _style(fig, "Observed coverage by year (amber = current, partial year)", 340)


def years_per_city_hist(cover, min_years):
    c = cover["years_observed"].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=c.index, y=c.values, marker=dict(color=[T.LIME if i >= min_years else "#3b4f73" for i in c.index]),
                           hovertemplate="%{x} observed years<br>%{y} cities<extra></extra>"))
    fig.update_xaxes(title="Distinct years observed per city", dtick=1)
    fig.update_yaxes(title="Cities")
    return _style(fig, f"History depth per city (lime = forecastable, ≥ {min_years} years)", 340)


def forecast_chart(res, city, model_used, as_of_year, height=470):
    c, obs = res["curve"], res["observed"]
    fig = go.Figure()
    fig.add_vrect(x0=as_of_year + 0.5, x1=c["year"].max() + 0.5, fillcolor=T.TEAL, opacity=0.07, line_width=0,
                  annotation_text="Forecast period", annotation_position="top left", annotation_font=dict(color=T.MUTED, size=11))
    alt = {"Linear": T.SKY, "Polynomial": T.VIOLET, "Naive": T.MUTED}
    for k, col in alt.items():
        if k != model_used:
            fig.add_trace(go.Scatter(x=c["year"], y=c[k], mode="lines", name=f"{k} (alternative)",
                                     line=dict(width=1.5, dash="dot", color=col), opacity=0.8,
                                     hovertemplate="%{x}: %{y:,.0f} t/yr<extra>%{fullData.name}</extra>"))
    fit = c[c["year"] <= res["last_year"]]
    fc = c[c["year"] >= res["last_year"]]
    fig.add_trace(go.Scatter(x=fit["year"], y=fit[model_used], mode="lines", name=f"{model_used} fit",
                             line=dict(width=2, dash="dash", color=T.TEAL),
                             hovertemplate="%{x}: %{y:,.0f} t/yr<extra>%{fullData.name}</extra>"))
    fig.add_trace(go.Scatter(x=fc["year"], y=fc[model_used], mode="lines+markers", name=f"{model_used} forecast",
                             line=dict(width=3.5, color=T.LIME), marker=dict(size=7),
                             hovertemplate="%{x}: %{y:,.0f} t/yr<extra>%{fullData.name}</extra>"))
    fig.add_trace(go.Scatter(x=obs["year"], y=obs["ewaste_generated_tonnes"], mode="markers", name="Observed (annualised)",
                             marker=dict(size=12, color="#f1f5f9", line=dict(width=2, color=T.TEAL)),
                             hovertemplate="%{x}: %{y:,.0f} t/yr<extra>Observed</extra>"))
    fig.update_xaxes(dtick=1, title=None)
    fig.update_yaxes(title="E-waste, tonnes per year (annualised estimate)", rangemode="tozero")
    return _style(fig, f"{city}: observed vs forecast e-waste generation", height)


def gap_grouped_bars(d, title=None):
    d = d.sort_values("ewaste_generated_tonnes")
    fig = go.Figure()
    for col, name, color in [("ewaste_generated_tonnes", "Generated", T.TEAL), ("collected_tonnes", "Collected (rate-derived)", T.SKY),
                             ("recycled_tonnes_est", "Recycled (rate-derived estimate)", T.LIME)]:
        fig.add_trace(go.Bar(y=d["city"], x=d[col], name=name, orientation="h", marker=dict(color=color),
                             hovertemplate="<b>%{y}</b><br>" + name + ": %{x:,.0f} t/yr<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Tonnes per year (annualised)")
    return _style(fig, title, max(380, 34 * len(d) + 110))


def gap_scatter(d, pct_col, pct_label, tonnes_col, title=None):
    d = d.dropna(subset=[pct_col, "ewaste_generated_tonnes", tonnes_col])
    fig = px.scatter(d, x="ewaste_generated_tonnes", y=pct_col, size=tonnes_col, color="region", hover_name="city",
                     log_x=True, size_max=34, custom_data=["state", tonnes_col, "year"])
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5, color="#0b1220")),
                      hovertemplate="<b>%{hovertext}</b>, %{customdata[0]}<br>Year %{customdata[2]}<br>Generated: %{x:,.0f} t/yr<br>"
                                    + pct_label + ": %{y:.1f}%<br>Gap quantity: %{customdata[1]:,.0f} t/yr<extra></extra>")
    fig.add_hline(y=d[pct_col].median(), line=dict(color=T.MUTED, dash="dot", width=1), annotation_text="median gap %",
                  annotation_font=dict(color=T.MUTED, size=11))
    fig.add_vline(x=d["ewaste_generated_tonnes"].median(), line=dict(color=T.MUTED, dash="dot", width=1))
    fig.update_xaxes(title="Generated e-waste, tonnes/year (log scale)")
    fig.update_yaxes(title=pct_label + " (%)")
    return _style(fig, title, 470)


def gap_rank_bar(d, pct_col, pct_label, title=None):
    d = d.sort_values(pct_col)
    fig = go.Figure(go.Bar(y=d["city"], x=d[pct_col], orientation="h", marker=dict(color=d[pct_col], colorscale=T.SCALE),
                           hovertemplate="<b>%{y}</b><br>" + pct_label + ": %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(title=pct_label + " (%)", range=[0, 100])
    return _style(fig, title, max(380, 24 * len(d) + 100))


def priority_bar(r, title=None):
    from src.analysis.priority import COLS, LABELS
    r = r.sort_values("priority_score")
    colors = [T.TEAL, T.SKY, T.LIME, T.AMBER]
    fig = go.Figure()
    for c, col in zip(COLS, colors):
        fig.add_trace(go.Bar(y=r["city"], x=r[f"{c}_points"], orientation="h", name=LABELS[c], marker=dict(color=col),
                             hovertemplate="<b>%{y}</b><br>" + LABELS[c] + ": %{x:.1f} pts<extra></extra>"))
    fig.add_trace(go.Scatter(y=r["city"], x=r["priority_score"], mode="text", text=[f"{v:.1f}" for v in r["priority_score"]],
                             textposition="middle right", showlegend=False, hoverinfo="skip", textfont=dict(color=T.TEXT)))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Priority score (0-100), stacked by indicator contribution", range=[0, max(100, r["priority_score"].max() * 1.12) if len(r) else 100])
    return _style(fig, title, max(400, 26 * len(r) + 120))


def contribution_bar(row):
    from src.analysis.priority import COLS, LABELS
    pts = [row[f"{c}_points"] for c in COLS]
    fig = go.Figure(go.Bar(x=pts, y=[LABELS[c] for c in COLS], orientation="h", marker=dict(color=[T.TEAL, T.SKY, T.LIME, T.AMBER]),
                           text=[f"{p:.1f}" for p in pts], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1f} pts<extra></extra>"))
    fig.update_xaxes(title="Points contributed to the score")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, f"{row['city']}: score composition", 300)


def sensitivity_chart(s, title=None):
    s = s.sort_values("base_rank", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["mean_rank"], y=s["city"], mode="markers", name="Mean rank",
                             marker=dict(size=9, color=T.TEAL),
                             error_x=dict(type="data", symmetric=False, array=s["worst_rank"] - s["mean_rank"],
                                          arrayminus=s["mean_rank"] - s["best_rank"], color="rgba(147,165,193,.6)", thickness=1.5),
                             customdata=np.stack([s["best_rank"], s["worst_rank"], s["base_rank"]], axis=-1),
                             hovertemplate="<b>%{y}</b><br>Base rank: %{customdata[2]}<br>Range: %{customdata[0]} to %{customdata[1]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=s["base_rank"], y=s["city"], mode="markers", name="Rank with your weights",
                             marker=dict(size=9, symbol="diamond", color=T.LIME), hoverinfo="skip"))
    fig.update_yaxes(type="category")
    fig.update_xaxes(title="Rank under 500 perturbed weight sets (1 = highest priority)")
    return _style(fig, title, max(380, 26 * len(s) + 110))


def radar(pct_df, axes, title=None):
    fig = go.Figure()
    for i, (_, r) in enumerate(pct_df.iterrows()):
        vals = [r[a] for a in axes]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=axes + axes[:1], name=r["city"], fill="toself", opacity=0.45,
                                      line=dict(width=2, color=T.COLORWAY[i % len(T.COLORWAY)]),
                                      hovertemplate="%{theta}: %{r:.0f}th percentile<extra>" + r["city"] + "</extra>"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor=T.GRID, linecolor=T.GRID,
                                                                           tickfont=dict(color=T.MUTED, size=10)),
                                 angularaxis=dict(gridcolor=T.GRID, linecolor=T.GRID, tickfont=dict(color=T.TEXT, size=11))))
    return _style(fig, title, 470)


def grouped_city_bars(d, value, label, title=None):
    fig = px.bar(d.sort_values(value, ascending=False), x="city", y=value, color="city", custom_data=["state"])
    fig.update_traces(hovertemplate="<b>%{x}</b> (%{customdata[0]})<br>" + label + ": %{y:,.1f}<extra></extra>")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=label)
    return _style(fig, title, 340)


def model_skill_chart(summary):
    s = summary.copy()
    fig = go.Figure(go.Bar(x=s["model"], y=s["median_abs_pct_error"], marker=dict(color=[T.TEAL, T.VIOLET, T.MUTED][:len(s)]),
                           text=[f"{v:.1f}%" for v in s["median_abs_pct_error"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>Median absolute % error: %{y:.1f}%<extra></extra>"))
    fig.update_yaxes(title="Median absolute % error (lower is better)")
    return _style(fig, "Hold-out error by model (last observed year of each city)", 360)


def holdout_scatter(h):
    fig = go.Figure()
    for k, col in [("Linear", T.TEAL), ("Polynomial", T.VIOLET), ("Naive", T.MUTED)]:
        fig.add_trace(go.Scatter(x=h["actual"], y=h[k], mode="markers", name=k, marker=dict(size=8, color=col, opacity=0.8),
                                 text=h["city"], hovertemplate="<b>%{text}</b><br>Actual: %{x:,.0f}<br>Predicted: %{y:,.0f}<extra>" + k + "</extra>"))
    lim = [max(1, h["actual"].min() * 0.8), h["actual"].max() * 1.1]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction", line=dict(color="#f1f5f9", dash="dot", width=1)))
    fig.update_xaxes(type="log", title="Actual (t/yr)")
    fig.update_yaxes(type="log", title="Predicted (t/yr)")
    return _style(fig, "Predicted vs actual on held-out year (log scale)", 400)
