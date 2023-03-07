
import pandas as pd
import pingouin as pg

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

simulations_tag = "PSEmpi_JR_v2-m12d19y2022-t14h.39m.50s"
folder = 'PAPER2\\R1_TH-type&noise\\' + simulations_tag + '\\'
df_th = pd.read_csv(folder + "results.csv")

simulations_tag = "PSEmpi_JRstd0.022CER-m12d19y2022-t23h.58m.26s"
folder = 'PAPER2\\R2_CER-type&noise\\' + simulations_tag + '\\'
df_cer = pd.read_csv(folder + "results.csv")


# Average out repetitions
df_th_avg = df_th.groupby(["subject", "model", "th", "cer", "g", "sigmath"]).mean().reset_index()
df_cer_avg = df_cer.groupby(["subject", "model", "th", "cer", "g", "sigmath"]).mean().reset_index()

df_th_groupavg = df_th_avg.groupby(["model", "th", "cer", "g", "sigmath"]).mean().reset_index()
df_cer_groupavg = df_cer_avg.groupby(["model", "th", "cer", "g", "sigmath"]).mean().reset_index()

# Calculate max rPLV PRE-BIF per subject and condition
prebif_lowg = 0
prebif_topg = 7
df_th_maxprebif = df_th_avg.loc[(df_th_avg["g"] < prebif_topg) & (df_th_avg["g"] > prebif_lowg) & (df_th_avg["sigmath"] == 0.022)].groupby(["subject", "model", "th", "cer", "sigmath"]).max(["rPLV"]).reset_index()
df_cer_maxprebif = df_cer_avg.loc[(df_cer_avg["g"] < prebif_topg) & (df_cer_avg["g"] > prebif_lowg) & (df_cer_avg["sigmath"] == 0.022)].groupby(["subject", "model", "th", "cer", "sigmath"]).max(["rPLV"]).reset_index()


# STATISTICS on rPLV
## Simple effects:: ANOVA on each level of sigma
res_aov, ass_norm, ass_sph = [], [], []

ass_norm.append(pg.normality(df_cer_maxprebif, dv="rPLV", group="th"))
ass_sph.append(pg.sphericity(df_cer_maxprebif, dv="rPLV"))

test_aov = pg.rm_anova(data=df_cer_maxprebif, dv="rPLV", within="cer", subject="subject")

test_wil = pg.pairwise_tests(df_cer_maxprebif, dv="rPLV", within="cer", subject="subject", effsize="cohen", parametric=False)
# Multiple comparisons correction
test_wil["p-corr"] = pg.multicomp(test_wil["p-unc"].values, alpha=0.05, method="fdr_bh")[1]



## STATISTICAL ANALYSIS
## directly wilcoxon (due to small sample)

res_ttests = []

pg.normality(df_th_maxprebif, dv="rPLV", group="th")
pg.normality(df_cer_maxprebif, dv="rPLV", group="cer")

for th, cer in [("woTh", "woCer"), ("Th", "Cer"), ("pTh", "pCer")]:
    g1 = df_th_maxprebif["rPLV"].loc[df_th_maxprebif["th"] == th].values
    g2 = df_cer_maxprebif["rPLV"].loc[(df_cer_maxprebif["cer"] == cer)].values
    test = pg.wilcoxon(x=g1, y=g2)
    res_ttests.append([th, cer] + list(test.values[0]))

res_ttests = pd.DataFrame(res_ttests, columns=["g1", "g2", "W-val", "alt", "p-val", "RBC", "CLES"])

res_ttests["p-corr"] = pg.multicomp(res_ttests["p-val"].values, alpha=0.05, method="fdr_bh")[1]



## PLOT -
structure_th, structure_cer = ["woTh", "Th",  "pTh"], ["woCer", "Cer", "pCer"]

# Colours
cmap_s, cmap_p = px.colors.qualitative.Set1, px.colors.qualitative.Pastel1
cmap_s2, cmap_p2 = px.colors.qualitative.Set2, px.colors.qualitative.Pastel2

c1, c2, c3 = cmap_s2[1], cmap_s2[0], cmap_s2[2]  # "#fc8d62", "#66c2a5", "#8da0cb"  # red, green, blue
c4, c5, c6 = cmap_s2[3], cmap_s2[4], cmap_s2[7]  # "#c994c7", "#df65b0", "#dd1c77"  # purple, pink, fucsia
opacity = 0.7

# Graph objects approach
fig3 = make_subplots(rows=2, cols=2, row_titles=["Thalamus", "Cerebellum"], column_widths=[0.3, 0.7],
                     specs=[[{"rowspan": 2}, {}], [{}, {}]], horizontal_spacing=0.15)

for i, th in enumerate(structure_th):

    group = "wo" if th == "woTh" else "w" if th == "Th" else "wp"

    # ADD LINEPLOTS THALAMUS
    name = "without Thalamus" if th=="woTh" else "single Thalamus" if th=="Th" else "parcelled Thalamus"
    c = c1 if th == "woTh" else c2 if th == "Th" else c3

    # Plot rPLV - active
    df_sub_avg = df_th_groupavg.loc[(df_th_avg["th"] == th) & (df_th_avg["sigmath"] == 0.022)]
    fig3.add_trace(
        go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name=name, legendgroup=group, mode="lines", opacity=opacity,
                   line=dict(width=4, color=c), showlegend=True), row=1, col=2)

    # ADD BOXPLOTS
    # datapoints - Active
    df_sub = df_th_maxprebif.loc[(df_th_maxprebif["th"] == th) & (df_th_maxprebif["sigmath"] == 0.022)]
    fig3.add_trace(go.Box(x=[group]*len(df_sub.th.values), y=df_sub.rPLV.values, fillcolor=c, line=dict(color="black", width=1),
                          legendgroup=group, opacity=opacity, showlegend=False), row=1, col=1)

    # ADD LINEPLOTS CEREBELLUM
    cer = structure_cer[i]
    name = "without Cerebellum" if cer == "woCer" else "single Cerebellum" if cer == "Cer" else "parcelled Cerebellum"
    c = c4 if cer == "woCer" else c5 if cer == "Cer" else c6

    # Plot rPLV - active
    df_sub_avg = df_cer_groupavg.loc[(df_cer_groupavg["cer"] == cer) & (df_cer_groupavg["sigmath"] == 0.022)]
    fig3.add_trace(
        go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name=name, legendgroup=group, mode="lines", opacity=opacity,
                   line=dict(width=4, color=c), showlegend=True), row=2, col=2)

    # ADD BOXPLOTS
    # datapoints - Active
    df_sub = df_cer_maxprebif.loc[(df_cer_maxprebif["cer"] == cer) & (df_cer_maxprebif["sigmath"] == 0.022)]
    fig3.add_trace(go.Box(x=[group]*len(df_sub.cer.values), y=df_sub.rPLV.values, fillcolor=c,
                          line=dict(color="black", width=1), opacity=opacity,
                          legendgroup=group, showlegend=False), row=1, col=1)

fig3.add_shape(x0=prebif_lowg, x1=prebif_topg, y0=0, y1=0.5, row=1, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)
fig3.add_shape(x0=prebif_lowg, x1=prebif_topg, y0=0, y1=0.5, row=2, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)



fig3.update_layout(template="plotly_white", #title="Importance of thalamus parcellation and input (10 subjects)",
                   yaxis1=dict(title=r"$\text{max. } r_{PLV} \text{ prebif}$", color="black"),
                   yaxis2=dict(title="$r_{PLV}$", color="black", range=[0, 0.5]),
                   yaxis4=dict(title="$r_{PLV}$", color="black", range=[0, 0.5]),
                   xaxis4=dict(title="Coupling factor (g)"),
                   legend=dict(orientation="h", xanchor="right", x=0.9, yanchor="bottom", y=1.02, groupclick="toggleitem"),
                   boxmode="group", height=400, width=700, font_family="Arial")


pio.write_html(fig3, file=folder + "/PAPER-R2_THCER_lineSpace-boxplots.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig3, file=folder + "/PAPER-R2_THCER_lineSpace-boxplots.svg")



## Checking bifurcations

fig3 = make_subplots(rows=3, cols=2, row_titles=["Thalamus", "Cerebellum"], column_widths=[0.3, 0.7],
                     specs=[[{"rowspan": 2}, {}], [{}, {}], [{},{}]], horizontal_spacing=0.15)

for i, th in enumerate(structure_th):

    group = "wo" if th == "woTh" else "w" if th == "Th" else "wp"

    # ADD LINEPLOTS THALAMUS
    name = "without Thalamus" if th=="woTh" else "single Thalamus" if th=="Th" else "parcelled Thalamus"
    c = c1 if th == "woTh" else c2 if th == "Th" else c3

    # Plot rPLV - active
    df_sub_avg = df_th_groupavg.loc[(df_th_avg["th"] == th) & (df_th_avg["sigmath"] == 0.022)]
    fig3.add_trace(
        go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name=name, legendgroup=group, mode="lines", opacity=opacity,
                   line=dict(width=4, color=c), showlegend=True), row=1, col=2)

    # ADD BOXPLOTS
    # datapoints - Active
    df_sub = df_th_maxprebif.loc[(df_th_maxprebif["th"] == th) & (df_th_maxprebif["sigmath"] == 0.022)]
    fig3.add_trace(go.Box(x=[group]*len(df_sub.th.values), y=df_sub.rPLV.values, fillcolor=c, line=dict(color="black", width=1),
                          legendgroup=group, opacity=opacity, showlegend=False), row=1, col=1)

    # ADD LINEPLOTS CEREBELLUM
    cer = structure_cer[i]
    name = "without Cerebellum" if cer == "woCer" else "single Cerebellum" if cer == "Cer" else "parcelled Cerebellum"
    c = c4 if cer == "woCer" else c5 if cer == "Cer" else c6

    # Plot rPLV - active
    df_sub_avg = df_cer_groupavg.loc[(df_cer_groupavg["cer"] == cer) & (df_cer_groupavg["sigmath"] == 0.022)]
    fig3.add_trace(
        go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name=name, legendgroup=group, mode="lines", opacity=opacity,
                   line=dict(width=4, color=c), showlegend=True), row=2, col=2)

    fig3.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_cx, name="cortical ROIs",
                             legendgroup="cortical ROIs", mode="lines",
                             line=dict(width=2, color=c), showlegend=False), row=3, col=2)

    fig3.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_cx, name="cortical ROIs",
                             legendgroup="cortical ROIs", mode="lines",
                             line=dict(width=2, color=c), showlegend=False), row=3, col=2)


fig3.add_shape(x0=prebif_lowg, x1=prebif_topg, y0=0, y1=0.5, row=1, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)
fig3.add_shape(x0=prebif_lowg, x1=prebif_topg, y0=0, y1=0.5, row=2, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)


fig3.update_layout(template="plotly_white", #title="Importance of thalamus parcellation and input (10 subjects)",
                   yaxis1=dict(title=r"$\text{max. } r_{PLV} \text{ prebif}$", color="black"),
                   yaxis2=dict(title="$r_{PLV}$", color="black", range=[0, 0.5]),
                   yaxis4=dict(title="$r_{PLV}$", color="black", range=[0, 0.5]),
                   xaxis4=dict(title="Coupling factor (g)"),
                   legend=dict(orientation="h", xanchor="right", x=0.9, yanchor="bottom", y=1.02, groupclick="toggleitem"),
                   boxmode="group", height=400, width=700, font_family="Arial")


pio.write_html(fig3, file="figures/PAPER-R2_THCER_lineSpace-boxplots.html", auto_open=True, include_mathjax="cdn")

