'''
Creating figures and calculating statistics for ThalamusResearch

  -  EXPERIMENT 1 :: thalamocortical w/ FC and KSD -
'''

import pandas as pd
import pingouin as pg
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio


# Load data computed in HPC
simulations_tag = "PSEmpi_JR_v2-m12d19y2022-t14h.39m.50s"
folder = 'PAPER2\\R1_TH-type&noise\\' + simulations_tag + '\\'
df = pd.read_csv(folder + "results.csv")

# Average out repetitions
df_avg = df.groupby(["subject", "model", "th", "cer", "g", "sigmath"]).mean().reset_index()


# Define bifurcation
bif_val = 7

df_avg["bif"] = ["prebif" if row.g < bif_val else "postbif" if row.g > bif_val else "critical" for id, row in df_avg.iterrows()]


# Calculate max rPLV per subject and condition
df_max = df_avg.loc[df_avg["g"] > 0].groupby(["subject", "model", "th", "cer", "sigmath", "bif"]).max(["rPLV"]).reset_index()
# df_max.to_csv(folder + "df_max.csv")

# Calculate min KSD per subject and condition
df_KSDmin = df_avg.loc[df_avg["g"] > 0].groupby(["subject", "model", "th", "cer", "sigmath", "bif"]).min(["dFC_KSD"]).reset_index()


# A) STATISTICS on rPLV
## 0. Two Way ANOVAs at each side of bifurcation
res_aov, ass_norm, ass_sph = [], [], []

for bif in ["prebif", "postbif"]:
    df_max_cond = df_max.loc[df_max["bif"] == bif]

    for sigma in set(df_avg.sigmath):
        df_max_cond_noise = df_max.loc[(df_max["sigmath"] == sigma) & (df_max["bif"] == bif)]
        ass_norm.append(pg.normality(df_max_cond_noise, dv="rPLV", group="th"))
        ass_sph.append(pg.sphericity(df_max_cond_noise, dv="rPLV"))

    test = pg.rm_anova(data=df_max_cond, dv="rPLV", within=["th", "sigmath"], subject="subject")
    res_aov.append(test)


## 1. Simple effects of structure:: ANOVA on each level of sigma
# only in prebifurcation as structure was not significant in two way anova
res_aov_simple, ass_norm, ass_sph = [], [], []
for sigma in set(df_avg.sigmath):
    for bif in ["prebif"]:
        df_max_cond = df_max.loc[(df_max["sigmath"] == sigma) & (df_max["bif"] == bif)]

        ass_norm.append(pg.normality(df_max_cond, dv="rPLV", group="th"))
        ass_sph.append(pg.sphericity(df_max_cond, dv="rPLV"))

        test = pg.rm_anova(data=df_max_cond, dv="rPLV", within="th", subject="subject")
        test["sigmath"], test["bif"] = sigma, bif
        res_aov_simple.append(test)


## 2. Multiple comparisons for sigma (3) in prebif and in postbif
res_mc = pd.DataFrame()
for th in set(df_avg.th.values):
    for bif in ["prebif", "postbif"]:
        df_max_cond = df_max.loc[(df_max["th"] == th) & (df_max["bif"] == bif)]
        test = pg.pairwise_tests(df_max_cond, dv="rPLV", within="sigmath", subject="subject", effsize="cohen", parametric=False)
        test["sigmath"], test["bif"], test["th"] = sigma, bif, th
        res_mc = res_mc.append(test)

res_mc = pd.DataFrame(res_mc, columns=list(test.columns.values))
# Multiple comparisons correction
res_mc["p-corr"] = pg.multicomp(res_mc["p-unc"].values, alpha=0.05, method="fdr_bh")[1]



## 3. Compare max rPLV prebif vs postbif
df_max2 = df_max.loc[df_max["bif"]!="critical"].groupby(["bif", "subject"]).max("rPLV").reset_index()
test = pg.pairwise_tests(df_max2, dv="rPLV", within="bif", subject="subject", effsize="cohen", parametric=False)
pg.plot_paired(df_max2, dv="rPLV", within="bif", subject="subject")



# B) STATISTICS on KSD
## 0. Two Way ANOVAs at each side of bifurcation
resksd_aov, ass_norm, ass_sph = [], [], []

for bif in ["prebif", "postbif"]:
    df_min_cond = df_KSDmin.loc[df_KSDmin["bif"] == bif]

    for sigma in set(df_avg.sigmath):
        df_min_cond_noise = df_KSDmin.loc[(df_KSDmin["sigmath"] == sigma) & (df_KSDmin["bif"] == bif)]
        ass_norm.append(pg.normality(df_min_cond_noise, dv="dFC_KSD", group="th"))
        ass_sph.append(pg.sphericity(df_min_cond_noise, dv="dFC_KSD"))

    test = pg.rm_anova(data=df_min_cond, dv="dFC_KSD", within=["th", "sigmath"], subject="subject")
    resksd_aov.append(test)


## 1. Simple effects for structure:: ANOVA on each level of sigma
# Only in prebifurcation, postbif dont show any significant value
resksd_aov_simple = []
for sigma in set(df_avg.sigmath):
    for bif in ["prebif"]:
        df_KSDmin_cond = df_KSDmin.loc[(df_KSDmin["sigmath"] == sigma) & (df_KSDmin["bif"] == bif)]

        # ass_norm.append(pg.normality(df_KSDmin_cond, dv="dFC_KSD", group="th"))
        # ass_sph.append(pg.sphericity(df_KSDmin_cond, dv="dFC_KSD"))

        test = pg.rm_anova(data=df_KSDmin_cond, dv="dFC_KSD", within="th", subject="subject")
        test["sigma"], test["bif"] = sigma, bif
        resksd_aov_simple.append(test)


## 2. Multiple comparisons on sigma (3): wilcoxon (due to small sample size)
resksd_mc = pd.DataFrame()
for th in set(df_avg.th):
    for bif in ["prebif"]:
        df_KSDmin_cond = df_KSDmin.loc[(df_KSDmin["th"] == th) & (df_KSDmin["bif"] == bif)]

        test = pg.pairwise_tests(df_KSDmin_cond, dv="dFC_KSD", within="sigmath", subject="subject", effsize="cohen", parametric=False)
        test["th"], test["bif"] = th, bif
        resksd_mc=resksd_mc.append(test)

res_mc_ksd = pd.DataFrame(resksd_mc, columns=list(test.columns.values) + ["th", "bif"])
# Multiple comparisons correction
resksd_mc["p-corr"] = pg.multicomp(resksd_mc["p-unc"].values, alpha=0.05, method="fdr_bh")[1]


## 3. Comparing active with parcelled th between pre and postbif
df_min2 = df_KSDmin.loc[df_KSDmin["bif"]!="critical"].groupby(["bif", "subject"]).min("dFC_KSD").reset_index()
test = pg.pairwise_tests(df_min2, dv="dFC_KSD", within="bif", subject="subject", effsize="cohen", parametric=False)
pg.plot_paired(df_min2, dv="dFC_KSD", within="bif", subject="subject")

res_mc = pd.DataFrame()
df_cond = df_max.loc[(df_max["th"] == "pTh") & (df_max["sigmath"] == 0.022) & (df_max["bif"] != "critical")]

test = pg.pairwise_tests(df_cond, dv="rPLV", within="bif", subject="subject", effsize="cohen", parametric=False)
pg.plot_paired(df_cond, dv="rPLV", within="bif", subject="subject")

res_mc = pd.DataFrame(res_mc, columns=list(test.columns.values))
# Multiple comparisons correction
res_mc["p-corr"] = pg.multicomp(res_mc["p-unc"].values, alpha=0.05, method="fdr_bh")[1]



#      FIGURE R1        ################
structure_th = ["woTh", "Th",  "pTh"]

df_groupavg = df_avg.groupby(["model", "th", "cer", "g", "sigmath"]).mean().reset_index()

# Colours
cmap_s, cmap_p = px.colors.qualitative.Set1, px.colors.qualitative.Pastel1
cmap_s2, cmap_p2 = px.colors.qualitative.Set2, px.colors.qualitative.Pastel2

c1, c2, c3 = cmap_s2[1], cmap_s2[0], cmap_s2[2]  # "#fc8d62", "#66c2a5", "#8da0cb"  # red, green, blue
opacity = 0.9

max_g_2show, bif_val = 60, 7

fig = make_subplots(rows=5, cols=3, column_widths=[0.2, 0.6, 0.2], shared_xaxes=True,
                    horizontal_spacing=0.125, vertical_spacing=0.03,
                    subplot_titles=[r"$\eta_{th}=2.2x10^{-8}; g<7$", "", r"$\eta_{th}=2.2x10^{-8}; g>7$", "", "", "", "", "", "",
                                    r"$\eta_{th}=0.022; g<7$", "", r"$\eta_{th}=0.022; g>7$", "", "", ""])

for i, th in enumerate(structure_th):
    # ADD LINEPLOTS
    name = "without Thalamus" if th == "woTh" else "single Thalamus" if th == "Th" else "parcelled Thalamus"
    c = c1 if th == "woTh" else c2 if th == "Th" else c3


    # Plot rPLV - Low noise
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigmath"] == 2.2e-08) & (df_avg["g"] < max_g_2show)]
    fig.add_trace(
        go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name=name, legendgroup=th, mode="lines",
                   line=dict(width=4, color=c), opacity=opacity, showlegend=False), row=1, col=2)

    # ADD BOXPLOTS sigma 0
    # datapoints - prebif
    df_sub = df_max.loc[(df_max["th"] == th) & (df_max["sigmath"] == 2.2e-08) & (df_max["bif"] == "prebif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.rPLV.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=1, col=1)
    # datapoints - postbif
    df_sub = df_max.loc[(df_max["th"] == th) & (df_max["sigmath"] == 2.2e-08) & (df_max["bif"] == "postbif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.rPLV.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=1, col=3)

    # Plot dFC_KSD - Low noise
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigmath"] == 2.2e-08) & (df_avg["g"] < max_g_2show)]
    fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, name=name, legendgroup=th, mode="lines",
                              line=dict(dash='solid', color=c, width=2), opacity=opacity, showlegend=False, visible=True),
                   row=2, col=2)

    # ADD BOXPLOTS sigma 0.022
    # datapoints - prebif
    df_sub = df_KSDmin.loc[(df_KSDmin["th"] == th) & (df_KSDmin["sigmath"] == 2.2e-08) & (df_KSDmin["bif"] == "prebif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.dFC_KSD.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=2, col=1)
    # datapoints - postbif
    df_sub = df_KSDmin.loc[(df_KSDmin["th"] == th) & (df_KSDmin["sigmath"] == 2.2e-08) & (df_KSDmin["bif"] == "postbif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.dFC_KSD.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=2, col=3)

    # Plot rPLV - High noise
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigmath"] == 0.022) & (df_avg["g"] < max_g_2show)]
    fig.add_trace(
        go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name=name, legendgroup=th, mode="lines",
                   line=dict(width=4, color=c), opacity=opacity, showlegend=True), row=4, col=2)

    # ADD BOXPLOTS sigma 0.022
    # datapoints - prebif
    df_sub = df_max.loc[(df_max["th"] == th) & (df_max["sigmath"] == 0.022) & (df_max["bif"] == "prebif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.rPLV.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=4, col=1)
    # datapoints - postbif
    df_sub = df_max.loc[(df_max["th"] == th) & (df_max["sigmath"] == 0.022) & (df_max["bif"] == "postbif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.rPLV.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=4, col=3)

    # Plot dFC_KSD - High noise
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigmath"] == 0.022) & (df_avg["g"] < max_g_2show)]
    fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, legendgroup=th, mode="lines",
                              line=dict(dash='solid', color=c, width=2), opacity=opacity, showlegend=False, visible=True),
                   row=5, col=2)

    # ADD BOXPLOTS sigma 0.022
    # datapoints - prebif
    df_sub = df_KSDmin.loc[(df_KSDmin["th"] == th) & (df_KSDmin["sigmath"] == 0.022) & (df_KSDmin["bif"] == "prebif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.dFC_KSD.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=5, col=1)
    # datapoints - postbif
    df_sub = df_KSDmin.loc[(df_KSDmin["th"] == th) & (df_KSDmin["sigmath"] == 0.022) & (df_KSDmin["bif"] == "postbif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.dFC_KSD.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=5, col=3)


c4, c5 = "gray", "dimgray" #cmap_s2[-1], cmap_p2[-1]
# Plot CX bifurcation
fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_cx, name="cortical ROIs",
                         legendgroup="cortical ROIs", mode="lines",
                         line=dict(width=4, color=c4), showlegend=False), row=3, col=2)

fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_cx, name="cortical ROIs",
                         legendgroup="cortical ROIs", mode="lines",
                         line=dict(width=4, color=c4), showlegend=False), row=3, col=2)

# Plot TH bifurcation
fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_th, name="thalamic ROIs",
                         legendgroup="thalamic ROIs", mode="lines",
                         line=dict(width=2, dash='dot', color=c5),
                         showlegend=False), row=3, col=2)

fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_th, name="thalamic ROIs",
                         legendgroup="thalamic ROIs", mode="lines",
                         line=dict(width=2, dash='dot', color=c5),
                         showlegend=False), row=3, col=2)

## ADD scenarios references
fig.add_shape(x0=0, x1=bif_val, y0=0, y1=0.5, row=1, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)
fig.add_shape(x0=0, x1=bif_val, y0=0, y1=1, row=2, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)
fig.add_shape(x0=0, x1=bif_val, y0=0, y1=0.5, row=4, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)
fig.add_shape(x0=0, x1=bif_val, y0=0, y1=1, row=5, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)

fig.update_layout(template="plotly_white",  # title="Importance of thalamus parcellation and input (10 subjects)",
                  yaxis1=dict(title=r"$\text{max. } r_{PLV}$"),  # , range=[0.25, 0.65]),
                  yaxis2=dict(title="$r_{PLV}$", color="black"),  #, range=[0, 0.5]),
                  yaxis3=dict(title=r"$\text{max. } r_{PLV}$"),  #, range=[0.25, 0.65]),

                  yaxis4=dict(title=r"$\text{min. KSD}$"),
                  yaxis5=dict(title="$KSD$"),  # , range=[0.35, 1]),
                  yaxis6=dict(title=r"$\text{min. KSD}$"),

                  yaxis8=dict(title="Voltage"),  #, range=[0, 0.5]),

                  yaxis10=dict(title=r"$\text{max. } r_{PLV}$"),  # , range=[0.25, 0.65]),
                  yaxis11=dict(title="$r_{PLV}$", color="black"),  # , range=[0, 0.5]),
                  yaxis12=dict(title=r"$\text{max. } r_{PLV}$"),  # , range=[0.25, 0.65]),

                  yaxis13=dict(title=r"$\text{min. KSD}$"),
                  yaxis14=dict(title="$KSD$"),  # , range=[0.35, 1]),
                  yaxis15=dict(title=r"$\text{min. KSD}$"),

                  xaxis14=dict(title="Coupling factor (g)"),
                  legend=dict(orientation="h", xanchor="right", x=0.8, yanchor="bottom", y=1.05),
                  boxgroupgap=0.5, width=900, height=700, font_family="Arial")


pio.write_image(fig, file=folder + "/PAPER-R1_lineSpace-boxplots_PLV-KSD.svg", engine="kaleido")
pio.write_html(fig, file=folder + "/PAPER-R1_lineSpace-boxplots_PLV-KSD.html", auto_open=True, include_mathjax="cdn")

