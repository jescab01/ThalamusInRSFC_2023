
'''
Creating figures and calculating statistics for ThalamusResearch

  -  EXPERIMENT 1 :: supplementary individual plots w/ FC and KSD -
'''

import pandas as pd
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Load data computed in HPC
simulations_tag = "PSEmpi_JR_v2-m12d19y2022-t14h.39m.50s"
folder = 'PAPER2\\R1_TH-type&noise\\' + simulations_tag + '\\'
df = pd.read_csv(folder + "/results.csv")

structure_th = ["woTh", "Th", "pTh"]

# Average out repetitions
df_avg = df.groupby(["subject", "model", "th", "cer", "g", "sigmath"]).mean().reset_index()


#      FIGURE 6sm        ################
structure_th = ["woTh", "Th",  "pTh"]

# Colours
cmap_s, cmap_p = px.colors.qualitative.Set1, px.colors.qualitative.Pastel1
cmap_s2, cmap_p2 = px.colors.qualitative.Set2, px.colors.qualitative.Pastel2

c1, c2, c3 = cmap_s2[1], cmap_s2[0], cmap_s2[2]  # "#fc8d62", "#66c2a5", "#8da0cb"  # red, green, blue
opacity = 0.8

subjects = sorted(set(df_avg.subject))

# Create the figure
fig = make_subplots(rows=10, cols=3, horizontal_spacing=0.15,
                     shared_xaxes=True, row_titles=["Subj" + str(i+1) for i, subj in enumerate(subjects)])

for ii, subj in enumerate(subjects):

    for i, th in enumerate(structure_th):

        leg = "without Thalamus" if th == "woTh" else "single Thalamus" if th == "Th" else "parcelled Thalamus"
        sl = True if ii==0 else False

        # ADD LINEPLOTS
        c = c1 if th == "woTh" else c2 if th == "Th" else c3

        # Plot rPLV - active
        df_sub_avg = df_avg.loc[(df_avg["th"] == th) & (df_avg["sigmath"] == 0.022) & (df_avg["subject"] == subj)]
        fig.add_trace(
            go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name=leg, legendgroup=leg, mode="lines",
                       line=dict(width=3, color=c), opacity=opacity, showlegend=sl), row=ii+1, col=1)

        # Plot dFC_KSD - active
        df_sub_avg = df_avg.loc[(df_avg["th"] == th) & (df_avg["sigmath"] == 0.022) & (df_avg["subject"] == subj)]
        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, legendgroup=leg, mode="lines",
                                  line=dict(dash='solid', color=c, width=1.5), opacity=opacity, showlegend=False, visible=True),
                       row=ii+1, col=2)

        c4, c5 = "gray", "dimgray"  # cmap_s2[-1], cmap_p2[-1]

        # Plot CX bifurcation
        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_cx, name=leg,
                                 legendgroup=leg, mode="lines", opacity=0.5,
                                 line=dict(width=4, color=c), showlegend=False), row=ii+1, col=3)

        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_cx, name=leg,
                                 legendgroup=leg, mode="lines", opacity=0.5,
                                 line=dict(width=4, color=c), showlegend=False), row=ii+1, col=3)

        # Plot TH bifurcation
        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_th, name=leg,
                                 legendgroup=leg, mode="lines",
                                 line=dict(width=2, dash='dot', color=c),
                                 showlegend=False), row=ii+1, col=3)

        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_th, name=leg,
                                 legendgroup=leg, mode="lines",
                                 line=dict(width=2, dash='dot', color=c),
                                 showlegend=False), row=ii+1, col=3)

for i in range(1, 31, 3):
    fig["layout"]["yaxis" + str(i)]["range"] = [-0.05, 0.65]
    if i == list(range(1, 31, 3))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "$r_{PLV}$"
    if i == list(range(1, 31, 3))[-1]:
        fig["layout"]["xaxis" + str(i)]["title"] = "Coupling factor (g)"
for i in range(2, 31, 3):
    fig["layout"]["yaxis" + str(i)]["range"] = [0, 1]
    if i == list(range(2, 31, 3))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "KSD"
    if i == list(range(2, 31, 3))[-1]:
        fig["layout"]["xaxis" + str(i)]["title"] = "Coupling factor (g)"
for i in range(3, 31, 3):
    if i == list(range(3, 31, 3))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "Bifurcations<br><b>cx</b> | th"
    if i == list(range(3, 31, 3))[-1]:
        fig["layout"]["xaxis" + str(i)]["title"] = "Coupling factor (g)"
fig.update_layout(template="plotly_white", width=900, height=1100, font_family="Arial",
                   legend=dict(orientation="h", xanchor="right", x=0.95, yanchor="bottom", y=1.04))

pio.write_image(fig, file=folder + "/PAPER-sm1_SUBJECTS-lineSpaces.svg", engine="kaleido")
pio.write_html(fig, file=folder + "/PAPER-sm1_SUBJECTS-lineSpaces.html", auto_open=True, include_mathjax="cdn")