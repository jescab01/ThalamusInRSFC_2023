
import pandas as pd
import pingouin as pg
import numpy as np
from collections import Counter

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

import pickle


## DATA for reference rPLV & KSD
simulations_tag = "PSEmpi_JR_v2-m12d19y2022-t14h.39m.50s"
folder = 'PAPER2\\R1_TH-type&noise\\' + simulations_tag + '\\'

df = pd.read_csv(folder + "results.csv")
# Average out repetitions and subjects
df_groupavg = df.groupby(["model", "th", "cer", "g", "sigmath"]).mean().reset_index()


## DATA for g_explore
simulations_tag = "g_explore-FIG3_N35-initialConditions_v2-m12d19y2022-t19h.29m.26s.pkl"
folder = 'PAPER2\\R3_scenariosCharact\\data\\'

# Load and plot already computed simulations.
with open(folder + simulations_tag, "rb") as input_file:
    g_sel, output = pickle.load(input_file)

param = "g"


# Colours
cmap_s2, cmap_p2 = px.colors.qualitative.Set2, px.colors.qualitative.Pastel2
c1, c2, c3 = cmap_s2[1], cmap_s2[0], cmap_s2[2]  # "#fc8d62", "#66c2a5", "#8da0cb"  # red, green, blue
opacity = 0.7
c4, c5 = "gray", "dimgray" #cmap_s2[-1], cmap_p2[-1]

##       FIGURE     ##################
sp_titles = ["", r"$\eta_{th}=0.022; g=4$", r"$\eta_{th}=0.022; g=34$",
             r"$\eta_{th}=2.2x10^{-8}; g=4$", r"$\eta_{th}=2.2x10^{-8}; g=34$"] + [""]*5*3
sp_titles[10]="Empirical"
fig = make_subplots(rows=4, cols=5, shared_yaxes=False, horizontal_spacing=0.065,
                    specs=[[{}] + [{"secondary_y": True}]*4,[{}] + [{"secondary_y": True}]*4, [{}, {}, {}, {}, {}], [{}, {}, {}, {}, {}]],
                    subplot_titles=sp_titles, )

for i, g in enumerate(g_sel):

    sl = True if i < 1 else False

    # Unpack output
    signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient, SC_cortex_idx = output[i]

    freqs = np.arange(len(signals[0]) / 2)
    freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

    cmap = px.colors.qualitative.Plotly
    sl_groups = list()
    for ii, signal in enumerate(signals):

        regionGroup = regionLabels[ii].split("_")[0]
        if regionGroup in ["Thal", "Frontal", "Temporal", "Parietal", "Occipital"]:

            if regionGroup not in sl_groups:
                sl_groups.append(regionGroup)

            if Counter(sl_groups)[regionGroup] < 3:
                sy = True if regionGroup == "Thal" else False
                sl1 = True if i == 0 else False
                sl_groups.append(regionGroup)

                # demean signals
                signal = signal - np.average(signal)
                # Timeseries
                fig.add_trace(go.Scatter(x=timepoints[2900:5000]/1000, y=signal[2900:5000], name=regionLabels[ii],
                                         legendgroup=regionGroup, opacity=opacity, line=dict(width=1),
                                         showlegend=sl1, marker_color=cmap[ii % len(cmap)]), secondary_y=sy, row=1, col=i+2)
                # Spectra
                freqRange = [2, 40]
                fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
                fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
                fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
                fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft, opacity=opacity, line=dict(width=1),
                                         marker_color=cmap[ii % len(cmap)], legendgroup=regionGroup, name=regionLabels[ii],
                                         showlegend=False), secondary_y=sy, row=2, col=i+2)

    # Functional Connectivity
    fig.add_trace(go.Heatmap(z=plv, x=regionLabels[SC_cortex_idx], y=regionLabels[SC_cortex_idx], colorbar=dict(thickness=4),
                             colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=i+2)

    # dynamical Fuctional Connectivity
    step = 2
    fig.add_trace(go.Heatmap(z=dplv, x=np.arange(transient/1000, len(dplv) * step, step),
                             y=np.arange(transient/1000, len(dplv) * step, step), colorscale='Viridis',
                             colorbar=dict(thickness=8, len=0.425, y=0, x=0.95, yanchor="bottom"),
                             showscale=sl, zmin=0, zmax=1), row=4, col=i+2)

# empirical FC matrices
fig.add_trace(go.Heatmap(z=plv_emp, x=regionLabels[SC_cortex_idx], y=regionLabels[SC_cortex_idx], colorbar=dict(thickness=4), legendgroup="",
                         colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=1)

# dynamical Fuctional Connectivity
dFC_emp = dFC_emp[:len(dplv)][:, :len(dplv)]
fig.add_trace(go.Heatmap(z=dFC_emp, x=np.arange(transient/1000, len(dFC_emp) * step, step),
                         y=np.arange(transient/1000, len(dplv) * step, step), colorscale='Viridis',
                         showscale=False, zmin=0, zmax=1), row=4, col=1)

# Update layout
fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.55, tracegroupgap=10),
                  template="plotly_white", height=800, width=1000, font_family="Arial",
                  xaxis2=dict(title="Time (s)"), xaxis3=dict(title="Time (s)"), xaxis4=dict(title="Time (s)"), xaxis5=dict(title="Time (s)"),
                  xaxis7=dict(title="Frequency (Hz)"), xaxis8=dict(title="Frequency (Hz)"), xaxis9=dict(title="Frequency (Hz)"), xaxis10=dict(title="Frequency (Hz)"),
                  xaxis16=dict(title="Time (s)"), xaxis17=dict(title="Time (s)"), xaxis18=dict(title="Time (s)"), xaxis19=dict(title="Time (s)"), xaxis20=dict(title="Time (s)"),

                  yaxis2=dict(title="Voltage (mV)", showticklabels=False),
                  yaxis3=dict(showticklabels=False), yaxis4=dict(showticklabels=False), yaxis5=dict(showticklabels=False),
                  yaxis6=dict(showticklabels=False), yaxis7=dict(showticklabels=False), yaxis8=dict(showticklabels=False), yaxis9=dict(showticklabels=False),
                  yaxis11=dict(title="Power (dB)<br>cx | th"), yaxis24=dict(title="Time (s)"),
                  yaxis20=dict(showticklabels=False), yaxis21=dict(showticklabels=False), yaxis22=dict(showticklabels=False), yaxis23=dict(showticklabels=False),
                  )

pio.write_html(fig, file=folder + "/PAPER-R3_g_explore.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, file=folder + "/PAPER-R3_g_explore.svg", engine="kaleido")
