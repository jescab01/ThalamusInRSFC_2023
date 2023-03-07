
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import numpy as np

simulations_tag = "PSEmpi_adjustingrange_allnodesNoise_v2-m01d05y2023-t09h.20m.01s"
folder = 'PAPER2\\R4.supp_sigmaCX\\' + simulations_tag + '\\'
df = pd.read_csv(folder + "results.csv")

# Average out repetitions
df_avg = df.groupby(["subject", "model", "th", "cer", "g", "pth", "sigmath", "pcx", "sigmacx"]).mean().reset_index()

mask = (df_avg.sigmath == df_avg.sigmacx).values

## Calculate snr
df_avg["snr"] = np.log((df_avg.max_th - df_avg.min_th)/df_avg.sigmath)

# size_min, size_max = 1, 12
# df_avg["plv_sd_size"] = ((size_max-size_min)/(df_avg.plv_sd.max - df_avg.plv_sd.min)) * (df_avg.plv_sd - df_avg.plv_sd.min) + size_min

subject = "NEMOS_035"
# for i, subject in enumerate(list(set(df_avg.subject))):

## FIG 4.supp - main
fig = make_subplots(rows=2, cols=4, horizontal_spacing=0.075, vertical_spacing=0.2,
                    subplot_titles=[r"$r_{PLV}$", "PLV mean", "FFT peak", "Bifurcations (cx)"],
                    x_title=r'$\text{Gaussian std. of cortical input } (\eta_{cx})$', y_title=r"$\text{Coupling factor (g)}$",
                    shared_yaxes=True, shared_xaxes=False)

x_type = "log"
for j in range(2):

    subset = df_avg.loc[mask] if j == 0 else df_avg.loc[np.invert(mask)]
    sl = True if j == 0 else False
    cb_y = 0.79 if j == 0 else 0.18
    length = 0.425
    fig.add_trace(go.Heatmap(z=subset.rPLV, x=subset.sigmacx, y=subset.g, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5,
                             showscale=True, colorbar=dict(y=cb_y, len=length, title="r", thickness=4,  x=0.19)), row=(j+1), col=1)

    fig.add_trace(go.Heatmap(z=subset.plv_m, x=subset.sigmacx, y=subset.g, colorscale='Turbo',
                             showscale=True, colorbar=dict(y=cb_y, len=length, title="", thickness=4, x=0.46)), row=(1+j), col=2)

    # fig.add_trace(go.Scatter(x=subset.pth, y=subset.g, mode="markers", line=dict(width=1)),
    #                          marker=dict(symbol="circle-open", size=subset.plv_sd_size), row=(1+j), col=2)

    fig.add_trace(go.Heatmap(z=subset.IAF, x=subset.sigmacx, y=subset.g, colorscale='Turbo', showscale=True,
                             colorbar=dict(y=cb_y, len=length, title="Hz", thickness=4, x=0.73)), row=(j+1), col=3)

    fig.add_trace(go.Heatmap(z=subset.max_cx - subset.min_cx, x=subset.sigmacx, y=subset.g, colorscale='Viridis',
                             showscale=True, zmin=0, zmax=0.14,
                             colorbar=dict(y=cb_y, len=length, title="mV", thickness=4, x=1)), row=(j + 1), col=4)

    # if sl:
    #     fig.add_trace(go.Heatmap(z=subset.snr, x=subset.sigmacx, y=subset.g, colorscale='Geyser',
    #                    showscale=True, colorbar=dict(y=cb_y, len=length, title="log<br>(snr)", thickness=4,  x=1)), row=(j+1), col=4)


fig.update_layout(width=800, height=600, font_family="Arial", xaxis1=dict(type=x_type), xaxis2=dict(type=x_type),
                  xaxis3=dict(type=x_type),
                  xaxis4=dict(type=x_type), xaxis5=dict(type=x_type), xaxis6=dict(type=x_type),
                  xaxis7=dict(type=x_type), xaxis8=dict(type=x_type))

# Edit titles
fig.layout.annotations = fig.layout.annotations + (fig.layout.annotations[4],)
fig.layout.annotations[4].y = -0.05

fig.layout.annotations[6].text = r'$\text{Gaussian std. of input to every node } (\eta)$'
fig.layout.annotations[6].y = 0.57

pio.write_html(fig, file=folder + "/PAPER-sm3_sigmaCX.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, file=folder + "/PAPER-sm3_sigmaCX.svg", engine="kaleido")

