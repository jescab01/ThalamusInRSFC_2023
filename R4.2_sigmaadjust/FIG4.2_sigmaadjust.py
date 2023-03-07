
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import numpy as np

simulations_tag = "PSEmpi_adjustrange_sigma-m12d20y2022-t04h.08m.02s"
folder = 'PAPER2\\R4.2_sigmaadjust\\' + simulations_tag + '\\'
df = pd.read_csv(folder + "results.csv")

df_avg = df.groupby(["subject", "model", "th", "cer", "g", "pth", "sigmath"]).mean().reset_index()

## Calculate snr
df_avg["snr"] = np.log((df_avg.max_th - df_avg.min_th)/df_avg.sigmath)

subject = "NEMOS_035"

fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.09, vertical_spacing=0.15,
                    subplot_titles=[r"$r_{PLV}$", "mean PLV", "std PLV", "FFT peak", "SNR (th)"],
                    x_title=r'$\text{Gaussian std. of thalamic input } (\eta_{th})$', y_title=r"$\text{Coupling factor (g)}$",
                    shared_yaxes=True, shared_xaxes=False)

cb_y0 = 0.79
cb_y1 = 0.22
length = 0.425

subset = df_avg.loc[(df_avg["subject"] == subject) & (df_avg["th"] == "pTh")]

fig.add_trace(go.Heatmap(z=subset.rPLV, x=subset.sigmath, y=subset.g, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5,
                         showscale=True, colorbar=dict(title="r", thickness=4, len=length, y=cb_y0, x=0.265)), row=1, col=1)

fig.add_trace(go.Heatmap(z=subset.plv_m, x=subset.sigmath, y=subset.g, colorscale='Turbo',
                         showscale=True, colorbar=dict(title="", thickness=4, len=length, y=cb_y0, x=0.63)), row=1, col=2)

fig.add_trace(go.Heatmap(z=subset.plv_sd, x=subset.sigmath, y=subset.g, colorscale='Turbo',
                         showscale=True, colorbar=dict(title="", thickness=4, len=length, y=cb_y0, x=0.99)), row=1, col=3)


fig.add_trace(go.Heatmap(z=subset.IAF, x=subset.sigmath, y=subset.g, colorscale='Turbo',
               showscale=True, colorbar=dict(title="Hz", thickness=4, len=length, y=cb_y1, x=0.265)), row=2, col=1)

fig.add_trace(go.Heatmap(z=subset.snr, x=subset.sigmath, y=subset.g, colorscale='Geyser',
               showscale=True, colorbar=dict(title="log<br>(snr)", thickness=4, len=length, y=cb_y1,  x=0.63)), row=2, col=2)


fig.add_vline(x=0.05, col=[1], line_width=1, line_dash="dot", line_color="lightgray", opacity=0.6)
fig.add_vline(x=0.15, col=[1], line_width=1, line_dash="dot", line_color="lightgray", opacity=0.6)
fig.add_vline(x=0.05, col=[3], line_width=1, line_dash="dot", line_color="gray", opacity=0.6)
fig.add_vline(x=0.15, col=[3], line_width=1, line_dash="dot", line_color="gray", opacity=0.6)

fig.add_vline(x=0.05, col=[2], row=[1], line_width=1, line_dash="dot", line_color="lightgray", opacity=0.6)
fig.add_vline(x=0.15, col=[2], row=[1], line_width=1, line_dash="dot", line_color="lightgray", opacity=0.6)
fig.add_vline(x=0.05, col=[2], row=[2], line_width=1, line_dash="dot", line_color="gray", opacity=0.6)
fig.add_vline(x=0.15, col=[2], row=[2], line_width=1, line_dash="dot", line_color="gray", opacity=0.6)

fig.update_layout(width=700, height=700, font_family="Arial", xaxis1=dict(type="log"), xaxis2=dict(type="log"),
                  xaxis3=dict(type="log"), xaxis4=dict(type="log"), xaxis5=dict(type="log"))

pio.write_html(fig, file=folder + "/PAPER-R4.2-sigmaadjust.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, file=folder + "/PAPER-R4.2-sigmaadjust.svg", engine="kaleido")

folder = "E:\jescab01.github.io\\research\\th\\figs"
pio.write_html(fig, file=folder + "/PAPER-R4.2-sigmaadjust.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, file=folder + "/PAPER-R4.2-sigmaadjust.svg", engine="kaleido")


