
'''
Creating figures and calculating statistics for ThalamusResearch

  -  EXPERIMENT 1 :: correlation distributions underlying KSD values -
'''

import time
import numpy as np
import scipy.signal
import scipy.stats
import pandas as pd

from tvb.simulator.lab import *
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from mpi4py import MPI
import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import pickle



result = list()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
    ctb_folderOLD = "E:\\LCCN_Local\PycharmProjects\CTB_dataOLD\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import multitapper
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc, kuramoto_order

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    ctb_folder = wd + "CTB_data3/"
    ctb_folderOLD = wd + "CTB_dataOLD/"

    import sys
    sys.path.append(wd)
    from toolbox.fft import multitapper
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc, kuramoto_order


subj_ids = np.arange(1, 2, 1)
subjects = ["subj" + str(id) for id in subj_ids]

models = ["jr"]

structure_th = ["pTh", "Th", "woTh"]

structure_cer = ["pCer"]

coupling_vals = [3, 7, 9, 40]  # 0.5

params = [[subj, model, th, cer, g, 0.09, 0.022, 0.09, 2.2e-8] for subj in subjects
          for model in models for th in structure_th for cer in structure_cer
          for g in coupling_vals]

params = np.asarray(params, dtype=object)
n = params.shape[0]


# Prepare simulation parameters
simLength = 60 * 1000  # ms
samplingFreq = 1000  # Hz
transient = 4000  # ms

for ii, set in enumerate(params):

    tic = time.time()
    print("Rank %i out of %i  ::  %i/%i " % (rank, size, ii + 1, len(params)))

    print(set)
    emp_subj, model, th, cer, g, pth, sigmath, pcx, sigmacx = set

    # STRUCTURAL CONNECTIVITY      #########################################
    # Use "pass" for subcortical (thalamus) while "end" for cortex
    # based on [https://groups.google.com/g/dsi-studio/c/-naReaw7T9E/m/7a-Y1hxdCAAJ]
    n2i_indexes = []  # not to include indexes

    # Thalamus structure
    if th == 'pTh':
        conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2pTh_pass.zip")
    else:
        conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
        if th == 'woTh':
            n2i_indexes = n2i_indexes + [i for i, roi in enumerate(conn.region_labels) if 'Thal' in roi]

    # Cerebellum structre
    if cer == "Cer":
        cer_indexes = [i for i, roi in enumerate(conn.region_labels) if ('Cer' in roi) or ('Ver' in roi)]
        # update weight matrix: summing up cerebellum weights and averaging tract lengths
        # weights right hem
        weights_sum = np.sum(conn.weights[cer_indexes[0::2], :], axis=0)
        conn.weights[cer_indexes[0], :] = weights_sum
        conn.weights[:, cer_indexes[0]] = weights_sum
        # weights left hem
        weights_sum = np.sum(conn.weights[cer_indexes[1::2], :], axis=0)
        conn.weights[cer_indexes[1], :] = weights_sum
        conn.weights[:, cer_indexes[1]] = weights_sum

        # tract lengths right hem
        tracts_avg = np.average(conn.tract_lengths[cer_indexes[0::2], :], axis=0)
        conn.tract_lengths[cer_indexes[0], :] = tracts_avg
        conn.tract_lengths[:, cer_indexes[0]] = tracts_avg
        # tract lengths left hem
        tracts_avg = np.average(conn.tract_lengths[cer_indexes[1::2], :], axis=0)
        conn.tract_lengths[cer_indexes[1], :] = tracts_avg
        conn.tract_lengths[:, cer_indexes[1]] = tracts_avg

        n2i_indexes = n2i_indexes + cer_indexes[2:]

    elif cer == "woCer":
        n2i_indexes = n2i_indexes + [i for i, roi in enumerate(conn.region_labels) if
                                     ('Cer' in roi) or ('Ver' in roi)]

    indexes = [i for i, roi in enumerate(conn.region_labels) if i not in n2i_indexes]
    conn.weights = conn.weights[:, indexes][indexes]
    conn.tract_lengths = conn.tract_lengths[:, indexes][indexes]
    conn.region_labels = conn.region_labels[indexes]
    conn.weights = conn.scaled_weights(mode="tract")

    conn.speed = np.array([15])

    # Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
    cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_2_L',
                     'Frontal_Sup_2_R', 'Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                     'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
                     'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_2_L', 'Frontal_Inf_Orb_2_R',
                     'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                     'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
                     'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                     'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
                     'OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L',
                     'OFCpost_R', 'OFClat_L', 'OFClat_R', 'Insula_L', 'Insula_R',
                     'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Mid_L',
                     'Cingulate_Mid_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                     'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                     'ParaHippocampal_R', 'Calcarine_L',
                     'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                     'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
                     'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
                     'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
                     'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                     'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
                     'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
                     'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
                     'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                     'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                     'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
                     'Temporal_Inf_R']

    # load text with FC rois; check if match SC
    FClabs = list(np.loadtxt(ctb_folder + "FCavg_" + emp_subj + "/roi_labels.txt", dtype=str))
    FC_cortex_idx = [FClabs.index(roi) for roi in
                     cortical_rois]  # find indexes in FClabs that matches cortical_rois
    SClabs = list(conn.region_labels)
    SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]


    # NEURAL MASS MODEL    #########################################################
    sigma_array = np.asarray([sigmath if 'Thal' in roi else sigmacx for roi in conn.region_labels])

    if type(pcx) == str:
        table = pd.read_pickle(ctb_folder + pcx)
        p_array = table["p_array"].loc[(table["subject"] == emp_subj) & (table["th"] == th)].values[0]

    else:
        p_array = np.asarray([pth if 'Thal' in roi else pcx for roi in conn.region_labels])

    if model == "jrd":  # JANSEN-RIT-DAVID
        # Parameters edited from David and Friston (2003).
        m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                               tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                               He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                               tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                               w=np.array([0.8]), c=np.array([135.0]),
                               c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                               c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                               v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                               p=np.array([p_array]), sigma=np.array([sigma_array]))

        # Remember to hold tau*H constant.
        m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
        m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

    else:  # JANSEN-RIT
        # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          p=np.array([p_array]), sigma=np.array([sigma_array]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

    # COUPLING FUNCTION   #########################################
    if model == "jrd":
        coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
    else:
        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))


    # OTHER PARAMETERS   ###
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)

    print("Simulating %s (%is)  ||  PARAMS: g%i sigma%0.2f" % (model, simLength / 1000, g, sigmath))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract gexplore_data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(gexplore_data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    if model == "jrd":
        raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                   (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
    else:
        raw_data = output[0][1][transient:, 0, :, 0].T
    raw_time = output[0][0][transient:]
    regionLabels = conn.region_labels

    # Extract signals of interest ## BE AWARE of the NOT
    # if "cb" not in mode:
    raw_data = raw_data[SC_cortex_idx, :]
    regionLabels = conn.region_labels[SC_cortex_idx]

    # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
    _, _, IAF, module, band_module = multitapper(raw_data, samplingFreq, regionLabels, peaks=True)

    bands = [["3-alpha"], [(8, 12)]]
    # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

    for b in range(len(bands[0])):
        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

        # Obtain Analytical signal
        efPhase = list()
        efEnvelope = list()
        for i in range(len(efSignals)):
            analyticalSignal = scipy.signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.angle(analyticalSignal))
            efEnvelope.append(np.abs(analyticalSignal))

        # Check point
        # from toolbox import timeseriesPlot, plotConversions
        # regionLabels = conn.region_labels
        # timeseriesPlot(raw_data, raw_time, regionLabels)
        # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)

        # CONNECTIVITY MEASURES
        ## PLV
        plv = PLV(efPhase)

        # Load empirical gexplore_data to make simple comparisons
        plv_emp = \
            np.loadtxt(ctb_folder + "FCavg_" + emp_subj + "/" + bands[0][b] + "_plv_avg.txt", delimiter=',')[:,
            FC_cortex_idx][
                FC_cortex_idx]

        # Comparisons
        t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
        t1[0, :] = plv[np.triu_indices(len(plv), 1)]
        t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
        plv_r = np.corrcoef(t1)[0, 1]

        ## dynamical Functional Connectivity
        # Sliding window parameters
        window, step = 4, 2  # seconds

        ## dFC
        dFC = dynamic_fc(raw_data, samplingFreq, transient, window, step, "PLV",
                         filtered=False, lowcut=lowcut, highcut=highcut)

        dFC_emp = np.loadtxt(ctb_folderOLD + "FC_" + emp_subj + "/" + bands[0][b] + "_dPLV4s.txt")

        # Compare dFC vs dFC_emp
        t2 = np.zeros(shape=(2, len(dFC) ** 2 // 2 - len(dFC) // 2))
        t2[0, :] = dFC[np.triu_indices(len(dFC), 1)]
        t2[1, :] = dFC_emp[np.triu_indices(len(dFC), 1)]
        dFC_ksd = scipy.stats.kstest(dFC[np.triu_indices(len(dFC), 1)], dFC_emp[np.triu_indices(len(dFC), 1)])[0]

        ## Metastability: Kuramoto Order Parameter
        ko_std, ko_mean = kuramoto_order(raw_data, samplingFreq, filtered=False, lowcut=lowcut, highcut=highcut)
        ko_emp = np.loadtxt(ctb_folderOLD + "FC_" + emp_subj + "/" + bands[0][b] + "_sdKO.txt")

        ## Gather results
        result.append((emp_subj, model, th, cer, g, pth, sigmath, pcx, sigmacx, plv_r, dFC, dFC_emp, ko_std, ko_emp))

    print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

results_df = pd.DataFrame(np.asarray(result, dtype=object), columns=["emp_subj", "model", "th", "cer", "g", "pth", "sigmath", "pcx", "sigmacx", "plv_r", "dFC", "dFC_emp", "ko_std", "ko_emp"])
results_df.to_pickle("data/KSDprebif_active3.pkl")

# Calculate KSDs from gathered matrices
ksds = []
for j, th in enumerate(structure_th):
    for i, g in enumerate(coupling_vals):
        sub = results_df.loc[(results_df["th"] == th) & (results_df["g"] == g)]
        ksds.append(scipy.stats.kstest(sub["dFC_emp"].values[0][np.triu_indices(len(sub["dFC"].values[0]), 1)].flatten(),
                                       sub["dFC"].values[0][np.triu_indices(len(sub["dFC"].values[0]), 1)].flatten())[0])

## PLOT results
fig = make_subplots(rows=4, cols=3, column_titles=structure_th, shared_xaxes=True, shared_yaxes=True,
                    row_titles=["g==" + str(g) for g in coupling_vals])

for j, th in enumerate(structure_th):
    for i, g in enumerate(coupling_vals):
        sl = True if (j == 0) & (i == 0) else False
        sub = results_df.loc[(results_df["th"] == th) & (results_df["g"] == g)]
        fig.add_trace(go.Histogram(x=sub["dFC_emp"].values[0][np.triu_indices(len(sub["dFC"].values[0]), 1)].flatten(), marker_color="lightgreen",
                                   name="empirical", legendgroup="empirical", showlegend=sl, xbins=dict(size=0.01)), row=i+1, col=j+1)
        fig.add_trace(go.Histogram(x=sub["dFC"].values[0][np.triu_indices(len(sub["dFC"].values[0]), 1)].flatten(), marker_color="lightgray",
                                   name="simulated", legendgroup="simulated", showlegend=sl, xbins=dict(size=0.01)), row=i+1, col=j+1)
        text = "rPLV = " + str(round(sub.plv_r.values[0], 2)) + "<br>KSD = "+str(round(ksds[j*len(coupling_vals)+i], 3))
        fig.add_annotation(x=0.3, y=35, text=text, showarrow=False, row=i+1, col=j+1)

# Overlay both histograms
fig.update_layout(barmode='overlay', template="plotly_white", font_family="Arial",
                  xaxis16=dict(title="Pearson's r", range=[0, 1]),
                  xaxis17=dict(title="Pearson's r", range=[0, 1]),
                  xaxis18=dict(title="Pearson's r", range=[0, 1]),
                  yaxis1=dict(range=[0, 50]),
                  yaxis4=dict(range=[0, 50]),
                  yaxis7=dict(range=[0, 50]),
                  yaxis10=dict(range=[0, 50]),
                  yaxis13=dict(range=[0, 50]),
                  yaxis16=dict(range=[0, 50]))

# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
pio.write_html(fig, file="CHECKS/data/PAPER-sm2_KSDprebif_active3b.html", auto_open=True)
pio.write_image(fig, file="CHECKS/data/PAPER-sm2_KSDprebif_active3b.svg", width=1000, height=700)