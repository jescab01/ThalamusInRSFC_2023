
import time
import pandas as pd
import numpy as np
import scipy.signal
import scipy.stats

import matplotlib.pyplot as plt
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import plotly.offline

from tvb.simulator.lab import *
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from tvb.simulator.models.JansenRit_WilsonCowan import JansenRit_WilsonCowan

import sys
sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.fft import multitapper
from toolbox.signals import epochingTool
from toolbox.mixes import timeseries_spectra
from toolbox.fc import PLV
from toolbox.dynamics import dynamic_fc, kuramoto_order


def simulate(emp_subj, model, g, g_wc=None, pth=0.15, sigmath=0.22, pcx=0.09, sigmacx=0, p_array=None, th='pTh', cer='pCer', t=10, stimulate=False, mode="sim", verbose=True):

    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
    ctb_folderOLD = "E:\\LCCN_Local\PycharmProjects\CTB_dataOLD\\"

    # Prepare simulation parameters
    simLength = t * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 2000  # ms

    tic = time.time()

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
    cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                     'Insula_L', 'Insula_R',
                     'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                     'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                     'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                     'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                     'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                     'Thalamus_L', 'Thalamus_R']

    # load text with FC rois; check if match SC
    FClabs = list(np.loadtxt(ctb_folder + "FCavg_" + emp_subj + "/roi_labels.txt", dtype=str))
    FC_cortex_idx = [FClabs.index(roi) for roi in
                     cortical_rois]  # find indexes in FClabs that matches cortical_rois
    SClabs = list(conn.region_labels)
    SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]


    #   NEURAL MASS MODEL  &  COUPLING FUNCTION   #########################################################

    sigma_array = np.asarray([sigmath if 'Thal' in roi else sigmacx for roi in conn.region_labels])

    if p_array is None:
        if pcx == "MLR":
            SC_Th_idx = [SClabs.index(roi) for roi in conn.region_labels if "Thal" in roi]

            degree_fromth = np.sum(conn.weights[:, SC_Th_idx], axis=1)
            degree_fromth_avg = np.average(np.sum(conn.weights[:, SC_Th_idx], axis=1))

            degree = np.sum(conn.weights, axis=1)
            degree_avg = np.average(degree)

            p_array = 0.09 + g * (-0.0003 * (degree - degree_avg) + -0.003 * (degree_fromth - degree_fromth_avg))

            p_array[SC_Th_idx] = pth

        else:
            p_array = np.asarray([pth if 'Thal' in roi else pcx for roi in conn.region_labels])

    elif type(p_array) == str:
        table = pd.read_pickle(ctb_folder + p_array)
        p_array = table["p_array"].loc[(table["subject"] == emp_subj) & (table["th"] == th)].values[0]


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

        coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=np.array([0.8]), e0=np.array([0.005]),
                                                v0=np.array([6.0]), r=np.array([0.56]))

    elif model == "jrwc":  # JANSEN-RIT(cx) + WILSON-COWAN(th)

        jrMask_wc = [[False] if 'Thal' in roi else [True] for roi in conn.region_labels]

        m = JansenRit_WilsonCowan(
            # Jansen-Rit nodes parameters. From Stefanovski et al. (2019)
            He=np.array([3.25]), Hi=np.array([22]),
            tau_e=np.array([10]), tau_i=np.array([20]),
            c=np.array([135.0]), p=np.array([0.09]),
            c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
            c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
            v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
            # Wilson-Cowan nodes parameters. From Abeysuriya et al. (2018)
            P=np.array([p_th]), sigma=np.array([sigma]), Q=np.array([0]),
            c_ee=np.array([3.25]), c_ei=np.array([2.5]),
            c_ie=np.array([3.75]), c_ii=np.array([0]),
            tau_e_wc=np.array([10]), tau_i_wc=np.array([20]),
            a_e=np.array([4]), a_i=np.array([4]),
            b_e=np.array([1]), b_i=np.array([1]),
            c_e=np.array([1]), c_i=np.array([1]),
            k_e=np.array([1]), k_i=np.array([1]),
            r_e=np.array([0]), r_i=np.array([0]),
            theta_e=np.array([0]), theta_i=np.array([0]),
            alpha_e=np.array([1]), alpha_i=np.array([1]),
            # JR mask | WC mask
            jrMask_wc=np.asarray(jrMask_wc))

        m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])

        coup = coupling.SigmoidalJansenRit_Linear(
            a=np.array([g]), e0=np.array([0.005]), v0=np.array([6]), r=np.array([0.56]), # Jansen-Rit Sigmoidal coupling
            a_linear=np.asarray([g_wc]),  # Wilson-Cowan Linear coupling
            jrMask_wc=np.asarray(jrMask_wc))  # JR mask | WC mask

    else:  # JANSEN-RIT
        # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          p=np.array([p_array]), sigma=np.array([sigma_array]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))


    # OTHER PARAMETERS   ###
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)
    if verbose:
        print("Simulating %s (%is)  ||  PARAMS: g%i sigma%0.2f" % (model, simLength / 1000, g, sigmath))

    # Run simulation
    if stimulate:
        stim_type, gain, nstates, tstates, pinclusion, deterministic = stimulate

        if deterministic == "config":
            times = sorted(np.random.randint(0, simLength * 1000, nstates))
            deterministic = [[times[i], times[i + 1]] for i in range(len(times) - 1)]
            deterministic.append([times[-1], simLength])

        stimulus = configure_states(stim_type, gain, nstates, tstates, pinclusion, conn, simLength, deterministic)
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon, stimulus=stimulus)
    else:
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    if model == "jrd":
        raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                   (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)

    else:
        raw_data = output[0][1][transient:, 0, :, 0].T
    raw_time = output[0][0][transient:]

    # Further analysis on Cortical signals
    raw_data = raw_data[SC_cortex_idx, :]
    regionLabels = conn.region_labels[SC_cortex_idx]

    # PLOTs :: Signals and spectra
    # timeseries_spectra(raw_data[:], simLength, transient, regionLabels, mode="inline", freqRange=[2, 40], opacity=1)
    if mode == "pHetero":
        return output[0][1][transient:, 0, :, 0].T, output[0][0][transient:], conn.region_labels
    else:
        bands = [["3-alpha"], [(8, 12)]]
        # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

        for b in range(len(bands[0])):
            (lowcut, highcut) = bands[1][b]

            # Band-pass filtering
            filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut, verbose=False)

            # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
            efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals", verbose=False)

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
            ## PLV and plot
            plv = PLV(efPhase, verbose=False)

            # Load empirical data to make simple comparisons
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

            ## dFC and plot
            if mode == "FC":
                dFC, matrices_fc = dynamic_fc(raw_data, samplingFreq, transient, window, step, "PLV",
                                              filtered=False, lowcut=lowcut, highcut=highcut, mode="all_matrices")
            else:
                dFC = dynamic_fc(raw_data, samplingFreq, transient, window, step, "PLV",
                                 filtered=False, lowcut=lowcut, highcut=highcut)

            dFC_emp = np.loadtxt(ctb_folderOLD + "FC_" + emp_subj + "/" + bands[0][b] + "_dPLV4s.txt")

            # Define n_points to compare
            n_points = len(dFC) if len(dFC) < len(dFC_emp) else len(dFC_emp)

            # Compare dFC vs dFC_emp
            t2 = np.zeros(shape=(2, n_points ** 2 // 2 - n_points // 2))
            t2[0, :] = dFC[np.triu_indices(n_points, 1)]
            t2[1, :] = dFC_emp[np.triu_indices(n_points, 1)]
            dFC_ksd = scipy.stats.kstest(dFC[np.triu_indices(n_points, 1)], dFC_emp[np.triu_indices(n_points, 1)])[0]

            ## Metastability: Kuramoto Order Parameter
            ko_std, ko_mean = kuramoto_order(raw_data, samplingFreq, lowcut=lowcut, highcut=highcut, verbose=False)
            ko_emp = np.loadtxt(ctb_folderOLD + "FC_" + emp_subj + "/" + bands[0][b] + "_sdKO.txt")

            ## PLOTs :: PLV + dPLV
            print("REPORT_ \nrPLV = %0.2f  |  KSD = %0.2f  |  KO_std = %0.2f - emp%0.2f" % (plv_r, dFC_ksd, ko_std, ko_emp))

        print("SIMULATION REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

        if mode == "FC":
            if stimulate:
                SC_th_idx = [SClabs.index(roi) for roi in conn.region_labels if "Thal" in roi]
                pattern = stimulus()
                return matrices_fc, plv, dFC, plv_emp, dFC_emp, regionLabels, simLength, transient, pattern[SC_th_idx, transient:]
            else:
                return matrices_fc, plv, dFC, plv_emp, dFC_emp, regionLabels, simLength, transient

        elif mode == "FIG":
            return output[0][1][transient:, 0, :, 0].T, raw_time, plv, dFC, plv_emp, dFC_emp, conn.region_labels, simLength, transient, SC_cortex_idx

        else:
            if stimulate:
                SC_th_idx = [SClabs.index(roi) for roi in conn.region_labels if "Thal" in roi]
                pattern = stimulus()
                return raw_data, raw_time, plv, dFC, plv_emp, dFC_emp, regionLabels, simLength, transient, pattern[SC_th_idx, transient:]
            else:
                return raw_data, raw_time, plv, dFC, plv_emp, dFC_emp, regionLabels, simLength, transient


def g_explore(output, g_sel, param="g", mode="html", folder="figures"):

    if len(output[0]) == 9:

        n_g = len(g_sel)
        col_titles = [""] + [param + "==" + str(g) for g in g_sel]
        specs = [[{} for g in range(n_g+1)]]*4
        id_emp = (n_g + 1) * 2
        sp_titles = ["Empirical" if i == id_emp else "" for i in range((n_g+1)*4)]
        fig = make_subplots(rows=4, cols=n_g+1, specs=specs, row_titles=["signals", "FFT", "FC", "dFC"],
                            column_titles=col_titles, shared_yaxes=True, subplot_titles=sp_titles)

        for i, g in enumerate(g_sel):

            sl = True if i < 1 else False

            # Unpack output
            signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient = output[i]

            freqs = np.arange(len(signals[0]) / 2)
            freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

            cmap = px.colors.qualitative.Plotly
            for ii, signal in enumerate(signals):

                # Timeseries
                fig.add_trace(go.Scatter(x=timepoints[:5000]/1000, y=signal[:5000], name=regionLabels[ii],
                                         legendgroup=regionLabels[ii],
                                         showlegend=sl, marker_color=cmap[ii % len(cmap)]), row=1, col=i+2)
                # Spectra
                freqRange = [2, 40]
                fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
                fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
                fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
                fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                         marker_color=cmap[ii % len(cmap)], name=regionLabels[ii],
                                         legendgroup=regionLabels[ii], showlegend=False), row=2, col=i+2)

            # Functional Connectivity
            fig.add_trace(go.Heatmap(z=plv, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4),
                                     colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=i+2)

            # dynamical Fuctional Connectivity
            step = 2
            fig.add_trace(go.Heatmap(z=dplv, x=np.arange(transient/1000, len(dplv) * step, step),
                                     y=np.arange(transient/1000, len(dplv) * step, step), colorscale='Viridis',
                                     colorbar=dict(thickness=8, len=0.4, y=0, yanchor="bottom"),
                                     showscale=sl, zmin=0, zmax=1), row=4, col=i+2)

        # empirical FC matrices
        fig.add_trace(go.Heatmap(z=plv_emp, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4), legendgroup="",
                                 colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=1)

        # dynamical Fuctional Connectivity
        dFC_emp=dFC_emp[:len(dplv)][:, :len(dplv)]
        fig.add_trace(go.Heatmap(z=dFC_emp, x=np.arange(transient/1000, len(dFC_emp) * step, step),
                                 y=np.arange(transient/1000, len(dplv) * step, step), colorscale='Viridis',
                                 showscale=False, zmin=0, zmax=1), row=4, col=1)

        w_ = 800 if n_g < 3 else 1000
        fig.update_layout(legend=dict(yanchor="top", y=1.05, tracegroupgap=1),
                          template="plotly_white", height=900, width=w_)

        # Update layout
        for col in range(n_g+1):  # +1 empirical column
            # first row
            idx = col + 1  # +1 to avoid 0 indexing in python
            if idx > 1:
                fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Time (s)"}
                if idx == 2:
                    fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Voltage (mV)"}

            # second row
            idx = 1 * (n_g+1) + (col+1)  # +1 to avoid 0 indexing in python
            if idx > 1 + n_g:
                fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Frequency (Hz)"}
                if idx == 3 + n_g:
                    fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Power (dB)"}

            # third row
            # idx = 2 * n_g+1 + (col+1)  # +1 to avoid 0 indexing in python
            # fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}
            # fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}

            # fourth row
            idx = 3 * (n_g+1) + (col+1)  # +1 to avoid 0 indexing in python
            fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'Time (s)'}
            if idx == (3 * (n_g+1) + 1):
                fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'Time (s)'}

        if mode == "html":
            pio.write_html(fig, file=folder + "/PAPER3_g_explore.html", auto_open=True)
        elif mode == "png":
            pio.write_image(fig, file=folder + "/g_explore" + str(time.time()) + ".png", engine="kaleido")
        elif mode == "svg":
            pio.write_image(fig, file=folder + "/g_explore.svg", engine="kaleido")

        elif mode == "inline":
            plotly.offline.iplot(fig)

    elif len(output[0]) == 10:

        n_g = len(g_sel)
        col_titles = [""] + [param + "==" + str(g) for g in g_sel]
        specs = [[{} for g in range(n_g+1)]]*5
        id_emp = (n_g + 1) * 2
        sp_titles = ["Empirical" if i == id_emp else "" for i in range((n_g+1)*4)]
        fig = make_subplots(rows=5, cols=n_g+1, specs=specs, row_titles=["signals", "FFT", "FC", "dFC", "TH-inputs"],
                            column_titles=col_titles, shared_yaxes=True, subplot_titles=sp_titles)

        bar_stim = np.max([np.max(np.abs(set_[-1])) for set_ in output])

        for i, g in enumerate(g_sel):

            sl = True if i < 1 else False

            # Unpack output
            signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient, stimulus = output[i]

            freqs = np.arange(len(signals[0]) / 2)
            freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

            cmap = px.colors.qualitative.Plotly
            for ii, signal in enumerate(signals):
                # Timeseries
                fig.add_trace(go.Scatter(x=timepoints[:5000] / 1000, y=signal[:5000], name=regionLabels[ii],
                                         legendgroup=regionLabels[ii],
                                         showlegend=sl, marker_color=cmap[ii % len(cmap)]), row=1, col=i + 2)
                # Spectra
                freqRange = [2, 40]
                fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
                fft = np.asarray(
                    fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
                fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
                fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                         marker_color=cmap[ii % len(cmap)], name=regionLabels[ii],
                                         legendgroup=regionLabels[ii], showlegend=False), row=2, col=i + 2)

            # Functional Connectivity
            fig.add_trace(go.Heatmap(z=plv, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4),
                                     colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=i + 2)

            # dynamical Fuctional Connectivity
            step = 2
            fig.add_trace(go.Heatmap(z=dplv, x=np.arange(transient / 1000, len(dplv) * step, step),
                                     y=np.arange(transient / 1000, len(dplv) * step, step), colorscale='Viridis',
                                     colorbar=dict(thickness=8, len=0.25, y=0.2, yanchor="bottom"),
                                     showscale=sl, zmin=0, zmax=1), row=4, col=i + 2)

            # stimulation pattern
            fig.add_trace(go.Heatmap(z=stimulus, x=timepoints/1000, y=list(range(len(stimulus))),
                                     colorbar=dict(thickness=8, len=0.15, y=-0.02, yanchor="bottom"),
                                     colorscale='IceFire', reversescale=True, zmin=-bar_stim, zmax=bar_stim), row=5, col=i+2)

        # empirical FC matrices
        fig.add_trace(go.Heatmap(z=plv_emp, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4),
                                 colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=1)

        # dynamical Fuctional Connectivity
        dFC_emp = dFC_emp[:len(dplv)][:, :len(dplv)]
        fig.add_trace(go.Heatmap(z=dFC_emp, x=np.arange(transient / 1000, len(dFC_emp) * step, step),
                                 y=np.arange(transient / 1000, len(dplv) * step, step), colorscale='Viridis',
                                 showscale=False, zmin=0, zmax=1), row=4, col=1)

        w_ = 800 if n_g < 3 else 1000
        fig.update_layout(legend=dict(yanchor="top", y=1.05, tracegroupgap=1),
                          template="plotly_white", height=1100, width=w_)

        # Update layout
        for col in range(n_g + 1):  # +1 empirical column
            # first row
            idx = col + 1  # +1 to avoid 0 indexing in python
            if idx > 1:
                fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Time (s)"}
                if idx == 2:
                    fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Voltage (mV)"}

            # second row
            idx = 1 * (n_g + 1) + (col + 1)  # +1 to avoid 0 indexing in python
            if idx > 1 + n_g:
                fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Frequency (Hz)"}
                if idx == 3 + n_g:
                    fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Power (dB)"}

            # third row
            # idx = 2 * n_g+1 + (col+1)  # +1 to avoid 0 indexing in python
            # fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}
            # fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}

            # fourth row
            idx = 3 * (n_g + 1) + (col + 1)  # +1 to avoid 0 indexing in python
            fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'Time (s)'}
            if idx == (3 * (n_g + 1) + 1):
                fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'Time (s)'}

        if mode == "html":
            pio.write_html(fig, file=folder + "/PAPER3_g_explore.html", auto_open=True)
        elif mode == "png":
            pio.write_image(fig, file=folder + "/g_explore" + str(time.time()) + ".png", engine="kaleido")
        elif mode == "svg":
            pio.write_image(fig, file=folder + "/g_explore.svg", engine="kaleido")

        elif mode == "inline":
            plotly.offline.iplot(fig)



