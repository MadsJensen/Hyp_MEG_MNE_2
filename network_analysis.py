"""@author: mje."""
import cPickle as Pickle
import os
import socket

import mne
import networkx as nx
import numpy as np
import numpy.random as npr
from nitime import TimeSeries
from nitime.analysis import CoherenceAnalyzer


# from mne.stats import fdr_correction


# Permutation test.
def permutation_test(a, b, num_samples, statistic):
    """Return p-value that statistic for a is different from statistc for b."""
    observed_diff = abs(statistic(b) - statistic(a))
    num_a = len(a)

    combined = np.concatenate([a, b])
    diffs = []
    for i in range(num_samples):
        xs = npr.permutation(combined)
        diff = np.mean(xs[:num_a]) - np.mean(xs[num_a:])
        diffs.append(diff)

    pval = np.sum(np.abs(diffs) >= np.abs(observed_diff)) / float(num_samples)
    return pval, observed_diff, diffs


# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "wintermute":
    data_path = "/home/mje/Projects/MEG_Hyopnosis/data/"
else:
    data_path = "/projects/" + \
                "MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "scratch/Tone_task_MNE_2/"

subjects_dir = data_path + "fs_subjects_dir"
result_dir = data_path + "results"

# change dir to save files the rigth place
os.chdir(data_path)


# %%
# load numpy files
label_ts_normal_crop =\
    np.load(data_path + "Nrm_press_label_ts_pca-flip_zscore_resample_0_05.npy")
label_ts_hyp_crop =\
    np.load(data_path + "Hyp_press_label_ts_pca-flip_zscore_resample_0_05.npy")

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

# labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
#                                     subjects_dir=subjects_dir)

labels_name = [label.name for label in labels]
# for label in labels:
#     labels_name += [label.name]


# %%
coh_list_nrm = []
coh_list_hyp = []

for j in range(len(label_ts_normal_crop)):
    nits = TimeSeries(label_ts_normal_crop[j],
                      sampling_rate=300)  # epochs_normal.info["sfreq"])
    nits.metadata["roi"] = labels_name

    coh_list_nrm += [CoherenceAnalyzer(nits)]

for j in range(len(label_ts_hyp_crop)):
    nits = TimeSeries(label_ts_hyp_crop[j],
                      sampling_rate=300)  # epochs_normal.info["sfreq"])
    nits.metadata["roi"] = labels_name

    coh_list_hyp += [CoherenceAnalyzer(nits)]

# Compute a source estimate per frequency band
bands = dict(theta=[4, 8],
             alpha=[8, 12],
             beta=[13, 25],
             gamma_low=[30, 48],
             gamma_high=[52, 90])


bands = dict(beta=[12, 25])

for band in bands.keys():
    print "\n******************"
    print "\nAnalysing band: %s" % band
    print "\n******************"

    # extract coherence values
    f_lw, f_up = bands[band]  # lower & upper limit for frequencies

    coh_matrix_nrm = np.empty([len(label_ts_normal_crop),
                               len(labels_name),
                               len(labels_name)])
    coh_matrix_hyp = np.empty([len(label_ts_hyp_crop),
                               len(labels_name),
                               len(labels_name)])

    # confine analysis to Aplha (8  12 Hz)
    freq_idx = np.where((coh_list_hyp[0].frequencies >= f_lw) *
                        (coh_list_hyp[0].frequencies <= f_up))[0]

    print coh_list_nrm[0].frequencies[freq_idx]

    # compute average coherence &  Averaging on last dimension
    for j in range(coh_matrix_nrm.shape[0]):
        coh_matrix_nrm[j, :, :]=np.mean(
            coh_list_nrm[j].coherence[:, :, freq_idx], -1)

    for j in range(coh_matrix_hyp.shape[0]):
        coh_matrix_hyp[j, :, :]=np.mean(
            coh_list_hyp[j].coherence[:, :, freq_idx], -1)

    #
    full_matrix = np.concatenate([coh_matrix_nrm, coh_matrix_hyp], axis=0)

    threshold = np.median(full_matrix[np.nonzero(full_matrix)]) + \
        np.std(full_matrix[np.nonzero(full_matrix)])

    bin_matrix_nrm = coh_matrix_nrm > threshold
    bin_matrix_hyp = coh_matrix_hyp > threshold

    #
    nx_nrm = []
    for j in range(bin_matrix_nrm.shape[0]):
        nx_nrm += [nx.from_numpy_matrix(bin_matrix_nrm[j, :, :])]

    nx_hyp = []
    for j in range(bin_matrix_hyp.shape[0]):
        nx_hyp += [nx.from_numpy_matrix(bin_matrix_hyp[j, :, :])]

    #
    degrees_nrm = []
    for j, trial in enumerate(nx_nrm):
        degrees_nrm += [trial.degree()]

    degrees_hyp = []
    for j, trial in enumerate(nx_hyp):
        degrees_hyp += [trial.degree()]

    cc_nrm = []
    for j, trial in enumerate(nx_nrm):
        cc_nrm += [nx.cluster.clustering(trial)]
    cc_hyp = []
    for j, trial in enumerate(nx_hyp):
        cc_hyp += [nx.cluster.clustering(trial)]

    # Degress
    pval_list = []
    for degree_number in range(bin_matrix_hyp.shape[0]):

        post_hyp = np.empty(len(degrees_hyp))
        for j in range(len(post_hyp)):
            post_hyp[j] = degrees_hyp[j][degree_number]

        post_normal = np.empty(len(degrees_nrm))
        for j in range(len(post_normal)):
            post_normal[j] = degrees_nrm[j][degree_number]

        pval, observed_diff, diffs = \
            permutation_test(post_hyp, post_normal, 10000, np.mean)

        pval_list += [{'area': labels_name[degree_number],
                       'pval': pval,
                       "obsDiff": observed_diff,
                       "diffs": diffs}]

    #  for CC
    pval_list_CC = []
    for cc_number in range(bin_matrix_hyp.shape[0]):

        post_hyp = np.empty(len(cc_hyp))
        for j in range(len(cc_hyp)):
            post_hyp[j] = cc_hyp[j][cc_number]

        post_normal = np.empty(len(cc_nrm))
        for j in range(len(post_normal)):
            post_normal[j] = cc_nrm[j][cc_number]

        pval, observed_diff, diffs = \
            permutation_test(post_hyp, post_normal, 10000, np.mean)

        pval_list_CC += [{'area': labels_name[cc_number],
                          'pval': pval,
                          "obsDiff": observed_diff,
                          "diffs": diffs}]

    Pickle.dump(pval_list,
                open(result_dir +
                     "/nx_press_%s_deg_zscore_BA_Coh_0-05_resample.p" % band,
                     "wb"))

    Pickle.dump(pval_list_CC,
                open(result_dir +
                     "/nx_press_%s_CC_zscore_BA_Coh_0-05_resample.p" % band,
                     "wb"))
