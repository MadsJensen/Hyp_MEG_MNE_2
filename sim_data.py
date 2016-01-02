"""Script to simulate data based on the actual recorded data.

@author mje
@email mads [] cnru.dk
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import socket

import mne
from mne import find_events, Epochs, compute_covariance
from mne.simulation import simulate_sparse_stc, simulate_raw
from mne.minimum_norm import read_inverse_operator

print(__doc__)

# Settings
n_jobs = 1
plt.ion()  # make plt interactive

# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "Wintermute":
    data_path = "/home/mje/Projects/MEG_Hyopnosis/data/"
    subjects_dir = "/home/mje/Projects/MEG_Hyopnosis/data/fs_subjects_dir"
else:
    data_path = "/projects/" + \
                "MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "scratch/Tone_task_MNE_2/"
    subjects_dir = "/projects/" + \
                   "MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "scratch/fs_subjects_dir/"


raw_fnormal = data_path + "tone_task-normal-tsss-mc-autobad-ica_raw.fif"
raw_fhyp = data_path + "tone_task-hyp-tsss-mc-autobad-ica_raw.fif"
raw_fnormal = data_path + "tone_task-normal-tsss-mc-autobad-ica_raw.fif"
raw_fhyp = data_path + "tone_task-hyp-tsss-mc-autobad-ica_raw.fif"

trans_nrm = data_path + "tone-trans.fif"
trans_hyp = data_path + "tone_hyp-trans.fif"
bem_fname = subjects_dir + "/subject_1/bem/" +\
                           "subject_1-5120-bem-sol.fif"


# change dir to save files the rigth place
os.chdir(data_path)

reject = dict(grad=4000e-13,  # T / m (gradiometers)
              mag=4e-12,  # T (magnetometers)
              #  eog=250e-6  # uV (EOG channels)
              )

raw_nrm = mne.io.Raw(raw_fnormal, preload=False)
raw_hyp = mne.io.Raw(raw_fhyp, preload=False)

raw_nrm = raw_nrm.crop(0., 30., copy=False)  # 30 sec is enough

epochs_fnrm = data_path + "subj_1_nrm-epo.fif"
epochs_fhyp = data_path + "subj_1_hyp-epo.fif"
inverse_fnrm = data_path + "subj_1-nrm-inv.fif"
inverse_fhyp = data_path + "subj_1-hyp-inv.fif"
# change dir to save files the rigth place
os.chdir(data_path)

reject = dict(grad=4000e-13,  # T / m (gradiometers)
              mag=4e-12,  # T (magnetometers)
              #  eog=250e-6  # uV (EOG channels)
              )

# Load data
inverse_nrm = read_inverse_operator(inverse_fnrm)
inverse_hyp = read_inverse_operator(inverse_fhyp)

epochs_nrm = mne.read_epochs(epochs_fnrm)
epochs_hyp = mne.read_epochs(epochs_fhyp)

epochs_nrm = epochs_nrm["Tone"]
epochs_hyp = epochs_hyp["Tone"]

src = mne.read_source_spaces(data_path + "subj_1-oct6-src.fif")

# Load labels
labels = mne.read_labels_from_annot('subject_1', parc='aparc',
                                    # regexp="Brodmann",
                                    subjects_dir=subjects_dir)

labels_to_sim = [labels[16], labels[48]]

##############################################################################
# Generate dipole time series
n_dipoles = 2  # number of dipoles to create
epoch_duration = 1.  # duration of each epoch/event
n = 0  # harmonic number


def data_fun(times):
    """Generate time-staggered sinusoids at harmonics of 10Hz."""
    global n
    n_samp = len(times)
    window = np.zeros(n_samp)
    start, stop = [int(ii * float(n_samp) / (2 * n_dipoles))
                   for ii in (2 * n, 2 * n + 1)]
    window[start:stop] = 1.
    n += 1
    data = 25e-9 * np.sin(2. * np.pi * 10. * n * times)
    data *= window
    return data

times = raw_nrm.times[:int(raw_nrm.info['sfreq'] * epoch_duration)]
# stc = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
#                           data_fun=data_fun, random_state=0)

stc = simulate_sparse_stc(src, times=times, n_dipoles=n_dipoles,
                          data_fun=data_fun,
                          labels=labels_to_sim,
                          random_state=0)

print stc.data.max()

# look at our source data
fig, ax = plt.subplots(1)
ax.plot(times, 1e9 * stc.data.T)
ax.set(ylabel='Amplitude (nAm)', xlabel='Time (sec)')
fig.show()

##############################################################################
# Simulate raw data
raw_sim = simulate_raw(raw_nrm, stc, trans_nrm, src, bem_fname, cov='simple',
                       iir_filter=[0.2, -0.2, 0.04], ecg=True, blink=True,
                       n_jobs=2, verbose=True)
raw_sim.plot()

##############################################################################
# Plot evoked data
events = find_events(raw_sim)  # only 1 pos, so event number == 1
epochs = Epochs(raw_sim, events, 1, -0.2, epoch_duration)
cov = compute_covariance(epochs, tmax=0., method='auto', return_estimators="all")  # quick calc
evoked = epochs.average()
evoked.plot_white(cov)
