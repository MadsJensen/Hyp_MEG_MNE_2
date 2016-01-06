"""
Doc string here.

@author: mje
@email: mads [] cnru.dk
"""
import numpy as np
import mne
import os
import socket

from mne.minimum_norm import (apply_inverse_epochs, read_inverse_operator)
from mne.baseline import rescale

# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "wintermute":
    data_path = "/home/mje/Projects/MEG_Hyopnosis/data/"
    subjects_dir = "/home/mje/Projects/MEG_Hyopnosis/data/fs_subjects_dir"
else:
    data_path = "/projects/" + \
                "MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "scratch/Tone_task_MNE_ver_2/"
    subjects_dir = "/projects/" + \
                   "MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "scratch/fs_subjects_dir/"

# CHANGE DIR TO SAVE FILES THE RIGTH PLACE
os.chdir(data_path)

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

# SETTINGS
snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
method = "MNE"
n_jobs = 1

# Load data
inverse_nrm = read_inverse_operator(inverse_fnrm)
inverse_hyp = read_inverse_operator(inverse_fhyp)

epochs_nrm = mne.read_epochs(epochs_fnrm)
epochs_hyp = mne.read_epochs(epochs_fhyp)

epochs_nrm = epochs_nrm["Tone"]
epochs_hyp = epochs_hyp["Tone"]


#
stcs_nrm = apply_inverse_epochs(epochs_nrm, inverse_nrm, lambda2,
                                method, pick_ori="normal",
                                return_generator=False)
stcs_hyp = apply_inverse_epochs(epochs_hyp, inverse_hyp, lambda2,
                                method, pick_ori="normal",
                                return_generator=False)


# resample
[stc.resample(300) for stc in stcs_nrm]
[stc.resample(300) for stc in stcs_hyp]

# Get labels from FreeSurfer cortical parcellation
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

# Average the source estimates within eachh label using sign-flips to reduce
# signal cancellations, also here we return a generator
src_nrm = inverse_nrm['src']
label_ts_nrm = mne.extract_label_time_course(stcs_nrm, labels,
                                             src_nrm,
                                             mode='pca_flip',
                                             return_generator=False)

src_hyp = inverse_hyp['src']
label_ts_hyp = mne.extract_label_time_course(stcs_hyp, labels, src_hyp,
                                             mode='pca_flip',
                                             return_generator=False)

# standardize TS's
label_ts_nrm_rescaled = []
for j in range(len(label_ts_nrm)):
    label_ts_nrm_rescaled += [rescale(label_ts_nrm[j], epochs_nrm.times,
                                      baseline=(None, -0.5), mode="zscore")]

label_ts_hyp_rescaled = []
for j in range(len(label_ts_hyp)):
    label_ts_hyp_rescaled += [rescale(label_ts_hyp[j], epochs_hyp.times,
                                      baseline=(None, -0.5), mode="zscore")]


from_time = np.abs(stcs_nrm[0].times + 0).argmin()
to_time = np.abs(stcs_nrm[0].times - 0.2).argmin()

label_ts_nrm_rescaled_crop = []
for j in range(len(label_ts_nrm)):
    label_ts_nrm_rescaled_crop +=\
        [label_ts_nrm_rescaled[j][:, from_time:to_time]]

label_ts_hyp_rescaled_crop = []
for j in range(len(label_ts_hyp)):
    label_ts_hyp_rescaled_crop +=\
       [label_ts_hyp_rescaled[j][:, from_time:to_time]]

np.save(data_path + "Hyp_tone_label_ts_pca-flip_zscore_resample_0_02.npy",
        label_ts_hyp_rescaled_crop)
np.save(data_path + "Nrm_tone_label_ts_pca-flip_zscore_resample_0_02.npy",
        label_ts_nrm_rescaled_crop)
