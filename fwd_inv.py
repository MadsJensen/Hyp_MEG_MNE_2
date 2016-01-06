"""Doc string goes here.

@author: mje mads [] cnru.dk
"""


import socket
import mne
from mne.minimum_norm import make_inverse_operator
import os

# SETUP PATHS AND PREPARE RAW DATA
hostname = socket.gethostname()

if hostname == "Wintermute":
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


bem = mne.read_bem_solution(subjects_dir + "/subject_1/bem/" +
                            "subject_1-5120-bem-sol.fif")
nrm_fname = data_path + "tone_task-normal-tsss-mc-autobad-ica_raw.fif"
hyp_fname = data_path + "tone_task-hyp-tsss-mc-autobad-ica_raw.fif"

nrm_trans = data_path + "nrm2-trans.fif"
hyp_trans = data_path + "hyp2-trans.fif"

# src = mne.setup_source_space("subject_1",
#                              "nrm-src-oct6.fif",
#                              spacing="oct6",
#                              subjects_dir=subjects_dir,
#                              n_jobs=2)

src = mne.read_source_spaces(data_path + "subj_1-oct6-src.fif")

fwd_nrm = mne.make_forward_solution(nrm_fname,
                                    trans=nrm_trans,
                                    src=src,
                                    bem=bem,
                                    meg=True,
                                    eeg=False,
                                    fname="subj_1-nrm-fwd.fif",
                                    overwrite=True)

fwd_hyp = mne.make_forward_solution(hyp_fname,
                                    trans=hyp_trans,
                                    src=src,
                                    bem=bem,
                                    meg=True,
                                    eeg=False,
                                    fname="subj_1-hyp-fwd.fif",
                                    overwrite=True)

fwd_nrm = mne.read_forward_solution("subj_1-nrm-fwd.fif")
fwd_hyp = mne.read_forward_solution("subj_1-hyp-fwd.fif")

raw_nrm = mne.io.Raw(nrm_fname, preload=False)
raw_hyp = mne.io.Raw(hyp_fname, preload=False)

reject = dict(grad=4000e-13,  # T / m (gradiometers)
              mag=4e-12  # T (magnetometers)
              # eeg=180e-6 #
              )

# SET PARAMETERS
tmin, tmax = -1, 2

# SELECT EVENTS TO EXTRACT EPOCHS FROM.
event_id = {'tone': 8}

# Setup for reading the raw data
events_nrm = mne.find_events(raw_nrm)
events_hyp = mne.find_events(raw_hyp)

picks = mne.pick_types(raw_nrm.info, meg=True, eeg=False, stim=False,
                       eog=False,
                       include=[], exclude='bads')
# Read epochs
epochs_nrm = mne.Epochs(raw_nrm, events_nrm, event_id, tmin, tmax, picks=picks,
                        baseline=(None, -0.5), reject=reject,
                        preload=True)

epochs_hyp = mne.Epochs(raw_hyp, events_hyp, event_id, tmin, tmax, picks=picks,
                        baseline=(None, -0.5), reject=reject,
                        preload=True)

cov_nrm = mne.compute_covariance(epochs_nrm, tmin=None, tmax=-0.5,
                                 method="shrunk")
cov_hyp = mne.compute_covariance(epochs_hyp, tmin=None, tmax=-0.5,
                                 method="shrunk")
cov_nrm.save("subj_1-nrm-cov.fif")
cov_hyp.save("subj_1-hyp-cov.fif")

inv_nrm = make_inverse_operator(epochs_nrm.info, fwd_nrm, cov_nrm,
                                loose=0.2, depth=0.8)
inv_hyp = make_inverse_operator(epochs_hyp.info, fwd_hyp, cov_hyp,
                                loose=0.2, depth=0.8)

mne.minimum_norm.write_inverse_operator("subj_1-nrm-inv.fif", inv_nrm)
mne.minimum_norm.write_inverse_operator("subj_1-hyp-inv.fif", inv_hyp)
