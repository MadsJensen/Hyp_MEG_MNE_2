# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:31:46 2015

@author: mje
"""

import numpy as np
import matplotlib.pyplot as plt
import socket
import os
import mne

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score

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
label_ts_nrm_crop =\
    np.load(data_path + "Nrm_tone_label_ts_mean-flip_zscore_resample_0_02.npy")
label_ts_hyp_crop =\
    np.load(data_path + "Hyp_tone_label_ts_mean-flip_zscore_resample_0_02.npy")

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

label_names = [label.name for label in labels]

# Reshape TS's
nrm_dim = label_ts_nrm_crop.shape
nrm_ts = np.empty([nrm_dim[0], nrm_dim[1] * nrm_dim[2]])
hyp_dim = label_ts_hyp_crop.shape
hyp_ts = np.empty([hyp_dim[0], hyp_dim[1] * hyp_dim[2]])

for j in range(len(nrm_ts)):
    nrm_ts[j, :] = label_ts_nrm_crop[j].reshape(-1)
for j in range(len(hyp_ts)):
    hyp_ts[j, :] = label_ts_hyp_crop[j].reshape(-1)


# scikit-learn settings
y = np.concatenate([np.zeros(len(nrm_ts)),
                    np.ones(len(hyp_ts))])

X = np.vstack([nrm_ts, hyp_ts])
# X = X*1e11

estimators = np.arange(0, 2050, 50)
estimators = estimators[1:]

class_cv_results = []

for j in range(estimators):
    bdt = AdaBoostClassifier(algorithm="SAMME.R",
                             n_estimators=estimators[j])
    bdt.fit(X, y)

    class_cv_results += [cross_val_score(bdt, X, y, cv=10, n_jobs=2,
                                         verbose=False)]

# y_pred = bdt.predict(X)
# confusion_matrix(y, y_pred)
