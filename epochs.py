"""
Script to epoch raw data.

@author: mje mads [] cnru.dk
"""
import mne
import socket
import numpy as np
import os


# Setup paths and prepare raw data
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


raw_fnormal = data_path + "tone_task-normal-tsss-mc-autobad-ica_raw.fif"
raw_fhyp = data_path + "tone_task-hyp-tsss-mc-autobad-ica_raw.fif"
raw_fnormal = data_path + "tone_task-normal-tsss-mc-autobad-ica_raw.fif"
raw_fhyp = data_path + "tone_task-hyp-tsss-mc-autobad-ica_raw.fif"

# change dir to save files the rigth place
os.chdir(data_path)

reject = dict(grad=4000e-13,  # T / m (gradiometers)
              mag=4e-12,  # T (magnetometers)
              #  eog=250e-6  # uV (EOG channels)
              )

conditions = ["normal", "hyp"]
for condition in conditions:

    if condition == "normal":
        raw = mne.io.Raw(raw_fnormal, preload=True)
    elif condition == "hyp":
        raw = mne.io.Raw(raw_fhyp, preload=True)

    # FIND events and correct for multple and missing
    # button presses
    events = mne.find_events(raw, stim_channel='STI101')

    # Plot the events to get an idea of the paradigm
    # Specify colors and an event_id dictionary for the legend.
    event_ids = {'button': 1, 'Tone': 8}
    color = {1: 'blue', 8: 'red'}

    # mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp,
    #                     color=color, event_id=event_id, show=False)

    for j in range(len(events)):
        if j == 0:
            events_corrected = events[j, :]
        else:
            if events[j, 2] == 8 and events[j + 1, 2] == 1:
                events_corrected = np.vstack([events_corrected, events[j, :]])
                events_corrected = np.vstack([events_corrected,
                                              events[j + 1, :]])

    # save corrected events
    mne.write_events("events_%s_corrected-eve.fif" % (condition),
                     events_corrected)

    events_corr_time = events_corrected.copy()
    for k in range(len(events_corr_time)):
        events_corr_time[k][0] = events_corr_time[k][0] / raw.info["sfreq"]

    # epochs settings
    # event_ids = {"Tone": 8}
    tmin, tmax = -1, 2
    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True, eog=True,
                           emg=True, exclude='bads')

    epochs = mne.Epochs(raw, events_corrected, event_ids, tmin, tmax,
                        proj=False, picks=picks, baseline=(None, -0.7),
                        preload=True, reject=reject)
#    epochs.save("tone_task_%s-epo.fif" % (condition))
#    evoked = epochs.average()
#    evoked.save("tone_task_%s-ave.fif" % (condition))
    exec("epochs_%s = epochs" % (condition))

epochs_normal.save("subj_1_nrm-epo.fif")
epochs_hyp.save("subj_1_hyp-epo.fif")

# layout = mne.find_layout(epochs.info, 'meg')  # use full layout

# title = 'ERF images - MNE sample data'
# mne.viz.plot_topo_image_epochs(epochs, layout, sigma=0.5, vmin=-200,
#                                vmax=200, colorbar=True, title=title)
# plt.show()
