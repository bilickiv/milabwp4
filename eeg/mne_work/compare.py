import mne
import numpy as np
import matplotlib.pyplot as plt
import mne_microstates
import mne_work.analyzer as analyzer
import os
import pandas as pd
import seaborn as sns
import matplotlib
from autoreject import compute_thresholds
import mne_work.preprocesser as preprocesser
from pathlib import Path
import scipy
import copy
from matplotlib.pyplot import figure
import ast

channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6',
  'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

_bad_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4',
  'EXG5', 'EXG6', 'EXG7', 'EXG8']

def compare_sequences(A,B,C,D):

    file1 = r"I:\Munka\Masodik\mne_work\microstates_e12c.txt"

    sequences_1 = []
    with open(file1, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for l in lines:
            seq = ast.literal_eval(l)
            sequences_1 = sequences_1 + seq

    file2 = r"I:\Munka\Masodik\csatolmanyok\matetol\labelSequence\afterICA_ec12_microstates.csv"
    sequences_2 = []
    with open(file2, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for l in lines:
            seq = ast.literal_eval(l)
            sequences_2 = sequences_2 + list(seq)

    sequences_2 = ["A" if s == A else "B" if s == B else "C" if s == C else "D" if s == D else None for s in seq]

    return sequences_1, sequences_2





def get_peaks_indexes():

    raw = mne.io.read_raw_bdf(f"../converter/files/e12_EC.bdf", exclude=_bad_channels, eog=channel_names, stim_channel='Status',
                              preload=True)

    resample_fraction = 64
    resample_freq = raw.info['sfreq'] // resample_fraction
    raw_freq = raw.info['sfreq']
    if resample_fraction > 1:
        raw.resample(resample_freq, npad="auto")
        raw_freq = raw_freq / resample_fraction

    epoch_length = 512 // resample_fraction
    overlap = 0

    biosemi_montage = mne.channels.make_standard_montage('biosemi32')
    channel_types = {ch_name: 'eeg' for ch_name in raw.ch_names}
    channel_types['Status'] = 'stim'
    raw.set_channel_types(channel_types)
    raw.set_montage(biosemi_montage, on_missing='ignore')

    raw.pick_types(meg=False, eeg=True, stim=True)
    raw.notch_filter(freqs=[50], n_jobs=1)
    raw.filter(l_freq=0.5, h_freq=40, n_jobs=1)

    raw.set_eeg_reference('average')
    epochs_data = mne.make_fixed_length_epochs(raw, duration=epoch_length / raw.info['sfreq'],
                                               overlap=overlap).get_data()
    events = mne.make_fixed_length_events(raw, duration=epoch_length / raw.info['sfreq'])
    epochs = mne.EpochsArray(epochs_data, raw.info, events=events)
    raw.add_events(events)
    raw.set_annotations(epochs.annotations)

    gfp = np.std(raw.get_data(), axis=0)
    peaks, _ = mne_microstates.find_peaks(gfp)

    return peaks


def compare_labels(seq1, seq2):

    print(len(seq1))
    print(len(seq2))


    same_labels = 0

    for l1, l2 in zip(seq1, seq2):

        if l1 == l2:
            same_labels += 1

    print(f"Összes label: {len(seq2)}")
    print(f"Egyező label: {same_labels}")
    print(f"Egyezés nagysága: {(same_labels / len(seq2)) * 100}%")
    print()



def count_sequences(seq):
    counted = 0
    for idx, s in enumerate(seq):
        if idx > 0 and idx < len(seq) - 1:
            if seq[idx - 1] != s and s != seq[idx + 1]:
                counted += 1
    print(counted)


def plot_microstates(s1):

    s1copy = copy.deepcopy(s1)
    s1copy = [1 if s == "A" else 2 if s == "B" else 3 if s == "C" else 4 if s == "D" else None for s in s1copy]
    s1c = [s if s == 1 else None for s in s1copy]
    s2c = [s if s == 2 else None for s in s1copy]
    s3c = [s if s == 3 else None for s in s1copy]
    s4c = [s if s == 4 else None for s in s1copy]


    figure(num=None, figsize=(64, 2), dpi=200, facecolor='lightgrey', edgecolor='k')
    plt.rcParams['figure.facecolor'] = "lightgrey"
    plt.rcParams['axes.facecolor'] = "lightgrey"
    plt.rcParams['lines.color'] = "white"
    plt.rc('legend', fontsize=26)
    plt.rc('axes', titlesize=30, labelsize=30)
    plt.rc('ytick', labelsize=22)
    plt.rc('xtick', labelsize=18)

    plt.plot(range(len(s1c)), s1c)
    plt.plot(range(len(s2c)), s2c)
    plt.plot(range(len(s3c)), s3c)
    plt.plot(range(len(s4c)), s4c)
    plt.show()


def main_compare():


    for A in range(1, 5):
        for B in range(1, 5):
            for C in range(1, 5):
                for D in range(1, 5):
                    if A != B and A != C and A != D and B != C and B != D and C != D:
                        print(f"{A} - {B} - {C} - {D}")
                        s1, s2 = compare_sequences(A,B,C,D)
                        #peak_indexes = get_peaks_indexes()
                        compare_labels(s1, s2)


#main_compare()

s1, s2 = compare_sequences(1,2,3,4)
compare_labels(s1, s2)
count_sequences(s1)
count_sequences(s2)

