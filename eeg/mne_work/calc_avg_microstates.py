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
import fnmatch


channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6',
  'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

_bad_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4',
  'EXG5', 'EXG6', 'EXG7', 'EXG8']

def calc_average_microstates():

    maps_dir = r"I:\Munka\Masodik\mne_work\microstates\maps"
    microstates_df = pd.read_csv(r"I:\Munka\Masodik\mne_work\microstates\connecting_microstates_1.csv")
    colnames = ["microstate_A", "microstate_B", "microstate_C", "microstate_D", "microstate_D2", ]
    microstate_counters = [0] * len(colnames)
    microstate_avgs = [None] * len(colnames)
    person_counter = 0
    for idx, cn in enumerate(colnames):

        m_idx = idx

        for _, row in microstates_df.iterrows():
            maps_df = pd.read_csv(f"{maps_dir}/{row['id']}_maps.csv")
            maps_df.rename(columns={"1": "A", "2": "B", "3": "C", "4":"D"}, inplace=True)
            if row[cn] != "Missing":
                letter = "A" if cn[-1] == "A" else "B" if cn[-1] == "B" else "C" if cn[-1] == "C" else "D" if cn[-1] == "D" else "D2" if cn[-1] == "2" else None
                map_microstate = maps_df[row[cn]]
                polarity_reverse = True if row[f"switch_{letter}_polarity"] == "True" else False
                if polarity_reverse:
                    map_microstate = -1 * map_microstate
                if microstate_avgs[m_idx] is None:
                    microstate_avgs[m_idx] = map_microstate
                else:
                    microstate_avgs[m_idx] += map_microstate
                microstate_counters[m_idx] += 1

    micro_avg_df = pd.DataFrame()
    for idx, mavg in enumerate(microstate_avgs):
        avg_microstate = mavg / microstate_counters[idx]
        micro_avg_df[colnames[idx]] = avg_microstate

    micro_avg_df.to_csv("calculated_avg_microstates.csv")


def plot_avg_microstates():


    avg_microstates_df = pd.read_csv(r"I:\Munka\Masodik\mne_work\calculated_avg_microstates.csv")

    #just for a sample - to consturct a bdf with the appropriate .info  data.
    raw = mne.io.read_raw_bdf(r"I:\Munka\Masodik\converter\files\e1_EC.bdf", exclude=_bad_channels, eog=channel_names, stim_channel='Status',
                              preload=True)
    resample_fraction = 64
    resample_freq = raw.info['sfreq'] // resample_fraction
    raw_freq = raw.info['sfreq']
    if resample_fraction > 1:
        raw_resampled = raw.resample(resample_freq, npad="auto")
        raw_freq = raw_freq / resample_fraction
        if not raw_freq % 1 == 0:
            sys.exit()

    epoch_length = 512 // resample_fraction
    overlap = 0

    biosemi_montage = mne.channels.make_standard_montage('biosemi32')
    channel_types = {ch_name: 'eeg' for ch_name in raw.ch_names}
    channel_types['Status'] = 'stim'
    raw.set_channel_types(channel_types)
    raw.set_montage(biosemi_montage, on_missing='ignore')

    raw.pick_types(meg=False, eeg=True, stim=True)
    # raw.notch_filter(freqs=[50, 100], n_jobs=1)
    raw.notch_filter(freqs=[50], n_jobs=1)
    raw.filter(l_freq=0.5, h_freq=40, n_jobs=1)

    raw.set_eeg_reference('average')
    epochs_data = mne.make_fixed_length_epochs(raw, duration=epoch_length / raw.info['sfreq'],
                                               overlap=overlap).get_data()
    events = mne.make_fixed_length_events(raw, duration=epoch_length / raw.info['sfreq'])
    epochs = mne.EpochsArray(epochs_data, raw.info, events=events)
    raw.add_events(events)
    raw.set_annotations(epochs.annotations)

    for c in avg_microstates_df.columns[1:]:
        mne.viz.plot_topomap(avg_microstates_df[c], raw.info, show=False, ch_type="eeg", res=512, size=3)
        plt.savefig(f"average_{c}.png")


def calculate_missings():

    df = pd.read_csv(r"I:\Munka\Masodik\mne_work\microstates\connecting_microstates_1.csv")

    values = df.values.tolist()
    counter_missing = 0

    for v in values:
        if v.count("Missing") > 2:
            counter_missing += 1

    print(counter_missing)



if __name__ == "__main__":
    plot_avg_microstates()