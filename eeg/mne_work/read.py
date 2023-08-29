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
import sys
"""
Event IDs: [66559]
<Info | 7 non-empty values
 bads: []
 ch_names: ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 
 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6',
  'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 
  'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status'] == 41

 chs: 40 EEG, 1 Stimulus
 custom_ref_applied: False
 highpass: 0.0 Hz
 lowpass: 1667.0 Hz
 meas_date: 2019-04-24 08:25:40 UTC
 nchan: 41
 projs: []
 sfreq: 8192.0 Hz

Process finished with exit code 0

"""
matplotlib.use('TkAgg')

channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6',
  'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

_bad_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4',
  'EXG5', 'EXG6', 'EXG7', 'EXG8']

event_id = 66559
dirpath = "../converter/files"

global_states_df = pd.read_csv(r"I:\Munka\Masodik\csatolmanyok\matetol\maps.csv")


def make_corr_matrices(data, fname):
    gfp = np.std(data, axis=0)
    peaks, _ = mne_microstates.find_peaks(gfp, distance=2)
    n_peaks = len(peaks)
    print(n_peaks)
    df_dict = {}
    for idx, i in enumerate(channel_names):
        df_dict[i] = [data[idx][p] for p in peaks]

    df = pd.DataFrame(df_dict)

    columns = df.columns
    corr_matrix = abs(df[columns].corr())
    Path(f"characteristics/heatmaps/filtered").mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(f"characteristics/heatmaps/filtered/{fname}_heatmap.csv")



def plot_corr_matrices(dirpath):
    for f in os.listdir(dirpath):
        fname = f.split(".")[0]
        corr_matrix = pd.read_csv(f"{dirpath}/{f}", index_col=0)
        plt.figure(figsize=[12,10])
        sns.heatmap(corr_matrix, linewidth=0.3)
        Path(f"characteristics_plots/heatmaps/filtered").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"characteristics_plots/heatmaps/filtered/{fname}.png")
        plt.cla()
        plt.clf()
        plt.close()



def categorize_peaks_spatial_correlation_mate(data, peaks):

    data_temp = data[:32]

    microstates = []
    states = ["A", "B", "C", "D"]
    for p in range(len(data[0])):
        abs_diff = [0] * 4
        for idx, state in enumerate(states):
            electrode_values = [abs(d[p]) for d in data_temp]
            electrode_global_values = abs(global_states_df[state].values)
            abs_diff[idx] = scipy.spatial.distance.correlation(electrode_values, electrode_global_values)


        min_index = abs_diff.index(min(abs_diff))
        microstates.append(states[min_index])

    change_dict = {"A": 1, "B": 2, "C": 3, "D": 4}

    return [change_dict[m] for m in microstates]


def categorize_spatial_correlation_sajat(data, peaks):


    glob_micro_Df = pd.read_csv(r"I:\Munka\Masodik\mne_work\calculated_avg_microstates.csv")

    data_temp = data[:32]

    microstates = []
    states = ["microstate_A", "microstate_B", "microstate_C", "microstate_D", "microstate_D2"]
    for p in range(len(data[0])):
        abs_diff = [0] * 5
        for idx, state in enumerate(states):
            electrode_values = [abs(d[p]) for d in data_temp]
            electrode_global_values = abs(glob_micro_Df[state].values)
            abs_diff[idx] = scipy.spatial.distance.correlation(electrode_values, electrode_global_values)


        min_index = abs_diff.index(min(abs_diff))
        microstates.append(states[min_index])

    change_dict = {"microstate_A": 1, "microstate_B": 2, "microstate_C": 3, "microstate_D": 4, "microstate_D2": 5}

    return [change_dict[m] for m in microstates]



def categorize_peaks(data, peaks):

    microstates = []
    states = ["A", "B", "C", "D"]
    for p in peaks:
        abs_diff = [0] * 4
        for idx, state in enumerate(states):
            for idx2, ch in enumerate(data):
                if idx2 == 31:
                    break
                state_val = global_states_df.iloc[idx2][state]
                signal_val = ch[p]
                #abs_diff[idx] += np.abs(np.abs(state_val) - np.abs(signal_val))


        min_index = abs_diff.index(min(abs_diff))
        microstates.append(states[min_index])

    return microstates


useless_files = pd.read_csv("bads/useless_data.csv")

def make_microstates(dirpath):

    for f in os.listdir(dirpath):

        print(f"{f} is in the making!")

        fname = f.split(".")[0]

        """
        Skips the reading of the file
        if the filename:
        - is in useless_files or
        - starts with "EO" or
        - has 3 .csv files under bads/{filename}  (bad_channels, bad_ica_components, bad_intervals)
        """

        if fname in useless_files.name.values or\
                "EO" in fname or (os.path.isdir(f"bads/{fname}") and len(fnmatch.filter(os.listdir(f"bads/{fname}"), '*.csv')) == 3):
            continue

        try:
            raw = mne.io.read_raw_bdf(f"{dirpath}/{f}", exclude=_bad_channels, eog=channel_names, stim_channel='Status', preload=True)
        except Exception as e:
            print(e)
            continue
        resample_fraction = 64
        resample_freq = raw.info['sfreq']//resample_fraction
        raw_freq = raw.info['sfreq']
        if resample_fraction > 1:
            raw.resample(resample_freq, npad="auto")
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
        #raw.notch_filter(freqs=[50, 100], n_jobs=1)
        raw.notch_filter(freqs=[50], n_jobs=1)
        raw.filter(l_freq=0.5, h_freq=40, n_jobs=1)

        raw.set_eeg_reference('average')
        epochs_data = mne.make_fixed_length_epochs(raw, duration=epoch_length/raw.info['sfreq'], overlap=overlap).get_data()
        events = mne.make_fixed_length_events(raw, duration=epoch_length / raw.info['sfreq'])
        epochs = mne.EpochsArray(epochs_data, raw.info, events=events)
        raw.add_events(events)
        raw.set_annotations(epochs.annotations)

        Path(f"bads/{fname}").mkdir(parents=True, exist_ok=True)
        if not os.path.isfile(f"bads/{fname}/{fname}_bad_intervals.csv"):
            raw.plot(block=True)
            bad_channels = raw.info['bads']
            bad_intervals = raw.annotations[raw.annotations.description == 'BAD_']
            bad_interval_starts = bad_intervals.onset
            bad_interval_durations = bad_intervals.duration
            bad_interval_descriptions = bad_intervals.description

            # Create a pandas DataFrame with the bad interval information
            bad_intervals_df = pd.DataFrame({'start_time': bad_interval_starts,
                                             'duration': bad_interval_durations,
                                             'description': bad_interval_descriptions})
            bad_channels_df = pd.DataFrame({'bad_channels': bad_channels})


            bad_intervals_df.to_csv(f"bads/{fname}/{fname}_bad_intervals.csv")
            bad_channels_df.to_csv(f"bads/{fname}/{fname}_bad_channels.csv")



        bad_intervals_df = pd.read_csv(f"bads/{fname}/{fname}_bad_intervals.csv")
        bad_channels_df = pd.read_csv(f"bads/{fname}/{fname}_bad_channels.csv")

        bad_intervals_ann = mne.Annotations(onset=bad_intervals_df['start_time'],
                                            duration=bad_intervals_df['duration'],
                                            description=bad_intervals_df['description'])

        raw.set_annotations(bad_intervals_ann)

        raw.info['bads'] = bad_channels_df["bad_channels"].tolist()
        #raw.drop_channels(raw.info['bads'])
        #raw.plot(block=True)
        raw.interpolate_bads(reset_bads=True)

        data = raw.get_data()
        new_signals = preprocesser.drop_bad_intervals(data, raw_freq,
                                                      bad_intervals_df['start_time'],
                                                      bad_intervals_df['duration'])

        new_raw = mne.io.RawArray(new_signals, info=raw.info)




        #fig = mne.viz.plot_raw(raw, duration=raw.times[-1], block=False, show=False)
        #fig.savefig('ica_plots/raw_plot.png')


        random_state = 42
        num_comp = 20


        ica = mne.preprocessing.ICA(n_components=num_comp, random_state=random_state)
        ica.fit(new_raw)
        if not os.path.isfile(f"bads/{fname}/{fname}_bad_ica_components.csv"):
            #new_raw.plot(block=False)
            ica.plot_sources(new_raw, start=0, stop=200, picks=list(range(num_comp)), block=True)
            bad_ica_df = pd.DataFrame({"bad_ica_components": ica.exclude, "num_of_components": num_comp, "random_state": random_state})
            bad_ica_df.to_csv(f"bads/{fname}/{fname}_bad_ica_components.csv")

        bad_ica_df = pd.read_csv(f"bads/{fname}/{fname}_bad_ica_components.csv")
        bad_icas = bad_ica_df["bad_ica_components"].tolist()
        #new_raw.plot(block=False)
        #ica.plot_sources(new_raw, start=0, stop=200, picks=list(range(num_comp)), block=True)
        ica.exclude = bad_icas
        #new_raw.plot(block=False)
        ica.apply(new_raw)
        #new_raw.plot(block=True)



        new_data = new_raw.get_data()
        gfp = np.std(new_data, axis=0)
        peaks, _ = mne_microstates.find_peaks(gfp)

        microstates = categorize_spatial_correlation_sajat(new_data, peaks)
        microstate_df = pd.DataFrame({"microstates": microstates})
        microstate_df.to_csv(f"microstates/microstates_absolute_sajatraBackfitting/microstates_absolute_{fname}.csv")


        new_raw.pick_types(meg=False, eeg=True)
        maps, segmentation = mne_microstates.segment(new_raw.get_data(), n_states=4, random_state=random_state)
        plt.clf()
        plt.cla()
        plt.close()
        Path(f"microstates/plots/{fname}/").mkdir(parents=True, exist_ok=True)
        #mne_microstates.plot_maps(maps, new_raw.info, segmentation, f"microstates/plots/{fname}/{fname}")

        maps_dict = {"1": maps[0],
                     "2": maps[1],
                     "3": maps[2],
                     "4": maps[3],}
        maps_df = pd.DataFrame(maps_dict)
        maps_df.to_csv(f"microstates/maps/{fname}_maps.csv")




        """
        for i in range(len(ica.ch_names)):
        figs = ica.plot_properties(raw, picks=i, show=False)
            for fig_idx, fig in enumerate(figs):
            fig.savefig(f'ica_plots/{fname}_component_{i}_diagnostics_{fig_idx}.png')
        """

        """
       # EZ A BLOKK SAJÁT ELJÁRÁS, NEM LETT VÉGÜL HASZNÁLVA, NINCS KÖZE AZ ML EREDMÉNYEKHEZ
        new_signals = preprocesser.interpolate_signals(raw, biosemi_montage, chunk_fraction=16, std_limit=0.000007)
        new_raw = mne.io.RawArray(new_signals, info=raw.info)
        new_raw.plot(block=True)
       
        mne.viz.plot_raw_psd_topo(_bad_channels, raw.info, show=True, block=True)
        make_corr_matrices(new_raw.get_data(), fname)
        """
make_microstates(dirpath)
#sys.exit()


