import copy
import itertools
import mne
import numpy as np
import matplotlib.pyplot as plt
import mne_microstates
import os
import pandas as pd
import matplotlib
import preprocesser as preprocesser
from mne.preprocessing import annotate_muscle_zscore
from pathlib import Path
import scipy
import fnmatch
import sys
from time import sleep
from file_preprocessing import connect_microstates
matplotlib.use('TkAgg')


class ICAContainer():

    def __init__(self):
        self.id = ""
        self.bad_icas_index = set()
        self.spatial_limit = 0.3
        self.spatial_dict = None
        self.best_matches = None
        self.ica_parent_list = None

    def make_parents(self, comblist):

        ica_parent_list = []
        for cl in comblist:
            if len(cl) == 1:
                ica_parent_list.append({"key": cl, "parents": [], "score": 99999999})
            else:
                parentlist = []
                for cl2 in comblist:
                    if len(cl2) + 1 == len(cl) and cl2.issubset(cl):
                        parentlist.append(cl2)
                ica_parent_list.append({"key": cl, "parents": parentlist, "score": 99999999})

        return ica_parent_list

    def check_parent_scores(self, icadict):
        if not icadict["parents"]:
            return True
        for parent in icadict["parents"]:
            for elem in self.ica_parent_list:
                if elem["key"] == parent:
                    if elem["score"] < icadict["score"]:
                        return False
        return True

    def remove_children(self, icadict):
        if self.check_parent_scores(icadict) is False:
            bad_children = [icadict["key"]]
            for elem in self.ica_parent_list:
                for parent in elem["parents"]:
                    if icadict["key"] == parent:
                        bad_children.append(elem["key"])
                        break

            _ica_parent_list = []
            deleted = []
            for elem in copy.deepcopy(self.ica_parent_list):
                if elem["key"] not in bad_children:
                    _ica_parent_list.append(elem)
                else:
                    deleted.append(elem["key"])
            print(f"The following future nodes were deleted: {deleted}")
            print(f"The old length of the parentlist: {self.ica_parent_list}")
            self.ica_parent_list = _ica_parent_list
            print(f"The new length of the parentlist: {self.ica_parent_list}")
        else:
            pass
def rearrange_ica_components(ica, channel_names):
    ica_components = ica.get_components()
    ic_ch_names = ica.ch_names

    return_list = []
    for chn in channel_names:
        for ix, ic in enumerate(ica_components):
            if ic_ch_names[ix] == chn:
                return_list.append(ic)

    return np.array(return_list)

def compare_microstates_spatial_correlation(maps_df, ref_df=r"calculated_avg_microstates_TEST.csv" ):
    avg_microstates = pd.read_csv(ref_df)
    microstates = maps_df

    microstate_cols = microstates.columns.tolist()

    spatial_dict = {}

    for amc in avg_microstates:
        if "microstate" in amc:
            av_microstate = avg_microstates[amc].values
            sp_d = {}
            for inv in [False, True]:
                for mc in microstate_cols:
                    if mc != "Unnamed: 0":
                        microstate = microstates[mc].values
                        if inv:
                            spatial_corr = scipy.spatial.distance.correlation(av_microstate, microstate * - 1)
                            sp_d[f"{mc}_inv"] = spatial_corr
                        else:
                            spatial_corr = scipy.spatial.distance.correlation(av_microstate, microstate)
                            sp_d[mc] = spatial_corr
                spatial_dict[f"{amc}"] = sp_d

    for key, values in spatial_dict.items():
        print(f"{key} - {values}")
    best_matches = []
    for key, values in spatial_dict.items():
        best_match = 999999
        best_microstate = None
        for key2, values2 in values.items():
            if values2 < best_match:
                best_match = values2
                best_microstate = key2
        print(f"The best match for {key} is {best_microstate} with a value of {best_match}")
        best_matches.append({"key": key, "best_microstate": best_microstate, "best_microstate_score": best_match})

    return spatial_dict, best_matches

def correct_best_states(spatial_dict, best_matches, spatial_limit, verbose=False):

    best_states = []
    for bm in best_matches:
        best_states.append(bm["best_microstate"])

    #megnézzük melyik kulcshoz hány state tartozik, majd kitöröljük azokat amikhez csak egy tartozik, mert azokkal nincs probléma
    counters = {}
    for bt in best_states:
        counter = counters.get(bt[0], 0)
        counter += 1
        counters[bt[0]] = counter
    if verbose:
        print(counters)
    possible_states_dict = {}

    for ct in copy.deepcopy(counters):
        if counters[ct] < 2:
            del counters[ct]
    if verbose:
        print(counters)



    good_states = [bm["key"] for bm in best_matches if bm["best_microstate"][0] not in counters.keys()]
    if verbose:
        print(good_states)

    #kiszedjük, hogy melyik olyan kulcshoz (1-n) ami több statehez (microstate_A, ......) is jó volt, melyik lehetséges statek tartoznak a spatial limit alatt
    for ct in counters:
        possible_states_list = []
        possible_states_list_inv = []
        key1 = ct
        key2 = f"{ct}_inv"
        for sp in spatial_dict:
            if sp in good_states:
                continue
            lim1 = spatial_dict[sp][key1]
            lim2 = spatial_dict[sp][key2]
            if lim1 < spatial_limit:
                possible_states_list.append({"key": sp, "value":lim1})
            if lim2 < spatial_limit:
                possible_states_list_inv.append({"key": sp, "value": lim2})
        possible_states_dict[key1] = possible_states_list
        possible_states_dict[key2] = possible_states_list_inv

    if verbose:
        print(possible_states_dict)

    selected_states = {}
    added_states = []
    for key, val in possible_states_dict.items():
        if val:
            sorted_val = val
            sorted_val.sort(key=lambda x: x['value'])

            ct = 0
            if verbose:
                print(sorted_val)
            while ct < len(sorted_val):
                if sorted_val[ct]["key"] not in added_states:
                    selected_states[key] = sorted_val[ct]
                    added_states.append(sorted_val[ct]["key"])
                    break
                else:
                    ct += 1

    if verbose:
        print(selected_states)

    best_matches_return = []
    for bs in best_matches:
        if bs["key"] in good_states:
            best_matches_return.append(bs)

    for ss in selected_states:
        dta = {"key": selected_states[ss]["key"], "best_microstate": ss,
               "best_microstate_score": selected_states[ss]["value"]}
        best_matches_return.append(dta)
    if verbose:
        print("ASD")
        print(best_matches_return)
        print("ASD")
        print()
        print()

    return best_matches_return

def make_microstates(dirpath, outdir="automated_test"):



    channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

    _bad_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4',
                 'EXG5', 'EXG6', 'EXG7', 'EXG8']



    cnter = 0
    for f in os.listdir(dirpath):

        ICAContainers = []
        fname = f.split(".")[0]

        if not f.endswith(".edf") and not f.endswith(".bdf"):
            continue

        if  (os.path.isdir(f"{outdir}/bads/{fname}") and len(fnmatch.filter(os.listdir(f"{outdir}/bads/{fname}"), '*.csv')) == 3):
            continue

        print(f"{f} is in the making!")

        fname = f.split(".")[0]

        try:
            raw = mne.io.read_raw_edf(f"{dirpath}/{f}", exclude=_bad_channels, eog=channel_names, stim_channel='Status',
                                      preload=True)
        except Exception as e:
            print(e)
            continue
        cnter += 1
        if cnter > 1000:
            continue


        # hanyad részére downsampleölje
        resample_fraction = 1
        resample_freq = raw.info['sfreq'] // resample_fraction
        raw_freq = raw.info['sfreq']
        if resample_fraction > 1:
            raw.resample(resample_freq, npad="auto")
            raw_freq = raw_freq / resample_fraction
            # ha nem egész szám
            if not raw_freq % 1 == 0:
                sys.exit()

        # mesterséges epoch, milyen időközönként legyen
        epoch_length = 512 // resample_fraction
        overlap = 0


        print(raw.get_channel_types())
        print(raw.ch_names)
        biosemi_montage = mne.channels.make_standard_montage('standard_1020')

        channel_types = {ch_name: 'eeg' for ch_name in raw.ch_names}
        #channel_types['Status'] = 'stim'
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
        #raw.add_events(events)
        #raw.set_annotations(epochs.annotations)

        annot, score_muscles = annotate_muscle_zscore(raw, filter_freq=(0.1, 0.3), min_length_good=2, ch_type='eeg', threshold=4)

        print(score_muscles)
        new_channel_data = np.random.rand(1, raw.n_times)  # Replace this with your actual data
        new_ch_info = mne.create_info(ch_names=['muscle_scores'], sfreq=raw.info['sfreq'], ch_types=['misc'])
        new_channel = mne.io.RawArray([score_muscles], info=new_ch_info)
        #raw.add_channels([new_channel], force_update_info=True)
        raw.set_annotations(annot)
        #raw.plot(block=True)

        Path(f"{outdir}/bads/{fname}").mkdir(parents=True, exist_ok=True)
        bad_intervals = raw.annotations[raw.annotations.description == 'BAD_muscle']
        bad_interval_starts = bad_intervals.onset
        bad_interval_durations = bad_intervals.duration
        bad_interval_descriptions = bad_intervals.description
        bad_channels = raw.info['bads']
        # Create a pandas DataFrame with the bad interval information
        bad_intervals_df = pd.DataFrame({'start_time': bad_interval_starts,
                                         'duration': bad_interval_durations,
                                         'description': bad_interval_descriptions})
        bad_channels_df = pd.DataFrame({'bad_channels': bad_channels})

        bad_intervals_df.to_csv(f"{outdir}/bads/{fname}/{fname}_bad_intervals.csv")
        bad_channels_df.to_csv(f"{outdir}/bads/{fname}/{fname}_bad_channels.csv")

        data = raw.get_data()
        new_signals = preprocesser.drop_bad_intervals(data, raw_freq,
                                                      bad_interval_starts,
                                                      bad_interval_durations)

        new_raw = mne.io.RawArray(new_signals, info=raw.info)

        # --------------------- ICA ----------------------

        random_state = 42
        cycle_limit = 1
        num_comp = 10
        cycle = 0
        numstates = 4
        icacont = None
        icaexclude = []
        while cycle < cycle_limit:
            cycle += 1
            bad_kmeans_maps = []
            # ELSE ÁGAT MAJD VALSZEG MÁSHOGY KELL
            if not ICAContainers:
                icacont = ICAContainer()

            new_raw_temp = copy.deepcopy(new_raw)
            ica = mne.preprocessing.ICA(n_components=num_comp, random_state=random_state)
            ica.fit(new_raw_temp)

            # ha az ezalatti new_raw.plot() sort lefuttatod akkor polotolja megjelölve azokat az ICA componenseket amik rosszak
            # new_raw.plot(block=False)
            #ica.plot_sources(new_raw_temp, start=0, stop=200, picks=list(range(num_comp)), block=True)
            # new_raw.plot(block=False)


            print("ICA EXCLUDE")
            print("ICA EXCLUDE")
            print("ICA EXCLUDE")
            print("ICA EXCLUDE")
            if cycle > 1:
                icaexclude = list(icacont.ica_parent_list[cycle - 2]["key"])
                print(icaexclude)
                print(icaexclude)
                print(icaexclude)
            print("ICA EXCLUDE")
            print("ICA EXCLUDE")
            print("ICA EXCLUDE")
            if cycle > 1:
                ica.exclude = icaexclude
            ica.apply(new_raw_temp)
            # new_raw.plot(block=True)

            new_data = new_raw_temp.get_data()
            gfp = np.std(new_data, axis=0)
            peaks, _ = mne_microstates.find_peaks(gfp)

            new_raw_temp.pick_types(meg=False, eeg=True)

            maps, segmentation = mne_microstates.segment(new_raw_temp.get_data(), n_states=numstates, random_state=random_state)

            maps_dict = {"1": maps[0],
                         "2": maps[1],
                         "3": maps[2],
                         "4": maps[3], }
            maps_df = pd.DataFrame(maps_dict)


            # ---------------- ITT ELLENŐRZI, HOGY VAN-E OLYAN map AMI NEM HASONLÍT A REFERENCIA MAPPEK VALAMELYIKÉHEZ

            #ami benne marad ebbe, ehhez kell az ica-kat hasonlítani
            possible_states = set(range(1,numstates + 1,1))
            removed_states = []

            #referenciához hasonlíja a kiszámolt mapeket, és visszaadja mindegyik map mindegyik referenciához
            #tartozó spatial értékét, és azt is, hogy melyik map melyik referenciához hasonlít legjobban
            spatial_dict, _best_matches = compare_microstates_spatial_correlation(maps_df)
            #best_matches.append({"key": key, "best_microstate": best_microstate, "best_microstate_score": best_match})
            best_matches = correct_best_states(spatial_dict, _best_matches, icacont.spatial_limit)
            #minden maphez a legjobbat szedi ki, de mivan ha ugyanaz a legjobb tartozik több maphez is?  spatial_dict -tel lehetne ellenőrizni - egyelőre ellenőrizve

            if cycle > 1:
                for elem in icacont.ica_parent_list:
                    if elem["key"] == set(icaexclude):
                        elem["score"] = sum(bm["best_microstate_score"] for bm in best_matches)
                        icacont.remove_children(elem)

            for bm in best_matches:
                if bm["best_microstate_score"] <= icacont.spatial_limit:
                    bm_code = bm["best_microstate"]
                    code = int(bm_code[0])
                    if code not in removed_states:
                        possible_states.remove(code)
                        removed_states.append(code)

            print("A MARADT POSSIBLE STATE")
            print(best_matches)
            print(possible_states)
            print("A MARADT POSSIBLE STATE VÉGE")
            for ps in possible_states:
                bad_kmeans_maps.append(maps[ps - 1])

            #----------------------  ELLENŐRZÉSNEK VÉGE ----------------------------------------------------

            #ITT DÖNTI EL, HOGY A ROSSZUL KIJÖTT MAPHEZ MELYIK ICA STATE INDEXE HASONLÍT A LEGJOBBAN
            # VALÓSZíNŰLEG ITT KÉNE VALAMI GRÁFOT FELÉPíTENI AMÚGY

            #MONDJUK KIVÁLASZTJA A 4 LEGJOBBAN HASONLÍTÓT, EBBŐL FELÉPÍT EGY COMBINATION -T, MAX N HOSZÚ
            #ÉS ELLENŐRZI MINDRE, HA MEGFELEL A FELTÉTELNEK AKKOR AZT VÁLASZTJA KI,
            #HA EGY SEM FELEL MEG A FELTÉTELNEK AKKOR A LEGJOBBAT KIVÁLASZTJA

            if bad_kmeans_maps:
                if cycle > 1:
                    cycle_limit = len(icacont.ica_parent_list)
                    print(f"New cycle limit: {cycle_limit}")
                    sleep(3)
                elif cycle == 1:
                    rearranged_comps = rearrange_ica_components(ica, channel_names)
                    for bm in bad_kmeans_maps:
                        spatial_scores = []
                        for ix, rearc in enumerate(rearranged_comps.T):
                            spatial_corr = scipy.spatial.distance.correlation(rearc, bm)
                            spatial_corr_inv = scipy.spatial.distance.correlation(rearc * - 1, bm)


                            best_spatial_score = min(spatial_corr, spatial_corr_inv)
                            spatial_scores.append({"index": ix, "spatial_corr": best_spatial_score})

                        sorted_list = sorted(spatial_scores, key=lambda x: x["spatial_corr"])
                        print("SORTED LIST!")
                        print(sorted_list)
                        print()
                        selected_indexes = [item["index"] for item in sorted_list[:5]]
                        icacont.bad_icas_index.update(*[selected_indexes])

                    icacont.spatial_dict = spatial_dict
                    icacont.best_matches = best_matches
                    comblist = []
                    for r in range(1, num_comp):
                        comblist.extend(itertools.combinations(icacont.bad_icas_index, r))
                    comblist = [set(c) for c in comblist]
                    icacont.ica_parent_list = icacont.make_parents(comblist)
                    cycle_limit = len(icacont.ica_parent_list)
                    print(f"New cycle limit: {cycle_limit}")
                    print(icacont.ica_parent_list)
                    sleep(3)
                    ICAContainers.append(icacont)
            else:
                new_raw = copy.deepcopy(new_raw_temp)
                break

        if cycle > 1:
            new_raw_temp = copy.deepcopy(new_raw)
            ica = mne.preprocessing.ICA(n_components=num_comp, random_state=random_state)
            ica.fit(new_raw_temp)
            best_dict = min(icacont.ica_parent_list, key=lambda x: x["score"])
            icaexclude = best_dict["key"]
            print(f"Not a single good enough state found! But selecting the best: {best_dict} ")
            ica.exclude = icaexclude
            ica.apply(new_raw_temp)

            maps, segmentation = mne_microstates.segment(new_raw_temp.get_data(), n_states=numstates,
                                                         random_state=random_state)

        Path(f"{outdir}/bads/{fname}").mkdir(parents=True, exist_ok=True)
        bad_ica_df = pd.DataFrame({"bad_ica_components": list(icaexclude), "num_of_components": num_comp, "random_state": random_state})
        bad_ica_df.to_csv(f"{outdir}/bads/{fname}/{fname}_bad_ica_components.csv")
        plt.clf()
        plt.cla()
        plt.close()
        Path(f"{outdir}/microstates/plots/{fname}/").mkdir(parents=True, exist_ok=True)
        mne_microstates.plot_maps(maps, new_raw.info, segmentation, f"{outdir}/microstates/plots/{fname}/{fname}")
        maps_dict = {}
        for i in range(len(maps)):
            maps_dict[f"{i+1}"] = maps[i]

        maps_df = pd.DataFrame(maps_dict)
        Path(f"{outdir}/microstates/maps/").mkdir(parents=True, exist_ok=True)
        maps_df.to_csv(f"{outdir}/microstates/maps/{fname}_maps.csv")
        #sys.exit()
        #ÖSSZEHASONLíANI SPATIAL CORRALATIONNAL AZ average_microstate -eket, meg amik kijöttek. Ha van olyan ami nagyon eltér, akkor megnézni hogy
        #VAN-E OLYAN ICA componens ami hasonlít az eltérésre? Ha igen, akkor exclude-olni. És újranézni. Addig amíg nem jön ki lehetőleg mind a 4-re jó.



if __name__ == "__main__":
    #mappa elérési útja ahol azok a .bdf/.edf fájlok vannak amiken át kell mennie
    dirpath = "../converter/files/repOD"

    #compare_microstates_spatial_correlation()
    make_microstates(dirpath)
    # sys.exit()
    connect_microstates()


