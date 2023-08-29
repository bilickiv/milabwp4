import numpy as np
import pandas as pd
import scipy
import os



def smooth_microstates(microstates, window_length=5, round=1):


    microstates_temp = microstates
    most_freq_val = lambda x: scipy.stats.mode(x)[0][0]
    for i in range(round):
        smoothed = []
        for idx, state in enumerate(microstates_temp):
            if idx < window_length:
                wl_l = idx
                wl_r = window_length
            elif idx > len(microstates_temp) - window_length - 1:
                wl_l = window_length
                wl_r = len(microstates_temp) - (idx + 1)
            else:
                wl_l = window_length
                wl_r = window_length
            val = microstates_temp[idx - wl_l: idx + wl_r]
            smoothed.append(most_freq_val(val))
        microstates_temp = smoothed

    return microstates_temp


def read_microstates():
    
    dirpath = r"I:\Munka\Masodik\mne_work\microstates\microstates_absolute"
    dirpath_out =r"I:\Munka\Masodik\mne_work\microstates\microstates_absolute\smoothed"
    for file in os.listdir(dirpath):
        if file.endswith(".csv"):
            print(f"{dirpath}/{file} is in the making!")
            df = pd.read_csv(f"{dirpath}/{file}")
            smoothed = smooth_microstates(df["microstates"].values)
            df2 = pd.DataFrame({"microstates": smoothed})
        
            df2.to_csv(f"{dirpath_out}/smoothed_{file}")
        
read_microstates()














