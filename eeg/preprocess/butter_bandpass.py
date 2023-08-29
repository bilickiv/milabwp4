from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5, btype="bandpass"):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, btype="bandpass"):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    y = lfilter(b, a, data)
    return y


def get_butter_bandpass_data(data, order=3, fs=8192, lowcut=0.1, highcut=40, btype="bandpass"):
    y = butter_bandpass_filter(data, lowcut, highcut, fs, order=order, btype=btype)
    return y
