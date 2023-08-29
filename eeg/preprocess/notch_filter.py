from scipy import signal

def get_notch_filtered_data(raw_signal, samp_freq=8192, notch_freq=50, quality_factor=30):
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    filtered_signal = signal.filtfilt(b_notch, a_notch, raw_signal)
    return filtered_signal