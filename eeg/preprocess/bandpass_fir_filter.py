from scipy import signal

def get_bandpass_fir_filtered_data(raw_signal, f1=0.1, f2=40, fs=8192, denominator_coefficient = [1]):
    nyq = 0.5 * fs
    taps = signal.firwin(127, [f1, f2], nyq=nyq, pass_zero=False, window="hamming", scale=False)
    bp_filtered_signal = signal.lfilter(taps, denominator_coefficient, raw_signal)
    return bp_filtered_signal