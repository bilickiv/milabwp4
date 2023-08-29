from scipy import signal

def get_resampled_data(raw_data, base_frequency=8192, target_frequency=500):
    epochs = len(raw_data)//base_frequency
    new_data = signal.resample(raw_data, target_frequency * epochs)
    return new_data