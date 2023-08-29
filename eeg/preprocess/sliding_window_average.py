import numpy as np

def get_aggregated_values_data(data, window_left=1, window_right=1):
    """aggregates the values with a sliding window"""
    aggregated_list = []
    for i in range(len(data)):
        if window_left <= i <= len(data) - window_right - 1:
            aggregated_list.append(np.mean(data[i-window_left: i+window_right+1]))

        elif window_left > i:
            aggregated_list.append(np.mean(data[0: i+window_right+1]))

        elif window_right > len(data) - i - 1:
            right_border = len(data) - i + 1
            aggregated_list.append(np.mean(data[i - window_left: i + right_border]))

    return np.array(aggregated_list)
