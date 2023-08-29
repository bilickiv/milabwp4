import numpy as np
import copy


"""
It is called ZCM helper, but it is mainly operates with the area below humps, 
so it will work for other types of datas as well.
"""

def _get_switchlist(data, threshold=5):

    """
    switchlist([number_value, length, last_index][][])
    for example  if the original list looks like [n,n,n,n,n,n,n,n,k,k,k,k,k,m,m,m,m,........] then
    switchlist[(1, 8, 7,),(0, 5, 12),.....], where if threshold == 5, then k < 5 and m >= 5 and n >= 5
    """
    data_copy = copy.deepcopy(data)
    switch_list = []
    _length = 0
    data_copy = list(data_copy)

    if len(data_copy) < 1:
        return None
    LOWER = data_copy[0] < threshold
    _number = 0 if LOWER else 1

    for idx, value in enumerate(data_copy):
        if (value < threshold) == LOWER:
            _length += 1
        else:
            if _length == 0:
                _length = 1
            switch_list.append((_number, _length, idx - 1))
            _number = 0 if value < threshold else 1
            _length = 1
            LOWER = not LOWER
        if idx == len(data) - 1:
            switch_list.append((_number, _length, idx))
    return switch_list


def _construct_list_from_switchlist(switchlist):
    """
    constructs a characteristic list from the switchlist

    for example:
    switchlist = [(0,3,2),(1,3,5)]
    it returns [0,0,0,1,1,1]

    :param switchlist: given switchlist
    :return: list of values
    """
    return_list = []
    for u_val in switchlist:
        for i in range(u_val[1]):
            return_list.append(u_val[0])
    return return_list

def _switchlist_smoother(switchlist, threshold=10, target=0, left_window=20, right_window=20, MODE_OR=False, OR_TARGET=10):
    """
    :param switchlist: given switchlist
    :param threshold: maximum length to smooth to
    :param target: value to smooth
    :param left_window: minimum length for the target section at the left side
    :param right_window: minimum length for the target section at the right side
    :param MODE_OR: if True, then if a segment is not target value and the sum of the left and the right target segment
    is bigger than OR_TARGET then it smoothes that
    :return: smoothed switchlist

    if a non-target section has at least left_window-length target section to the left and
    right_window-length target section to the right, the non-target section becomes target section

    """
    switchlist_copy = copy.deepcopy(switchlist)

    for idx, u_val in enumerate(switchlist):
        if 0 < idx < len(switchlist) - 1:
            if not MODE_OR:
                if u_val[0] != target and u_val[1] < threshold and switchlist[idx-1][1] >= left_window and switchlist[idx+1][1] >= right_window:
                    switchlist_copy[idx] = (target, u_val[1], u_val[2])
            else:
                if u_val[0] != target and u_val[1] < threshold and switchlist[idx+1][1] + switchlist[idx-1][1] >= OR_TARGET:
                    switchlist_copy[idx] = (target, u_val[1], u_val[2])

    constr_list = _construct_list_from_switchlist(switchlist_copy)
    return_switchlist = _get_switchlist(constr_list, threshold=1)

    return return_switchlist

def _zcm_filter(zcm, zcm_threshold, threshold=10, target=0, given_constr_list=None):
    """
    :param zcm: list of zcm values
    :param zcm_threshold: if lower than this value, the sleep/wake algorithm scores sleep
    :param threshold: maximum length  the non target values (sections) to delete
    :param target: target value
    :param given_constr_list: provided constr_list in advance to make a switchlist, to use the filtering on
    :return: filtered zcm list and a characteristics list with the values (0,1,None), where the None sections has been cut off

    from the longest in length target section, if there's a section
     either to the right or to the left, which is not a target value
     and it's length is lower or equal than the threshold, it cuts the section out.

     for example:
     target == 0
     threshold == 5
     if the list is:
     [0,0 1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
     it becomes
     (0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]


     if the list is:
     [1,1,0,0 1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
     it becomes
     (0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]


     if the list is:
     [1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
     it becomes
     (1,1,1,1,1,1,1,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
     """

    if not given_constr_list:
        switchlist = _get_switchlist(zcm, zcm_threshold)
    else:
        switchlist = _get_switchlist(given_constr_list, 1)

    if len(switchlist) < 2:
        return zcm, _construct_list_from_switchlist(switchlist)

    index = 0
    max_target = switchlist[0]
    while max_target[0] != target:
        index += 1
        max_target = switchlist[index]
    index = 0

    for idx, u_val in enumerate(switchlist):
        if u_val[0] == target and u_val[1] > max_target[1]:
            max_target = u_val
            index = idx

    #how much it stepped to left
    non_target_counter_left = 1

    indexes_to_cut = []
    while index - non_target_counter_left >= 0:
        if switchlist[index - non_target_counter_left][1] <= threshold and \
                switchlist[index - non_target_counter_left][0] != target:
            indexes_to_cut.append(switchlist[index - non_target_counter_left])
        elif switchlist[index - non_target_counter_left][1] > threshold and \
                switchlist[index - non_target_counter_left][0] != target:
            break
        non_target_counter_left += 1

    # how much it stepped to right
    non_target_counter_right = 1

    while (index + non_target_counter_right < len(switchlist)):
        if switchlist[index + non_target_counter_right][1] <= threshold and \
                switchlist[index + non_target_counter_right][0] != target:
            indexes_to_cut.append(switchlist[index + non_target_counter_right])
        elif switchlist[index + non_target_counter_right][1] > threshold and \
                switchlist[index + non_target_counter_right][0] != target:
            break
        non_target_counter_right += 1

    constr_list = _construct_list_from_switchlist(switchlist)

    if indexes_to_cut:

        for val in indexes_to_cut:
            len_to_replace = val[1]
            constr_list[val[2] - val[1] + 1 :val[2]+1] = [None] * len_to_replace


        zcm_filtered = []
        for idx, val in enumerate(constr_list):

            if val is not None:
                if isinstance(zcm, list):
                    zcm_filtered.append(zcm[idx])
                else:
                    zcm_filtered.append(zcm.iloc[idx])

        return zcm_filtered, constr_list

    else:
        return zcm, constr_list


def _get_longest_zcm_target(zcm, zcm_threshold, target=0):
    """finds the longest section in a list where with the target value"""
    switchlist = _get_switchlist(zcm, zcm_threshold)

    max_target = switchlist[0]
    max_length = 0
    for idx, u_val in enumerate(switchlist):
        if u_val[0] == target and u_val[1] > max_length:
            max_target = u_val
            max_length = u_val[1]

    if max_length > 0:
        return max_target
    else:
        return None


def _get_all_zcm_target(zcm, zcm_threshold, target=0, limit :int=240, smooth=False, left_window=30, right_window=30, given_switchlist=None):
    """finds all the sections above the limit (in minutes) in a list with the target value
    :param zcm: list of zcm values
    :param zcm_threshold: if lower than this value, the sleep/wake algorithm scores sleep
    :param target: target value
    :param limit: returns all the target sections above the limit
    :param given_switchlist: provided switch_list in advance to use the filtering on with the zcm values
    :return: list of switchlist-entries

    """
    if not given_switchlist:
        switchlist = _get_switchlist(zcm, zcm_threshold)
    else:
        switchlist = given_switchlist
    if smooth:
        switchlist = _switchlist_smoother(switchlist, 0, target, left_window, right_window)
    targets = []

    for idx, u_val in enumerate(switchlist):
        if u_val[0] == target and u_val[1] > limit:
            targets.append(u_val)

    if len(targets) > 0:
        return targets
    else:
        return None



def aggregate_zcm_values(data, window_left=5, window_right=5):
    """aggregates the zcm values with a sliding window
    if a window can't be fitted entirely to the data, then it keeps the original value
    """
    aggregated_list = []
    for i in range(len(data)):
        if window_left <= i <= len(data) - window_right:
            aggregated_list.append(np.mean(data[i-window_left: i+window_right+1]))
        else:
            aggregated_list.append(data[i])
    return aggregated_list


def aggregate_zcm_values_ver2(data, window_left=5, window_right=5):
    """aggregates the values with a sliding window
    if a window can't be fitted entirely to the data, then it just uses that part of the window that fits
    """
    aggregated_list = []
    for i in range(len(data)):
        if window_left <= i <= len(data) - window_right - 1:
            aggregated_list.append(np.mean(data[i-window_left: i+window_right+1]))
        elif window_left > i:
            aggregated_list.append(np.mean(data[0: i+window_right+1]))
        elif window_right > len(data) - i - 1:
            right_border = len(data) - i + 1
            aggregated_list.append(np.mean(data[i - window_left: i + right_border]))

    return aggregated_list
