from PIL import Image, ImageColor
import pandas as pd
import os

def plot_states():
    df = pd.read_csv("characteristics/eeg_sequence_transitions.csv")
    _transitions = df["transitions"].values.tolist()
    _ids = df["id"].values.tolist()
    _categories = df["category"].values.tolist()

    for ix, tr in enumerate(_transitions):
        print(f"{_ids[ix]}'s plotting of categories is in the making!")
        tr_array = tr.split(",")[:-1]
        classes = [t[0] for t in tr_array]
        lengths = [int(t[1:]) for t in tr_array]

        category = _categories[ix]

        state_string = ""

        for idx, cl in enumerate(classes):
            state_string += cl * lengths[idx]

        len_of_st = len(state_string)

        smallest_rect_num = 0

        #smallest rectangle that can contain enough pixel
        while smallest_rect_num * smallest_rect_num < len_of_st:
            smallest_rect_num += 1

        im = Image.new('RGB', (smallest_rect_num, smallest_rect_num))
        row = -1

        color_dict = {"A": "blue", "B": "red", "C": "yellow", "D": "green"}
        for idx, state in enumerate(state_string):
            if idx % smallest_rect_num == 0:
                row += 1
            im.putpixel((idx%smallest_rect_num, row), ImageColor.getcolor(color_dict[state], 'RGB'))
        im.save(f"plots/plot_states_colored/{category}/{_ids[ix]}_states_colored_{_categories[ix]}.png")

def combine_states():

    for _dir in ["1", "2", "3"]:
        images = [Image.open(f"plots/plot_states_colored/{_dir}/{x}") for x in os.listdir(f"plots/plot_states_colored/{_dir}")]

        smallest_rect_num = 0
        while smallest_rect_num * smallest_rect_num < len(images):
            smallest_rect_num += 1

        max_width = 0
        for idx, im in enumerate(images):
            if im.size[0] > max_width:
                max_width = im.size[0]

        total_width_picture = max_width * smallest_rect_num + (smallest_rect_num - 1) * 30
        new_im = Image.new('RGB', (total_width_picture, total_width_picture))

        x_offset = 0
        y_offset = 0
        for idx, im in enumerate(images):
            new_im.paste(im, (x_offset, y_offset))
            if idx != 0 and idx%smallest_rect_num == 0:
                y_offset += 30 + max_width
                x_offset = 0
            else:
                x_offset += 30 + max_width

        new_im.save(f"plots/plot_states_colored/combined/{_dir}_states_colored_combined.png")
