import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_color_palette(
    columns,
    color_number_tup,
    linspace_range=(0.5, 1),
    linspace_offset=1,
):
    color_list = []
    for i, (color, number) in enumerate(color_number_tup):
        cmap = sns.color_palette(color, as_cmap=True)

        if isinstance(number, int):
            num = number + linspace_offset
        elif isinstance(number, tuple):
            num = number[0] + linspace_offset

        color_array = [cmap(x) for x in np.linspace(*linspace_range, num=num)]

        if isinstance(number, int):
            color_list += [color_array[i] for i in range(number)]
        elif isinstance(number, tuple):
            color_list += [color_array[number[1]]]

    return {col : color for col, color in zip(columns, color_list)}
