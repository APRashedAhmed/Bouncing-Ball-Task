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

def plot_effective_stats(
        df_data,
        tight_layout=True,
        suptitle=None,
):
    if suptitle is None:
        suptitle = f"Task Statstics for {len(df_data)} Videos"
    
    # Create the subplots
    palette = get_color_palette(
        ["Low", "High"],
        (("Blues", 1), ("Reds", 1)),
        linspace_range=(0.75, 1),
    )
    palette_trial = get_color_palette(
        ["Catch", "Straight", "Nonwall", "Bounce"],
        (("Greens", 1), ("Blues", 1), ("Wistia", 1), ("Reds", 1)),
        linspace_range=(0.75, 1),
    )
    palette_contingency = get_color_palette(
        ["Low", "Medium", "High"],
        (("Blues", 1), ("Wistia", 1), ("Reds", 1)),
        linspace_range=(0.75, 1),
    )

    plot_params = [
        [
            (
                "PCCNVC_effective",
                "Observed Hazard Rates",
                "Effective Hazard Rate Bins",
                {
                    "hue": "Hazard Rate",
                    "legend": True,
                    "palette": palette,
                }
            ),
            (
                "PCCOVC_effective",
                "Observed Trial Contingency",
                "Effective Contingency Bins",
                {
                    "hue": "Contingency",
                    "palette": palette_contingency,
                    "legend": True,
                },
            ),
            (
                "PVC_effective",
                "Observed Trial Random Bounce",
                "Effective Random Bounce Bins",
                {
                    "hue": "trial",
                    "palette": palette_trial,
                    "legend": True,
                },
            ),
            (
                "length",
                "Distribution of Video Lengths",
                "Video Length Bins",
                {
                    "hue": "trial",
                    "legend": True,
                    "palette": palette_trial,
                }
            ),
        ],
        [
            (
                "Color Change Random",
                "Number of Random Color Changes",
                "Random Color Change Bins",
                {
                    "hue": "Hazard Rate",
                    "legend": False,
                    "discrete": True,
                    "palette": palette,
                }
            ),
            (
                "Color Change Bounce",
                "Number of Bounce Color Changes",
                "Bounce Color Changes",
                {
                    "discrete": True,
                    "hue": "Contingency",
                    "palette": palette_contingency,
                    "legend": False,
                }
            ),
            (
                "Random Bounces",
                "Number of Random Bounces",
                "Number of Random Bounces",
                {
                    "discrete": True,
                    "hue": "trial",
                    "palette": palette_trial,
                    "legend": False,
                }
            ),
            (
                "Bounces",
                "Number of Wall Bounces",
                "Number of Wall Bounces",
                {
                    "discrete": True,
                    "hue": "trial",
                    "palette": palette_trial,
                    "legend": True,
                }
            ),
        ],
    ]

    rows = 2
    fig, axes = plt.subplots(
        rows,
        len(plot_params[0]),
        figsize=(len(plot_params[0])*4, rows*4),
    )

    for i, row_plots in enumerate(plot_params):
        for j, (col, title, xlabel, plot_dict) in enumerate(row_plots):
            ax = axes[i, j]
            sns.histplot(
                df_data,
                x=col,
                ax=ax,
                **plot_dict,
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            if j != 0:
                ax.set_ylabel(None)

    plt.suptitle(suptitle)
    if tight_layout:
        plt.tight_layout()

    return fig, axes
