from collections import Counter
from collections.abc import Iterable
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from bouncing_ball_task.bouncing_ball import BouncingBallTask


def print_task_summary(meta, use_logger=True):
    if use_logger:
        out_func = logger.info
    else:
        out_func = print
        
    length_trials = meta["length_trials_ms"] / 60000
    length_trials_min = int(length_trials)
    length_trials_s = np.round((length_trials % 1) * 60, 1)
    out_func("Dataset Generation Summary")
    out_func(
        f'  Num Total Trials: {meta["num_trials"]} ({length_trials_min} min {length_trials_s} sec)'
    )
    max_key_len = max([len(key) for key in meta.keys()])

    for key, val in meta.items():
        if key == "num_trials":
            continue
        key += ":"
        out_func(f"    {key:<{max_key_len + 2}}{val}")


def print_type_stats(
        trials,
        trial_type,
        duration,
        use_logger=True,
        return_str=False,
):
    if use_logger:
        out_funcs = [logger.info, logger.debug]
    else:
        out_funcs = [print] * 2 
    
    position, velocity, color, pccnvc, pccovc, pvc, meta = zip(*trials)
    stats_comb = [f"{nvc}-{ovc}" for nvc, ovc in zip(pccnvc, pccovc)]
    list_messages = []

    trial_type_lengths_s = (
        np.array([m["length"] for m in meta]) * duration
    ) / 1000
    trial_type_lengths_min = trial_type_lengths_s / 60
    length_trial_total_min = int(sum(trial_type_lengths_min))
    length_trial_total_s = np.round((sum(trial_type_lengths_min) % 1) * 60, 1)

    length_trial_s_min = np.round(min(trial_type_lengths_s), 1)
    length_trial_s_max = np.round(max(trial_type_lengths_s), 1)

    msg = f"  Num {trial_type.title()} Trials: {len(trials)} ({length_trial_total_min} min {length_trial_total_s} sec)"
    if return_str:
        list_messages.append(msg)
    else:
        out_funcs[0](msg)

    trial_type_stats = [
        ("Min Video Length (s):", length_trial_s_min),
        ("Max Video Length (s):", length_trial_s_max),
        ("Color Splits:", Counter(np.argmax(color, axis=1))),
        ("pccnvc Splits:", Counter(pccnvc)),
        ("pccovc Splits:", Counter(pccovc)),
        ("Stat Comb Splits:", Counter(stats_comb)),
    ]

    if trial_type.lower() != "catch":
        trial_type_stats += [
            ("End time Splits:", Counter(m["idx_time"] for m in meta)),
            ("Left/Right Splits:", Counter(m["side_left_right"] for m in meta)),
            ("Top/Bottom Splits:", Counter(m["side_top_bottom"] for m in meta)),
            ("Velocity Splits:", Counter(m["idx_velocity_y"] for m in meta)),
        ]

    if trial_type.lower() == "bounce":
        trial_type_stats += [
            ("Final y position splits", Counter(m["idx_y_position"] for m in meta)),
        ]
        df = pd.DataFrame(meta)
        for idx_x, df_idx in df.groupby("idx_time"):
            trial_type_stats += [
                (
                    f"x={idx_x} - Final y position splits",
                    Counter(df_idx["idx_y_position"]),
                )
            ]

    max_desc_len = max([len(desc) for (desc, _) in trial_type_stats])
    for (description, value) in trial_type_stats:
        if isinstance(value, Counter):
            total = value.total()
            value_str = f"({', '.join([str(float(np.round(val[1] / total, 2))) for val in sorted(value.items(), key=lambda pair: pair[0])])})"
        else:
            value_str = f"{value}"
        msg = f"    {description:<{max_desc_len + 2}}{value_str}"
        if return_str:
            list_messages.append(msg)
        else:
            out_funcs[1](msg)

    if return_str:
        return list_messages


def print_block_stats(df_data, dict_metadata, duration, use_logger=True):
    if use_logger:
        out_funcs = [logger.info, logger.debug, logger.trace]
    else:
        out_funcs = [print] * 3
    
    # blocks, meta_blocks = zip(*params_blocks)
    num_blocks = dict_metadata['num_blocks']
    block_lengths = [
        block['num_trials'] for _, block in
        dict_metadata['blocks'].items()
    ]
    out_funcs[0](f"  Num blocks: {num_blocks} - Lengths = {block_lengths}")
    
    for idx_block, df_block in df_data.groupby("Dataset Block"):
        metadata_block = dict_metadata["blocks"][idx_block]
        # position, velocity, color, pccnvc, pccovc, pvc, meta = zip(*block)

        stats_comb = [
            f"{nvc}-{ovc}" for nvc, ovc in
            df_block[["PCCNVC", "PCCOVC"]].values
        ]
        # stats_comb = [f"{nvc}-{ovc}" for nvc, ovc in zip(pccnvc, pccovc)]

        assert len(df_block) == metadata_block["num_trials"]

        length_block_min = int(metadata_block["length_block_min"])
        length_block_s_rem = np.round((metadata_block["length_block_min"] % 1) * 60)

        out_funcs[1](
            f"    Block {idx_block} - {len(df_block)} videos "
            f"({length_block_min} min {length_block_s_rem} sec)"
        )

        # import ipdb; ipdb.set_trace()
        block_stats = [
            ("Min Video Length (s):", df_block["length_ms"].min() / 1000),
            ("Max Video Length (s):", df_block["length_ms"].max() / 1000),
            ("Trial type Counts:", Counter(df_block["trial"].values)),
            ("Color Counts:", Counter(df_block["Final Color"].values)),
            ("pccnvc Counts:", Counter(df_block["PCCNVC"])),
            ("pccovc Counts:", Counter(df_block["PCCOVC"])),
            ("stats comb Counts:", Counter(stats_comb)),
        ]
        
        max_desc_len = max([len(desc) for (desc, _) in block_stats])
        
        for (description, value) in block_stats:
            if isinstance(value, Counter):
                total = value.total()
                value_str = f"({', '.join([str(float(np.round(val[1] / total, 2))) for val in sorted(value.items(), key=lambda pair: pair[0])])})"
            else:
                value_str = f"{value}"
            out_funcs[2](f"      {description:<{max_desc_len + 2}}{value_str}")


def plot_params(
    params,
    size_frame: Iterable[int] = (256, 256),
    sequence_length=60,
    duration=40,
    target_future_timestep=1,
    ball_radius: int = 10,
    dt: float = 0.1,
    mask_center: float = 0.5,
    mask_fraction: float = 1 / 3,
    mask_color=(127, 127, 127),
    min_t_color_change=15,
    sample_mode="parameter_array",
    target_mode="parameter_array",
    save_target=True,
    sequence_mode="reverse",
    debug=True,
    save_animation=False,
    display_animation=True,
    mode="combined",
    multiplier=2,
    include_timestep=True,
    as_mp4=True,
):

    batch_size = len(params)
    (
        initial_position,
        initial_velocity,
        initial_color,
        pccnvc,
        pccovc,
        pvc,
        meta,
    ) = zip(*params)
    task = BouncingBallTask(
        size_frame=size_frame,
        sequence_length=sequence_length,
        ball_radius=ball_radius,
        target_future_timestep=target_future_timestep,
        dt=dt,
        batch_size=batch_size,
        sample_mode=sample_mode,
        target_mode=target_mode,
        mask_center=mask_center,
        mask_fraction=mask_fraction,
        mask_color=mask_color,
        sequence_mode=sequence_mode,
        debug=debug,
        probability_velocity_change=pvc,
        probability_color_change_no_velocity_change=pccnvc,
        probability_color_change_on_velocity_change=pccovc,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        initial_color=initial_color,
        min_t_color_change=min_t_color_change,
    )
    initial_params = np.concatenate(
        [np.array(initial_position), np.stack(initial_color)],
        axis=1,
    )
    assert np.all(
        np.isclose(np.array(initial_position), task.sequence[-1][1][:, :2])
    )
    assert np.all(
        np.isclose(np.stack(initial_color), task.sequence[-1][1][:, 2:])
    )
    samples, targets = zip(*[x for x in task])
    assert np.all(
        np.isclose(np.array(initial_position), task.sequence[-1][1][:, :2])
    )
    assert np.all(
        np.isclose(np.stack(initial_color), task.sequence[-1][1][:, 2:])
    )

    samples = np.array(samples)
    targets = np.array(targets)

    for i in range(batch_size):
        print(np.array(params[i][0]) * multiplier, params[i][1:])
        assert np.all(
            np.isclose(np.array(initial_position[i]), targets[-1][i, :2])
        )
        assert np.all(
            np.isclose(np.stack(initial_color[i]), targets[-1][i, 2:])
        )

        _, _ = task.animate(
            arrays=(targets[:, i, :]),
            path_dir=None,
            name="",
            duration=duration,
            mode=mode,
            multiplier=multiplier,
            save_target=save_target,
            save_animation=save_animation,
            display_animation=display_animation,
            num_sequences=1,
            as_mp4=as_mp4,
            include_timestep=include_timestep,
            return_path=True,
        )
            

def generate_dataset_name(name_dataset, seed=None):
    if name_dataset is None:
        name_dataset = "hbb_dataset"        
    name_list = [name_dataset, datetime.now().strftime("%y%m%d_%H%M%S")]
    if seed is not None:
        name_list.append(str(seed))
    return "_".join(name_list)
        
