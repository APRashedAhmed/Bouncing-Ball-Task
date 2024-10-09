"""Utility scripts related to the human bouncing ball task."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from bouncing_ball_task import index


def last_visible_color(samples, ball_radius, mask_start, mask_end, tol=1):
    """Determine the last visible RGB color of a ball before it fully enters a
    masked region in a series of trajectories represented by a 3D array.

    The function calculates the last visible timestep for each trajectory where
    the ball is not yet fully within the specified masked x-coordinate region,
    considering the ball's radius and a tolerance for near-edge visibility

    Parameters
    ----------
    samples : ndarray
        A 3D numpy array of shape (batch_size, timesteps, features), where each
        row represents a timestep in a trajectory, the first feature is the
        x-coordinate, and the last three features are expected to be RGB color
        values.

    ball_radius : float
        The radius of the ball. This is used to adjust the masked region to
        account for the entire ball potentially entering the masked region.

    mask_start : float
        The starting x-coordinate of the masked region.

    mask_end : float
        The ending x-coordinate of the masked region.

    tol : float, optional
        A tolerance value used to extend the masked region slightly inwards
        (default is 1). This allows for capturing the last visible color
        slightly before the ball is fully masked, depending on the specific
        application's precision requirements.

    Returns
    -------
    last_visible_color
        A 2D numpy array (ndarray) of shape (batch_size, features-2), where each
        row contains the RGB color values of the ball at the last visible
        timestep.
    """
    batch_size, timesteps, features = samples.shape

    # Create a boolean tensor indicating whether the timestep is within the
    # middle gray region or not
    x_coords = samples[:, :, 0]  # Extract the x-coordinates
    out_mask = ~(
        (x_coords >= (mask_start + ball_radius - tol))
        & (x_coords <= (mask_end - ball_radius + tol))
    )

    # Reverse the boolean tensor along the timesteps direction
    out_mask_reversed = np.flip(out_mask, axis=1)

    # Use argmax to find the first 1.0 in the reversed tensor, then calculating
    # timesteps - 1 - index gives the correct index in the original tensor
    first_out_mask_reversed = np.argmax(out_mask_reversed, axis=1)
    last_visible_index = timesteps - 1 - first_out_mask_reversed

    # Use this to index the original tensor for the last visible color
    last_visible_colors = samples[np.arange(batch_size), last_visible_index, 2:]

    return last_visible_colors  # , last_visible_index


def load_human_trial_raw_data(
    run_name,
    dir_human_task=index.dir_data / "hmdcpd",
):
    # Need the overall task metadata df and pkl
    # Verify and load
    path_human_trial_metadata = dir_human_task / f"{run_name}_trial_meta.csv"
    assert path_human_trial_metadata.exists()
    df_human_trial_metadata = pd.read_csv(
        str(path_human_trial_metadata), index_col=0
    )

    path_human_dataset_metadata = (
        dir_human_task / f"{run_name}_dataset_meta.pkl"
    )
    assert path_human_dataset_metadata.exists()
    with open(str(path_human_dataset_metadata), "rb") as f:
        dict_human_dataset_metadata = pickle.load(f)

    # Performs basic type conversions to make it easy to use
    df_human_trial_metadata["Path Video"] = df_human_trial_metadata[
        "Path Video"
    ].astype(str)

    # Use this to add bounce correction to the data
    bounce_final_pos_dict = {
        elem_index: x_pos_index
        for elem_index, x_pos_index in zip(
            dict_human_dataset_metadata["bounce"][
                "final_position_index_bounce"
            ],
            dict_human_dataset_metadata["bounce"]["final_x_index_bounce"],
        )
    }

    # Add idx_time to all bounce data
    bounce_mask = df_human_trial_metadata.trial == "bounce"
    df_human_trial_metadata.loc[bounce_mask, "idx_time"] = (
        df_human_trial_metadata.loc[bounce_mask, "idx_position"]
        .map(bounce_final_pos_dict)
        .values
    )

    # Change the column to ints
    df_human_trial_metadata["idx_time"] = (
        pd.to_numeric(df_human_trial_metadata["idx_time"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    # Add in the last color entered
    mask_start = dict_human_dataset_metadata["mask_start"]
    mask_end = dict_human_dataset_metadata["mask_end"]
    min_video_length = dict_human_dataset_metadata["video_length_min_f"]
    ball_radius = 10

    # Get the full ball path arrays
    human_params = np.array(
        [
            pd.read_csv(path).iloc[-min_video_length:, 1:].to_numpy()
            for path in df_human_trial_metadata["Path Parameters"]
        ]
    )

    # Find the last color that was visible
    color_entered = 1 + np.argmax(
        last_visible_color(
            human_params,
            ball_radius,
            mask_start,
            mask_end,
        ),
        axis=1,
    )

    # Add it to the df
    df_human_trial_metadata.loc[:, "color_entered"] = color_entered

    # Add a new column called final_color_response
    final_color_dict = {
        "red": 1,
        "green": 2,
        "blue": 3,
    }
    df_human_trial_metadata.loc[:, "final_color_response"] = (
        df_human_trial_metadata.loc[:, "Final Color"]
        .map(final_color_dict)
        .values
    )

    return df_human_trial_metadata, dict_human_dataset_metadata


def load_participant_raw_data(
    run_name,
    participant_id,
    dir_human_task=index.dir_data / "hmdcpd",
    dir_task_public=index.dir_repo.parent / "human-mdcpd-honeycomb/public",
    valid_colors={1: "red", 2: "green", 3: "blue"},
    validate=True,
    columns_to_keep=(
        "rt",
        "response",
        "trial_index",
        "internal_node_id",
        "correct_response",
    ),
    drop_na_responses=True,
):
    path_human_data = (
        dir_human_task / f"{run_name}_results_{participant_id}.csv"
    )
    assert (
        path_human_data.exists()
    ), f"Data '{str(path_human_data)}' file for {participant_id} does not exist"
    df_human_data = pd.read_csv(str(path_human_data))

    # Filter data based on which entries has a correct response value
    df_response = df_human_data.loc[~df_human_data.correct_response.isna()]

    # Subselect off those the ones that have a response value
    if drop_na_responses:
        df_response = df_response.loc[~df_human_data.response.isna()]

    # Do some type conversions
    columns_astype_int = ["rt", "response", "correct_response"]
    columns_astype_int = [
        col for col in columns_astype_int if col in columns_to_keep
    ]
    for column in columns_astype_int:
        df_response[column] = df_response[column].astype(int)

    # Get the paths to all the videos
    path_videos = (
        df_human_data[df_human_data.trial_type == "video-keyboard-response"]
        .stimulus.apply(lambda x: str(dir_task_public / x.split('"')[1]))
        .to_list()
    )

    # Stash participant metadata
    participant_metadata = {
        "task_version": df_response.task_version.values[0],
        "study_id": df_response.study_id.values[0],
        "participant_id": df_response.participant_id.values[0],
        "start_date": df_human_data.start_date.values[0],
    }

    # Only keep the desired columns
    df_response = df_response[list(columns_to_keep)]

    if validate:
        # All collected responses are in the valid responses
        assert all(
            val in valid_colors.keys() for val in df_response.response.unique()
        )

        # Compare the path colors to the correct_responses
        colors_from_video_paths = [
            Path(path).stem.split("_")[-1] for path in path_videos
        ]
        colors_from_correct_response = [
            valid_colors[val] for val in df_response.correct_response
        ]
        assert colors_from_video_paths == colors_from_correct_response

    # Add in the path to videos
    df_response.loc[:, "Path Video"] = path_videos

    return df_response, participant_metadata


def load_model_trial_raw_data(
    run_name,
    dir_human_task=index.dir_data / "hmdcpd",
    rename_iom_columns=("ibo",),
):

    path_all_data_model = dir_human_task / f"{run_name}_all_data.csv"
    assert path_all_data_model.exists()
    df_all_data_model = pd.read_csv(str(path_all_data_model), index_col=0)

    # And metadata for the specifics of the dataset used to generate the data
    path_model_dataset_metadata = (
        dir_human_task / f"{run_name}_dataset_meta.pkl"
    )
    assert path_model_dataset_metadata.exists()
    with open(str(path_model_dataset_metadata), "rb") as f:
        dict_model_dataset_metadata = pickle.load(f)

    # Use this to add bounce correction to the data
    bounce_final_pos_dict = {
        elem_index: x_pos_index
        for elem_index, x_pos_index in zip(
            dict_model_dataset_metadata["bounce"][
                "final_position_index_bounce"
            ],
            dict_model_dataset_metadata["bounce"]["final_x_index_bounce"],
        )
    }

    # Add idx_time to all bounce data
    bounce_mask = df_all_data_model.trial == "bounce"
    df_all_data_model.loc[bounce_mask, "idx_time"] = (
        df_all_data_model.loc[bounce_mask, "idx_position"]
        .map(bounce_final_pos_dict)
        .values
    )

    # Change the column to ints
    df_all_data_model["idx_time"] = (
        pd.to_numeric(df_all_data_model["idx_time"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    # Rename ideal observer model columns to match other model columns
    if rename_iom_columns is not None:
        # Function to rename columns
        def rename_columns(col_name):
            for iom_col in rename_iom_columns:
                parts = col_name.split(f"_{iom_col}")
                if len(parts) > 1:
                    # Reassemble with the new segment inserted
                    return f"{parts[0]}_model_iom_{iom_col}{parts[1]}"
            return col_name

        df_all_data_model.rename(columns=rename_columns, inplace=True)

    return df_all_data_model, dict_model_dataset_metadata


def fix_idx_time(df_all_data, meta_data):
    # Add bounce correction to the data
    bounce_final_pos_dict = {
        elem_index: x_pos_index
        for elem_index, x_pos_index in zip(
            meta_data["bounce"][
                "final_position_index_bounce"
            ],
            meta_data["bounce"]["final_x_index_bounce"],
        )
    }

    # Add idx_time to all bounce data
    bounce_mask = df_all_data.trial == "bounce"
    df_all_data.loc[bounce_mask, "idx_time"] = (
        df_all_data.loc[bounce_mask, "idx_position"]
        .map(bounce_final_pos_dict)
        .values
    )

    # Change it to ints
    df_all_data["idx_time"] = (
        pd.to_numeric(df_all_data["idx_time"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    return df_all_data

def group_model_names(df_all_data_model):
    response_columns = [
        name
        for name in df_all_data_model.columns
        if name.startswith("response")
    ]
    model_groups = {
        "_".join(name.split("_")[2:-1])
        if "_iom_" not in name
        else "_".join(name.split("_")[2:])
        for name in response_columns
    }
    group_dict = {
        group: [
            "_".join(name.split("_")[1:])
            for name in response_columns
            if group in name
        ]
        for group in model_groups
    }
    return group_dict


## Write function that loads the task data and all participant data, combines them,
## and then does some validation on them.

if __name__ == "__main__":

    run_name_human = "task_run_240409_1343"
    run_name_model = "mbb_dataset_240418_153646"

    # Participant choice
    all_participants = [
        "abdullah",
        "steven",
        "braden",
        "zeinab",
        "jimmy",
    ]

    group_models = [
        "ibo",
        "simple4",
        # "medium4",
    ]

    df_human_trial_metadata, human_dataset_metadata = load_human_trial_raw_data(
        run_name_human
    )

    # # Where the dataset videos live - needs to exist
    # dir_task_public = index.dir_repo.parent / "human-mdcpd-honeycomb/public"
    # assert dir_task_public.exists()

    # Some metadata for later
    mask_start = human_dataset_metadata["mask_start"]
    mask_end = human_dataset_metadata["mask_end"]
    min_video_length = human_dataset_metadata["video_length_min_f"]
    ball_radius = 10

    participant_data = {
        name: load_participant_raw_data(run_name_human, name)
        for name in all_participants
    }

    df_all_data_model, model_dataset_metadata = load_model_trial_raw_data(
        run_name_model
    )
