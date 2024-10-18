# FUNCTIONS for dealing with gifs
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
# import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from bouncing_ball_task import index
from bouncing_ball_task.utils.pyutils import get_unique_filename
from bouncing_ball_task.utils.types import Color, Position


def build_background(
    size_frame: tuple[int, int],
    mask_start: int,
    mask_end: int,
    mask_color: Color,
    dtype=np.uint8,
):
    background = np.zeros((*size_frame[::-1], 3), dtype=dtype)
    background[:, mask_start:mask_end, :] = mask_color
    return background


def draw_circle(
    position: Position,
    color: Color,
    background: np.ndarray,
    ball_radius: int,
    mask_color: Color,
    thickness: int = -1,
):
    return cv2.circle(
        np.copy(background),
        tuple(np.round(position).astype(int).tolist()),
        ball_radius,
        color=[255, 255, 255] if color == mask_color else list(color),
        thickness=thickness,
    )


def draw_frame(
    position: Position,
    color: Color,
    ball_radius: int,
    mask_color: Color,
    size_frame,
    mask_start,
    mask_end,
    circle_border_thickness: int = 2,
):
    frame = np.zeros((*size_frame[::-1], 3), dtype=np.uint8)
    # import ipdb; ipdb.set_trace()
    frame = cv2.circle(
        frame,
        tuple(np.round(position).astype(int).tolist()),
        ball_radius,
        color=color,
        thickness=-1,
    )
    frame[:, mask_start:mask_end, :] = mask_color
    x_position = position[0]

    if (
        mask_start - ball_radius - circle_border_thickness
        <= x_position
        <= mask_end + ball_radius + circle_border_thickness
    ):
        frame = cv2.circle(
            frame,
            tuple(np.round(position).astype(int).tolist()),
            ball_radius + int(np.round(circle_border_thickness / 2)),
            color=[0, 0, 0],
            thickness=circle_border_thickness,
        )
    return frame


def save_gif(
    sample_images: list,
    path_dir: Union[Path, str] = "temp/bouncing_ball/",
    name: str = "animation",
    index: Optional[int] = None,
    duration=0,
    loop=0,
    return_path: bool = False,
    include_timestep: bool = True,
    as_mp4=False,
):
    name_split = name.split(".")
    video_name = name_split[0]
    if index is not None:
        video_name = "_".join((video_name, f"n_{index}"))

    if len(name_split) > 1:
        video_name = f"{video_name}.{name_split[-1]}"
    elif as_mp4:
        video_name = f"{video_name}.mp4"
    else:
        video_name = f"{video_name}.gif"

    path_video = Path(path_dir) / video_name

    if not path_video.parent.exists():
        Path(path_video).parent.mkdir(parents=True)

    if not isinstance(sample_images[0], Image.Image):
        sample_images = [Image.fromarray(image) for image in sample_images]

    if include_timestep:
        longest_text = f"t = {len(sample_images)}"
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf", size=25
        )
        draw = ImageDraw.Draw(sample_images[0])
        text_width, text_height = draw.textsize(longest_text, font)
        image_width, image_height = sample_images[0].size
        [
            ImageDraw.Draw(img).text(
                (image_width - text_width, image_height - text_height),
                f"t = {t}",
                color="white",
                font=font,
            )
            for t, img in enumerate(sample_images)
        ]

    if as_mp4:
        # Convert duration from milliseconds to frames per second
        fps = (
            1000 / duration if duration != 0 else 30
        )  # Default to 30 fps if duration is 0
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        video = cv2.VideoWriter(
            str(path_video), fourcc, fps, sample_images[0].size
        )

        for image in sample_images:
            open_cv_image = np.array(image)
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            video.write(open_cv_image)

        video.release()
    else:
        sample_images[0].save(
            str(path_video),
            save_all=True,
            append_images=sample_images[1:],
            duration=duration,
            loop=loop,
        )

    if return_path:
        return path_video


# def save_gif(
#     sample_images: list[Image, ...],
#     path_dir: Union[Path, str] = index.dir_data / "temp/bouncing_ball/",
#     name: str = "animation",
#     index: Optional[int] = None,
#     duration=0,
#     loop=0,
#     return_path: bool = False,
#     include_timestep: bool = True,
#     as_mp4=False
# ):

#     if name.endswith(".gif"):
#         name = name[:-4]
#     if index is not None:
#         name = "_".join((name, f"n_{index}"))
#     name += ".gif"

#     # Make sure it's unique
#     name = get_unique_filename(name, path_dir)
#     path_gif = Path(path_dir) / name

#     if not path_gif.parent.exists():
#         Path(path_gif).parent.mkdir(parents=True)

#     if not isinstance(sample_images[0], Image.Image):
#         sample_images = [Image.fromarray(image) for image in sample_images]

#     if include_timestep:
#         longest_text = f"t = {len(sample_images)}"
#         font = ImageFont.truetype(
#             "/usr/share/fonts/truetype/freefont/FreeMono.ttf", size=25
#         )
#         draw = ImageDraw.Draw(sample_images[0])
#         text_width, text_height = draw.textsize(longest_text, font)
#         image_width, image_height = sample_images[0].size
#         [
#             ImageDraw.Draw(img).text(
#                 (image_width - text_width, image_height - text_height),
#                 f"t = {t}",
#                 color="white",
#                 font=font,
#             )
#             for t, img in enumerate(sample_images)
#         ]

#     sample_images[0].save(
#         str(path_gif),
#         save_all=True,
#         append_images=sample_images[1:],
#         duration=duration,
#         loop=loop,
#     )

#     logger.debug(f"Saving image sequence to {str(path_gif)}")

#     if return_path:
#         return path_gif


def build_array_list(
    sequence: list[Union[Position, np.ndarray], ...],
    mode: Optional[str] = None,
    background: Optional[np.ndarray] = None,
    ball_radius: Optional[int] = None,
    mask_color: Optional[Color] = None,
):
    # Infer what sequence mode generated the input
    if mode is None:
        element = sequence[0]
        if isinstance(element, tuple):
            mode = "parameter"
        else:
            # Can be either array or parameter array depending on the shape
            # elif isinstance(element, (np.ndarray, torch.tensor)):
            if len(element.shape) == 1:  # Vector of position and color
                mode = "parameter_array"
            else:
                mode = "array"

    # A mode was passed and doesnt need to be inferred
    else:
        mode = mode.lower()

    # If it's already arrays, return
    if mode == "array":
        return sequence
    # Otherwise ensure we have these inputs for the remaining modes
    elif any([var is None for var in (background, ball_radius, mask_color)]):
        raise ValueError(
            "Cannot use parameter-mode if one of the following is None: "
            "'background', 'ball_radius', 'mask_color'."
        )

    # Convert to arrays
    if mode == "parameter":
        return [
            draw_circle(position, color, background, ball_radius, mask_color)
            for position, color in sequence
        ]
    elif mode == "parameter_array":
        return [
            draw_circle(
                parameters[:2].tolist(),
                parameters[2:].tolist(),
                background,
                ball_radius,
                mask_color,
            )
            for parameters in sequence
        ]
    else:
        raise ValueError(f"Invalid mode inputted: {mode}.")


def combined_array_sequence(
    samples, targets, preds, task_parameters, thickness=1, multiplier=4
):
    size_frame = [size * multiplier for size in task_parameters.size_frame]
    mask_start = task_parameters.mask_start * multiplier
    mask_end = task_parameters.mask_end * multiplier
    ball_radius = task_parameters.ball_radius * multiplier
    mask_color = task_parameters.mask_color
    thickness *= multiplier

    background = build_background(
        size_frame,
        mask_start,
        mask_end,
        mask_color,
    )

    arrays = []

    for sample, target, pred in zip(samples.cpu(), targets.cpu(), preds.cpu()):
        pred_position = pred[:2] * multiplier
        pred_color = pred[2:].tolist()

        pred_drawn = draw_circle(
            pred_position.tolist(),
            pred_color,
            background,
            ball_radius,
            mask_color,
            thickness=-1,
        )

        line_1_offset = np.array(
            [ball_radius / np.sqrt(2), ball_radius / np.sqrt(2)]
        )
        line_2_offset = np.array(
            [ball_radius / np.sqrt(2), -ball_radius / np.sqrt(2)]
        )

        pred_line_1_drawn = cv2.line(
            pred_drawn,
            np.round(pred_position + line_1_offset).numpy().astype(int),
            np.round(pred_position - line_1_offset).numpy().astype(int),
            [255, 255, 255],
            thickness,
        )

        pred_line_2_drawn = cv2.line(
            pred_line_1_drawn,
            np.round(pred_position + line_2_offset).numpy().astype(int),
            np.round(pred_position - line_2_offset).numpy().astype(int),
            [255, 255, 255],
            thickness,
        )

        sample_drawn = draw_circle(
            (sample[:2] * multiplier).tolist(),
            sample[2:].tolist(),
            pred_line_2_drawn,
            ball_radius,
            mask_color,
            thickness=thickness,
        )

        target_position = target[:2] * multiplier
        target_color = target[2:].tolist()

        target_drawn = draw_circle(
            target_position.tolist(),
            target_color,
            sample_drawn,
            ball_radius,
            mask_color,
            thickness=thickness,
        )

        circ1_drawn = cv2.line(
            target_drawn,
            np.round(target_position + line_1_offset).numpy().astype(int),
            np.round(target_position - line_1_offset).numpy().astype(int),
            target_color,
            1 * multiplier,
        )

        circ2_drawn = cv2.line(
            circ1_drawn,
            np.round(target_position + line_2_offset).numpy().astype(int),
            np.round(target_position - line_2_offset).numpy().astype(int),
            target_color,
            1 * multiplier,
        )
        arrays.append(circ2_drawn)

    return arrays


def build_image_list(
    arrays: Union[list[np.ndarray, ...], tuple[list[np.ndarray, ...], ...]],
    mode: str = "original",
):
    # Convert to a tuple of len 1 if we have just one sequence
    if not isinstance(arrays[0], (list, tuple)):
        arrays = (arrays,)

    # Concats all frames alongside each other
    # TODO Add a border between each frame
    if mode == "concat":
        return [
            Image.fromarray(np.concatenate(frames, 1))
            for frames in zip(*arrays)
        ]

    # Returns a tuple of image sequences or just one if only one was passed
    elif mode == "original":
        output = tuple(
            [
                [Image.fromarray(frame) for frame in zip(array)]
                for array in arrays
            ]
        )
        return output if len(output) > 1 else output[0]

    else:
        raise ValueError(f"Invalid mode inputted: {mode}.")


def display_animation(output, interval=50, blit=True, repeat_delay=500):
    fig = plt.figure()
    fig_outputs = [[plt.imshow(out, animated=True)] for out in output]
    sequence_animation = animation.ArtistAnimation(
        fig,
        fig_outputs,
        interval=interval,
        blit=blit,
        repeat_delay=repeat_delay,
    )
    return fig, sequence_animation
