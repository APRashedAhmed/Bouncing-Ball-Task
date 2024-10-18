"""Functions for "basic python", not related to specific packages or goals."""
import os
import ast
import inspect
import shutil
import tempfile
import random
from collections.abc import Iterable
from functools import partial, wraps
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
from loguru import logger

def set_global_seed(seed_value=None):
    """
    Set a global random seed for reproducibility across various libraries.

    This function sets the same random seed for Python's `random` module, 
    NumPy (if installed), PyTorch (if installed), and TensorFlow (if installed).
    It also sets the `PYTHONHASHSEED` environment variable to ensure
    deterministic hashing in Python.

    Parameters
    ----------
    seed_value : Optional[int]
        The seed value to use for initializing the random number generators
    """
    # Generate a seed if none is passed
    if seed_value is None:
        seed_value = random.randint(0, 2**32 - 1)
    
    # Set seed for Python's random module
    random.seed(seed_value)
    
    # Set seed for numpy if available
    try:
        import numpy as np
        np.random.seed(seed_value)
    except ImportError:
        pass
    
    # Set seed for PyTorch if available
    try:
        import torch
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
    except ImportError:
        pass
    
    # Set seed for TensorFlow if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed_value)
    except ImportError:
        pass
    
    # Set seed for OS-level randomness
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    return seed_value


def get_unique_filename(name: str, directory: Path) -> str:
    """Generates a unique filename if a file with the given name already exists
    in the specified directory.

    Parameters
    ----------
    name : str
        The original filename (including the extension).
    directory : Path
        The directory where to check for the file's existence.

    Returns
    -------
    str
        A unique filename.

    Example
    -------
    >>> directory = Path("some_directory")
    >>> name = "example.txt"
    >>> unique_name = get_unique_filename(name, directory)
    """
    directory = Path(directory)

    # Extract the base name and the extension
    name_without_ext = name.split(".")[0]
    extension = name.split(".")[-1]

    counter = 1
    original_name = name_without_ext
    full_path = directory / f"{name_without_ext}.{extension}"

    # Check if a file with that name exists, and if so, increment the counter
    # until a unique filename is found.
    while full_path.exists():
        name_without_ext = f"{original_name}_{counter}"
        full_path = directory / f"{name_without_ext}.{extension}"
        counter += 1

    return f"{name_without_ext}.{extension}"


def retry(
    func=None,
    exception=Exception,
    n_tries=5,
    delay=1,
    backoff=1,
    loglevel=logger.debug,
):
    """Retry decorator with exponential backoff. See the following page for more
    details:

    https://stackoverflow.com/questions/42521549/retry-function-in-python

    Parameters
    ----------
    func : typing.Callable, optional
        Callable on which the decorator is applied, by default None

    exception : Exception or tuple of Exceptions, optional
        Exception(s) that invoke retry, by default Exception

    n_tries : int, optional
        Number of tries before giving up, by default 5

    delay : int, optional
        Initial delay between retries in seconds, by default 5

    backoff : int, optional
        Backoff multiplier e.g. value of 2 will double the delay, by default 1

    loglevel : bool, optional
        The level at which to log retries

    Returns
    -------
    typing.Callable
        Decorated callable that calls itself when exception(s) occur.

    Examples
    --------
    >>> import random
    >>> @retry(exception=Exception, n_tries=4)
    ... def test_random(text):
    ...    x = random.random()
    ...    if x < 0.5:
    ...        raise Exception("Fail")
    ...    else:
    ...        print("Success: ", text)
    >>> test_random("It works!")
    """
    if func is None:
        return partial(
            retry,
            exception=exception,
            n_tries=n_tries,
            delay=delay,
            backoff=backoff,
            loglevel=loglevel,
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        ntries, ndelay = n_tries, delay
        while ntries > 1:
            try:
                return func(*args, **kwargs)
            except exception as e:
                if ndelay:
                    loglevel(f"{str(e)}, Retrying in {ndelay} seconds...")
                    time.sleep(ndelay)
                    ndelay *= backoff
                ntries -= 1
        return func(*args, **kwargs)

    return wrapper


def isiterable(obj):
    """Function that determines if an object is an iterable, not including str.

    Parameters
    ----------
    obj : object
        Object to test if it is an iterable.

    Returns
    -------
    bool : bool
        True if the obj is an iterable, False if not.
    """
    if isinstance(obj, str):
        return False
    else:
        return isinstance(obj, Iterable)


def flatten(inp_iter):
    """Recursively iterate through values in nested iterables, and return a
    flattened list of the inputted iterable.

    Parameters
    ----------
    inp_iter : iterable
        The iterable to flatten.

    Returns
    -------
    value : object
        The contents of the iterable as a flat list.
    """

    def inner(inp):
        for val in inp:
            if isiterable(val):
                yield from inner(val)
            else:
                yield val

    return list(inner(inp_iter))


def get_imported(
    filepath: Union[str, Path] = None
) -> tuple[list[str], list[str]]:
    """Get a list of imported modules and functions/classes from a Python file.

    Parameters
    ----------
    filepath : Union[str, Path], optional
        The path to the Python file to analyze. If not provided, it will default
        to the path of the file that calls this function.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists:
        - The first list contains imported modules and packages.
        - The second list contains imported functions and classes.
    """
    if filepath is None:
        # Get the path of the script that is calling this function
        frame = inspect.stack()[1]
        filepath = frame.filename

    # Ensure filepath is a Path object
    filepath = Path(filepath)

    modules = []
    funcs_classes_names = []
    funcs_classes_modules = []

    with open(filepath) as f:
        tree = ast.parse(f.read(), filename=str(filepath))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                name = n.name
                if n.asname:
                    name += f" as {n.asname}"
                modules.append(name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for n in node.names:
                name = n.name
                if n.asname:
                    name = n.asname
                funcs_classes_names.append(name)
                funcs_classes_modules.append(module)

    max_name_len = max([len(name) for name in funcs_classes_names])
    funcs_classes = [
        f"{name:<{max_name_len + 2}} - {module}"
        for name, module in zip(funcs_classes_names, funcs_classes_modules)
    ]

    return (modules, funcs_classes)


def read_lines(file_path: Union[str, Path]) -> list[str]:
    """Reads lines from a file and stores them in a list.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the file to read from, either as a string or a Path object.

    Returns
    -------
    List[str]
        A list containing the lines read from the file.

    Examples
    --------
    >>> lines = read_lines_from_file("example.txt")
    >>> lines = read_lines_from_file(Path("example.txt"))
    """
    lines = []
    with open(file_path) as f:
        for line in f:
            lines.append(line.strip())  # Stripping newline characters
    return lines


def write_lines(lines: list[str], file_path: Union[str, Path]) -> None:
    """Writes each string from a list of strings to a new line in a file.

    Parameters
    ----------
    lines : List[str]
        The list of strings to write to the file.

    file_path : Union[str, Path]
        The path to the file to write to, either as a string or a Path object.

    Examples
    --------
    >>> write_lines_to_file(["Hello", "World"], "example.txt")
    >>> write_lines_to_file(["Hello", "World"], Path("example.txt"))
    """
    with open(file_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def alternating_ab_sequence(A, B, length, shuffle=True):
    """Generate a sequence of length length by optionally shuffling and
    alternating elements from two numpy linspaces A and B.

    Parameters
    ----------
    A : numpy.ndarray
        First linspace array.

    B : numpy.ndarray
        Second linspace array.

    length : int
        Desired length of the resulting sequence.

    shuffle : bool, optional
        Indicates whether to shuffle the linspace before extending the sequence.

    Returns
    -------
    numpy.ndarray
        The generated sequence of length `length`.

    Raises
    ------
    ValueError
        If A and B have different lengths.
    """
    if len(A) != len(B):
        raise ValueError("Input arrays A and B must have the same length.")

    N = len(A)
    sequence = np.empty(length)  # Preallocate the sequence array

    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    # Alternate elements directly into the preallocated sequence
    seq_idx = 0
    for idx in range(min(N, length // 2)):
        sequence[seq_idx] = A[indices[idx]]
        seq_idx += 1
        if seq_idx < length:
            sequence[seq_idx] = B[indices[idx]]
            seq_idx += 1
        else:
            break

    # If length > 2*N, repeat the process
    while seq_idx < length:
        if shuffle:  # Re-shuffle for each repetition
            np.random.shuffle(indices)
        for idx in range(N):
            if seq_idx >= length:
                break
            sequence[seq_idx] = (
                A[indices[idx]] if seq_idx % 2 == 0 else B[indices[idx]]
            )
            seq_idx += 1
            if seq_idx < length:
                sequence[seq_idx] = B[indices[idx]]
                seq_idx += 1
            else:
                break

    return sequence


def repeat_sequence(array, length, shuffle=True, roll=False, axis=0, shift=-1):
    """Create a new sequence by continuously reshuffling the input array along
    the specified axis for every repeat until the sequence reaches length
    `length`. This function supports N-dimensional arrays.

    Parameters
    ----------
    array : numpy.ndarray
        Input N-dimensional array to repeat.

    length : int
        Desired length of the resulting sequence along the specified axis.

    shuffle : bool, optional
        Indicates whether to shuffle the array along the specified axis before
        extending the sequence.

    roll : bool, optional
        Indicates whether to roll the array along the specified axis instead of
        shuffling before extending the sequence. Default is False.

    axis : int, optional
        The axis along which to repeat and optionally shuffle or roll the array.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        The generated sequence of length `length` along the specified axis, with
        each repeat being reshuffled or rolled.

    Raises
    ------
    ValueError
        If both shuffle and roll are True.
    """
    if shuffle and roll:
        raise ValueError("shuffle and roll parameters are mutually exclusive")

    # Function to shuffle array along a specific axis
    def shuffle_along_axis(arr, axis=0):
        if axis == 0:
            np.random.shuffle(arr)
            return arr
        else:
            # Shuffle along the specified axis
            shuff_indices = np.random.permutation(arr.shape[axis])
            return np.take(arr, shuff_indices, axis=axis)

    # Function to roll array along a specific axis
    def roll_along_axis(arr, axis=0):
        return np.roll(arr, shift=shift, axis=axis)

    # Determine the number of full repeats and the remaining part to reach the desired length
    num_full_repeats = length // array.shape[axis]
    remainder = length % array.shape[axis]

    reshuffled_or_rolled_arrays = []

    temp_array = array.copy()
    for i in range(num_full_repeats):

        if shuffle:
            temp_array = shuffle_along_axis(temp_array, axis=axis)
        elif roll:
            temp_array = (
                roll_along_axis(temp_array, axis=axis) if i > 0 else temp_array
            )  # Roll for repeats after the first
        reshuffled_or_rolled_arrays.append(temp_array)

        # Shuffle the array for each repeat if shuffle is True
        # temp_array = array.copy() if temp_array is None else temp_array.copy()
        temp_array = temp_array.copy()

    # Handle the remainder
    if remainder > 0:
        temp_array = (
            temp_array.copy()
        )  # Ensure a fresh copy for the last partial segment
        # temp_array = array.copy()

        if shuffle:
            temp_array = shuffle_along_axis(temp_array, axis=axis)
        elif roll:
            # Roll if needed (and if not the first segment)
            temp_array = (
                roll_along_axis(temp_array, axis=axis)
                if num_full_repeats > 0
                else temp_array
            )

        # For other axes, use slicing to get the remainder portion
        slicer = [slice(None)] * temp_array.ndim
        slicer[axis] = slice(remainder)
        reshuffled_or_rolled_arrays.append(temp_array[tuple(slicer)])

    # Concatenate reshuffled segments along the specified axis
    return np.concatenate(reshuffled_or_rolled_arrays, axis=axis)


def import_and_run(filename, global_namespace=None):
    """Imports a Python script and runs all its code, including the part under
    `if __name__ == "__main__":` in the specified namespace or in a new
    dictionary.

    Parameters
    ----------
    filename : (str, Path)
        Path to the Python script file to be executed.

    global_namespace : dict, optional
        A dictionary to use for the namespace during execution. If None, a new
        dictionary is created.

    Returns
    -------
    dict
        The namespace after script execution, containing all variables,
        functions, etc.
    """
    if global_namespace is None:
        global_namespace = {}

    with open(filename) as file:
        code = file.read()

    # Set __name__ to '__main__' to simulate running as the main script
    global_namespace["__name__"] = "__main__"
    exec(code, global_namespace)
    return global_namespace


def find_relative_path(base_path: Path, target_path: Path) -> Path:
    # Resolve both paths (normalize and remove any symbolic links)
    base_path = base_path.resolve()
    target_path = target_path.resolve()

    # Get the parts of both paths
    base_parts = base_path.parts
    target_parts = target_path.parts

    # Find the common path length
    common_length = len(set(base_parts) & set(target_parts))

    # Number of steps needed to go back to the common ancestor
    steps_up = len(base_parts) - common_length

    # Steps to go down from the common ancestor to the target path
    steps_down = target_parts[common_length:]

    # Constructing the relative path
    relative_path = Path(*([".."] * steps_up + list(steps_down)))

    return relative_path

def repeat_sequence_imbalanced(
        array,
        balance,
        length,
        # shuffle=True,
        roll=True,
        # axis=0,
        shift=-1
):
    output = []
    unique_values = np.unique(balance)
    num_values = len(unique_values)
    value_counter = np.zeros(num_values)            
    splits = {
        value: array[balance == value]
        for value in unique_values
    }

    for i in range(length):
        if roll:
            if i != 0 and not (i % num_values):
                unique_values = np.roll(unique_values, shift=shift)

        value = int(unique_values[i % num_values])
        output.append(
            splits[value][int(value_counter[value] % len(splits[value]))]
        )
        value_counter[value] += 1
    return output

