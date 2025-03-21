"""Functions for "basic python", not related to specific packages or goals."""
import os
import re
import ast
import inspect
import shutil
import tempfile
import random
import dataclasses
from collections.abc import Iterable
from functools import partial, wraps
from pathlib import Path
from typing import Optional, List, Tuple, Union, get_args

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

def create_sequence_splits(seq, total=None):
    # Replace any None values with -1 (wildcard marker)
    seq = [x if x is not None else -1 for x in seq]

    # Prepare to calculate fixed sum and track wildcards
    fixed_sum = 0
    wildcard_indices = []
    wildcard_weights = []
    
    # Build the final list using this placeholder list.
    result = [None] * len(seq)
    
    for idx, value in enumerate(seq):
        # Treat any non-negative value as fixed
        if value >= 0:
            fixed_sum += value
            result[idx] = value
        else:
            # Record index and weight (absolute value of negative wildcard)
            wildcard_indices.append(idx)
            wildcard_weights.append(abs(value))
    
    if wildcard_indices:
        # If total is None or fixed_sum < 1.0, use total as 1.0
        total = total if total is not None and fixed_sum > 1.0 else 1.0
        
        # Calculate the remainder to be distributed among wildcards
        remainder = total - fixed_sum
            
        # Ensure that the fixed values do not exceed or equal the total.
        if remainder < 0.0:
            raise ValueError(
                f"Using wildcard values but total of non-wildcard values {fixed_sum}"
                f"exceeds total {total}."
            )

        # Prevent a divide by zero
        total_weight = sum(wildcard_weights)
        if wildcard_indices and total_weight == 0:
            raise ValueError("Wildcard weights sum to zero, cannot distribute remainder.")

        # Allocate the remainder to each wildcard based on its weight
        for i, idx in enumerate(wildcard_indices):
            result[idx] = remainder * wildcard_weights[i] / sum(wildcard_weights)
        
    return [val / sum(result) for val in result]


def get_dataclasses(namespace: dict = None):
    if namespace is None:
        namespace = globals()
    
    dcs = {}
    for obj in list(namespace.values()):
        # Check if obj is a type (i.e. a class) and is a dataclass.
        if isinstance(obj, type) and dataclasses.is_dataclass(obj):
            if not obj.__name__.startswith("_"):
                try:
                    # Try to instantiate without arguments (this works if all fields have defaults).
                    obj.asdict = dataclasses.asdict(obj())
                    obj.keys = obj.asdict.keys()
                    obj.values = obj.asdict.values()
                    obj.items = obj.asdict.items()
                    
                    name = s2 = re.sub(r'([a-z])([A-Z])', r'\1_\2', obj.__name__).lower()
                    dcs[name] = obj
                except TypeError:
                    # If the dataclass requires parameters, skip it.
                    continue
    namespace.update(dcs)
    return dcs


def register_defaults(module_globals):
    default_dcs = get_dataclasses(module_globals)
    module_globals["_default_dcs"] = default_dcs
    
    def __getattr__(name: str):
        for _, instance in default_dcs.items():
            if hasattr(instance, name):
                return getattr(instance, name)
        raise AttributeError(f"module {module_globals.get('__name__', 'unknown')} has no attribute {name}")
    
    def __dir__():
        base = set(module_globals.keys())
        extra = set()
        for _, instance in default_dcs.items():
            # If the instance is a dataclass, use its __dataclass_fields__ keys.
            if hasattr(instance, "__dataclass_fields__"):
                extra.update(instance.__dataclass_fields__.keys())
        base_new = sorted(base.union(extra))
        
        return base_new

    # import ipdb; ipdb.set_trace()    
    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__


def parse_bool(x):
    """Convert common boolean string representations to a boolean."""
    if isinstance(x, bool):
        return x
    x_lower = str(x).lower()
    if x_lower in ("true", "1", "yes"):
        return True
    elif x_lower in ("false", "0", "no"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Expected boolean value, got {x}")

def wrap_type_with_none_check(type_func):
    """
    Wrap a conversion function so that if the input string equals "None"
    (case-insensitive), it returns Python's None.
    """
    def wrapped(x):
        if isinstance(x, str) and x.lower() == "none":
            return None
        return type_func(x)
    return wrapped


def get_underlying_type(field_obj, default_value):
    if field_obj.type is not None:
        # Check if field_type is a Union that includes NoneType.
        if hasattr(field_obj.type, "__origin__"):
            if field_obj.type.__origin__ is Union:
                args = get_args(field_obj.type)
                non_none = [arg for arg in args if arg is not type(None)]
                if len(non_none) == 1:
                    return non_none[0]
                
            else:
                return field_obj.type.__origin__
        else:
            return field_obj.type
            
    elif default_value is not None:
        return type(default_value)
    else:
        # Fallback to str if no specific type can be determined.
        return str

def add_dataclass_args(parser, instance):
    if not dataclasses.is_dataclass(instance):
        raise ValueError("The provided instance is not a dataclass.")
    
    for field_name, field_obj in instance.__dataclass_fields__.items():
        option_str = f"--{field_name}"
        # Check if this option has already been added.
        if option_str in parser._option_string_actions:
            # Option already exists; skip (or you could log a warning).
            continue        

        # Determine default value. If no default is present, skip the field.
        if field_obj.default is not dataclasses.MISSING:
            default_value = field_obj.default
        elif field_obj.default_factory is not dataclasses.MISSING:
            default_value = field_obj.default_factory()
        else:
            continue
        

        # # Determine the type conversion function.
        # # Prefer the type hint if provided; otherwise, use type(default_value)
        # if field_obj.type is not None:
        #     try:
        #         base_type = field_obj.type.__origin__
        #     except AttributeError:
        #         # In case of non-typing types (such as <class 'int'>, for instance)
        #         base_type = field_obj.type
        # else:
        #     base_type = type(default_value)

       # Use the field annotation to get the conversion type.
        base_type = get_underlying_type(field_obj, default_value)
                        
        # if field_name == "sequence_mode":
        #     import ipdb; ipdb.set_trace()
            
        if base_type == bool:
            conv_type = wrap_type_with_none_check(parse_bool)
        elif base_type in (tuple, list):
            def parse_iterable(s):
                try:
                    v = ast.literal_eval(s)
                    if isinstance(v, (tuple, list)):
                        return v
                    else:
                        raise argparse.ArgumentTypeError(f"Expected tuple or list, got {v}")
                except Exception as e:
                    raise argparse.ArgumentTypeError(f"Could not parse {s} as tuple/list: {e}")
            conv_type = wrap_type_with_none_check(parse_iterable)
        # elif default_value is None:
        #     conv_type = wrap_type_with_none_check(str)
        else:
            conv_type = wrap_type_with_none_check(base_type)
        
        # Add the argument with a -- prefix.
        parser.add_argument(option_str,
                            type=conv_type,
                            default=default_value,
                            help=f"Default: {default_value}")
    return parser


# def parse_bool(x):
#     if isinstance(x, bool):  # already a bool, no conversion needed
#         return x
#     x_lower = str(x).lower()
#     if x_lower in ('true', '1', 'yes'):
#         return True
#     elif x_lower in ('false', '0', 'no'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError(f"Expected a boolean value, got {x}")

# def wrap_none_check(type_func):
#     def wrapped(x):
#         if isinstance(x, str) and x.lower() in {"none", "null", "nan"}:
#             return None
#         return type_func(x)
#     return wrapped

# def add_args_from_dict(parser, param_dict):
#     for key, value in param_dict.items():
#         option_str = f"--{key}"
#         # Check if this option has already been added.
#         if option_str in parser._option_string_actions:
#             # Option already exists; skip (or you could log a warning).
#             continue

#         # Determine the argument type.
#         # Choose the conversion function.
#         if isinstance(value, bool):
#             arg_type = wrap_none_check(parse_bool)
#         elif isinstance(value, (tuple, list)):
#             # Define a custom parser for iterables.
#             def parse_iterable(s):
#                 try:
#                     v = ast.literal_eval(s)
#                     if isinstance(v, (tuple, list)):
#                         return v
#                     else:
#                         raise argparse.ArgumentTypeError(f"Expected a tuple or list, got {v}")
#                 except Exception as e:
#                     raise argparse.ArgumentTypeError(f"Could not parse {s} as tuple/list: {e}")
#             arg_type = wrap_none_check(parse_iterable)
#         elif value is None:
#             arg_type = wrap_none_check(str)
#         else:
#             # Use the type of the default (or str if the default is None).
#             base_type = type(value) if value is not None else str
#             arg_type = wrap_none_check(base_type)

#         parser.add_argument(option_str, type=arg_type, default=value)
        
#     return parser
