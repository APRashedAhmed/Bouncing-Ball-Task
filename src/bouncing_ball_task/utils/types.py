"""Custom types used in the repo.

Note: if you want to type check in python against the types below, use the
function ``get_args`` to return an iterable of types.

Example:
```
from typing import get_args
from timescales.utils.types import Config

assert isinstance(dict({}), get_args(Config)) # Passes
```

See this SO post for more examples/info:
https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic
"""
import os
from argparse import Namespace
from collections.abc import Iterable, Iterator, Sequence
from typing import Dict, NamedTuple
from typing import SupportsFloat as Numeric
from typing import Tuple, Union

import torch

PathLike = Union[str, os.PathLike]

# Task types
Position = Union[tuple[Numeric, Numeric], torch.tensor]  # 2D position
Color = Union[tuple[Numeric, Numeric, Numeric], torch.tensor]  # RGB
