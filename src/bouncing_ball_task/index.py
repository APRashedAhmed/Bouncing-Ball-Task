"""Index script that returns `Path` objects to *git tracked* locations the repo.

Uses the tracked files by git to determine what is worth referencing or not, and
then add them to the globals dict so they are importable. Starts with the list
of tracked files not in the importable module, then looks for all directories
`_dir_levels` deep from the main repo. In this setup, setting `_dir_levels` to
`0` would return just the top level directories.
"""
import subprocess as _subprocess
from pathlib import Path as _Path

# The main hard assumption made in this setup is this file is one directory
# below the repo
dir_repo = _Path(__file__).resolve().parent.parent.parent

# # Programatic Additions
# How many directories deeper than the top level do we want to be referencable
_dir_levels = 2
# Git command to get all files that arent in the main import directory
_cmd_git_files = "git ls-files -- . ':!:bouncing_ball_task/**'"
# Iterate through all files returned by the command above if run from DIR_REPO
for _path in _subprocess.check_output(
    _cmd_git_files, shell=True, cwd=str(dir_repo)
).splitlines():
    # Convert from binary to string
    _path = _path.decode()
    # Select for strings that have at least one "/" in them (are a directory)
    if "/" in _path:
        _parent_list = list(_Path(_path).parents)
        # Parents are listed in ascending order, so go for the elements
        # starting with the second to last index since the last index is the
        # '.' directory.
        for _idx in range(-2, -_dir_levels - 3, -1):
            try:
                _index_path = _parent_list[_idx]
                _index_name = "dir_" + str(_index_path).lower()
                _index_name = _index_name.replace("/", "_").replace("-", "_")
                globals()[_index_name] = dir_repo / _index_path
            except IndexError:
                pass
