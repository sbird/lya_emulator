"""Module to store some utility functions."""
import os
import os.path
import subprocess

def get_git_hash(path):
    """Get the git hash of a file."""
    rpath = os.path.realpath(path)
    if not os.path.isdir(rpath):
        rpath = os.path.dirname(rpath)
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd = rpath, universal_newlines=True)
    return commit_hash
