import os


def runcmd(cmd, workdir, out=None):
    """
    Execute cmd in workdir

    Parameters
    ----------
    cmd: str
        command to execute
    workdir: str
        directory in which to execute the command
    out: str or None
        if not None, redirect stdout and stderr to this file
    """
    base = os.getcwd()
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    os.system(cmd)
    os.chdir(base)
    return


def find_dirs_with_subdirs(basedir, token):
    """
    Find directories that contain subdires
    whose names contain the specified token
    Return list

    Parameters
    ----------
    basedir: str
        base directory to search
    token: str
        token to search for in subdirectory names
    """
    matching_dirs = []
    for root, dirs, files in os.walk(basedir):
        for subdir in dirs:
            if token in subdir:
                matching_dirs.append(root)
                break
    return matching_dirs


def is_subdir(basedir, sdir):
    """
    Check if sdir is a subdir of basedir

    Parameters
    ----------
    basedir: str
        base directory
    sdir: str
        directory to check

    Returns
    -------
    out: bool
        True if sdir is a subdir of basedir
    """
    return os.path.commonpath([basedir, sdir]) == basedir


def find_fnames_with_token(basedir, token):
    """
    Find files in basedir that contains specified token.
    Return list.

    Parameters
    ----------
    basedir: str
        base directory to search
    token: str
        token to search for in filenames

    Returns
    -------
    out: list
        list of matching file paths
    """
    matching_files = []
    for root, _, files in os.walk(basedir):
        for file in files:
            if token in file:
                filepath = os.path.join(root, file)
                matching_files.append(filepath)
    return matching_files


def find_dirs_with_token(basedir, token):
    """
    Find subdirs in basedir that contains specified token.
    Return list.

    Parameters
    ----------
    basedir: str
        base directory to search
    token: str
        token to search for in directory names

    Returns
    -------
    out: list
        list of matching directory paths
    """
    matching_dirs = []
    for root, dirs, _ in os.walk(basedir):
        for mydir in dirs:
            if token in mydir:
                dirpath = os.path.join(root, mydir)
                matching_dirs.append(dirpath)
    return matching_dirs
