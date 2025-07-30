import os


def runcmd(cmd, workdir, out=None):
    """
    Execute cmd in workdir
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
    """
    return os.path.commonpath([basedir, sdir]) == basedir


def find_fnames_with_token(basedir, token):
    """
    Find files in basedir that contains specified token.
    Return list.
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
    """
    matching_dirs = []
    for root, dirs, _ in os.walk(basedir):
        for mydir in dirs:
            if token in mydir:
                dirpath = os.path.join(root, mydir)
                matching_dirs.append(dirpath)
    return matching_dirs
