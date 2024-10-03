import os

def runcmd(cmd,workdir,out=None):
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
    return os.path.commonpath([basedir,sdir])==basedir
