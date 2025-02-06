import os

def is_mini():
    return int(os.environ.get('FULLRUN', '0')) == 0

def is_quick_vecs():
    return int(os.environ.get('QUICKVECS', '0')) == 1
