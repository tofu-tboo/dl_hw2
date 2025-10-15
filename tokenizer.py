import re
import numpy as np

def tokenize(sentence):
    return np.array([re.compile(r"[A-Za-z0-9']+").findall(s.lower()) for s in sentence], dtype=object)