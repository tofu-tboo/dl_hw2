import re

def tokenize(sentence: str):
    return re.compile(r"[A-Za-z0-9']+").findall(sentence.lower())
