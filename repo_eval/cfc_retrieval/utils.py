import re
import os
from typing import List
from functools import lru_cache
from nltk.tokenize import word_tokenize

REGEX_TEXT = ("(?<=[a-z0-9])(?=[A-Z])|"
              "(?<=[A-Z0-9])(?=[A-Z][a-z])|"
              "(?<=[0-9])(?=[a-zA-Z])|"
              "(?<=[A-Za-z])(?=[0-9])|"
              "(?<=[@$.'\"])(?=[a-zA-Z0-9])|"
              "(?<=[a-zA-Z0-9])(?=[@$.'\"])|"
              "_|\\s+")
SPLIT_REGEX = re.compile(REGEX_TEXT)


def tokenize_nltk(text):
    words = word_tokenize(text)
    output_list = []
    for w in words:
        w_list = re.findall(r'\w+', w)
        output_list.extend(w_list)
    return output_list


@lru_cache(maxsize=5000)
def split_identifier_into_parts(identifier: str) -> List[str]:
    """
    Split a single identifier into parts on snake_case and camelCase
    """
    identifier_parts = list(s.lower() for s in SPLIT_REGEX.split(identifier) if len(s) > 0)

    if len(identifier_parts) == 0:
        return [identifier]
    return identifier_parts


def file_distance(src_file, dest_file):
    distance = -1
    try:
        commonpath = os.path.commonpath([src_file, dest_file])
        rel_file1_path = os.path.relpath(src_file, commonpath)
        rel_file2_path = os.path.relpath(dest_file, commonpath)
        distance = rel_file1_path.count(os.sep) + rel_file2_path.count(os.sep)
    except Exception as e:
        # print(e, src_file, dest_file)
        pass

    return distance
