from itertools import product
import re


AMBIGUOUS_CHARS = {
    'O': '0',
    'I': '1',
    'T': '1',
    'L': '1',
    'A': '4',
    'S': '5',
    'G': '6',
    'Z': '7',
    'B': '8',
}

# build bidirectional lookup: char -> set of possible substitutes (including itself)
AMBIGUOUS_LOOKUP = {}
for a, b in AMBIGUOUS_CHARS.items():
    AMBIGUOUS_LOOKUP.setdefault(a, {a}).add(b)
    AMBIGUOUS_LOOKUP.setdefault(b, {b}).add(a)

def letter_to_number(text: str) -> str:
    '''Unifies different OCR detections to a common spelling for faster convergence e.g. DE053 & DEOS3'''
    for l, n in AMBIGUOUS_CHARS.items():
        text = text.replace(l, n)
    return text

def number_to_letter(text: str) -> str:
    '''Unifies different OCR detections to a common spelling for faster convergence e.g. DE053 & DEOS3'''
    for l, n in AMBIGUOUS_CHARS.items():
        text = text.replace(n, l)
    return text

def insert_dash_before_lang(s: str) -> str:
    return re.sub(r"(DE|EN)(\d+)$", r"-\1\2", s)

def generate_ambiguous_permutations(text: str) -> list[str]:
    char_options = [
        AMBIGUOUS_LOOKUP.get(char, {char})
        for char in text
    ]
    return ["".join(combo) for combo in product(*char_options)]