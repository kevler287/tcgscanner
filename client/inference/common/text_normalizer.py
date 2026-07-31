from itertools import product


AMBIGUOUS_CHARS = {
    'O': '0',
    'I': '1',
    'A': '4',
    'S': '5',
    'G': '6',
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

def generate_ambiguous_permutations(text: str) -> list[str]:
    char_options = [
        AMBIGUOUS_LOOKUP.get(char, {char})
        for char in text
    ]
    return ["".join(combo) for combo in product(*char_options)]