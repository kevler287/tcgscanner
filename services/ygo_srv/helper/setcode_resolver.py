from helper.text_normalizer import *

SINGLE_CHAR_LANG_CODES = {
    "E": "English",
    "G": "German"
}

DOUBLE_CHAR_LANG_CODES = {
    "EN": "English",
    "DE": "German",
    "JP": "Japanese",
    "KR": "Korean",
}

LOOKUPS = {
    0: lambda code: "English",
    1: SINGLE_CHAR_LANG_CODES.get,
    2: DOUBLE_CHAR_LANG_CODES.get,
}

def resolve_expansion_code(expansion_code: str):
    return generate_ambiguous_permutations(text=expansion_code)

def resolve_collector_number(collector_number: str):
    cn = letter_to_number(collector_number)

    prefix_opts = AMBIGUOUS_LOOKUP.get(cn[0])
    if prefix_opts is None:
        return [cn]

    return [prefix + cn[1:] for prefix in prefix_opts]

def resolve_language_code(language_code: str):
    language_code = number_to_letter(language_code)
    language = LOOKUPS.get(len(language_code), lambda code: None)(language_code)
    return language_code, language