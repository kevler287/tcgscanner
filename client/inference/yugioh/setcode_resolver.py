from client.inference.common.text_normalizer import *

SINGLE_CHAR_LANG_CODES = {
    "E": "English",
    "G": "German"
}

DOUBLE_CHAR_LANG_CODES = {
    "EN": "English",
    "DE": "German",
    "JP": "Japanese",
    "KR": "Korean"
}

# works for most of the set codes. Still there are few exceptions which will not be parsable with this module
def resolve_setcode(setcode: str):
    parts = setcode.split("-")
    if len(parts) != 2:
        return None

    expansion_code = parts[0]
    language_code = parts[:-3]
    collector_number = parts[1][-3:]

    ec_opts = resolve_expansion_code(expansion_code)
    language = resolve_language_code(language_code),
    cn_opts = resolve_collector_number(collector_number)

    return ec_opts, language, cn_opts

def resolve_expansion_code(expansion_code: str):
    return generate_ambiguous_permutations(text=expansion_code)

def resolve_language_code(language_code: str):
    if len(language_code) == 0:
        return "English"
    if len(language_code) == 1:
        return SINGLE_CHAR_LANG_CODES[language_code]
    if len(language_code) == 2:
        return DOUBLE_CHAR_LANG_CODES[language_code]
    return None

def resolve_collector_number(collector_number: str):
    cn = letter_to_number(collector_number)

    prefix_opts = AMBIGUOUS_LOOKUP.get(cn[0])
    if prefix_opts is None:
        return [cn]

    return [prefix + cn[1:] for prefix in prefix_opts]

    
    