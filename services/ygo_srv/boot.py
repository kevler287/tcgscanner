import csv
import json
import os

import pandas as pd

import rule_engine

DATA_DIR = os.environ.get("DATA_DIR", "/data")
CATALOG_CSV_PATH = os.path.join(DATA_DIR, "catalog.csv")
CATALOG_PROCESSED_CSV_PATH = os.path.join(DATA_DIR, "catalog_processed.csv")
SET_CONFIG_PATH = os.path.join(DATA_DIR, "setcode_collisions.json")


def remove_25th_suffix(rows):
    for row in rows:
        row["expansionCode"] = row["expansionCode"].replace("-25TH", "")

def strip_en_collector_number_prefix(rows):
    for row in rows:
        if row["collectorNumber"].startswith("EN"):
            row["collectorNumber"] = row["collectorNumber"][2:]

def remove_empty_collector_number(rows):
    return [row for row in rows if row["collectorNumber"].strip()]


WILDCARD_LANG_CODE = "*"


def add_lang_codes_column(rows):
    for row in rows:
        # Default: product has a lang_code, but unrestricted (Case 3)
        row["lang_codes"] = [WILDCARD_LANG_CODE]


def split_lang_code(rows):
    for row in rows:
        code = row["expansionCode"]
        if "-" in code:
            base, lang_code = code.split("-", 1)
            row["expansionCode"] = base
            # Overwrite the wildcard default with the specific code (Case 2)
            row["lang_codes"] = [lang_code]


def load_set_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    compiled = {}
    for expansion_code, rule_defs in raw.items():
        compiled_rules = []
        for rule_def in rule_defs:
            condition = rule_def["condition"]
            patch = {k: v for k, v in rule_def.items() if k != "condition"}
            try:
                rule = rule_engine.Rule(condition)
            except rule_engine.errors.RuleSyntaxError as e:
                raise ValueError(f"Invalid condition for set '{expansion_code}': {condition!r} -> {e}")
            compiled_rules.append((rule, patch))
        compiled[expansion_code] = compiled_rules

    return compiled


def apply_set_config(rows, set_config):
    unresolved = []
    for row in rows:
        rules = set_config.get(row["expansionCode"])
        if not rules:
            continue

        for rule, patch in rules:
            if rule.matches(row):
                row.update(patch)
                break
        else:
            unresolved.append((row["expansionCode"], row["collectorNumber"], row["name"]))

    if unresolved:
        unresolved.sort(key=lambda entry: (entry[0], entry[1]))
        print(f"[boot] WARNING: {len(unresolved)} row(s) matched a configured set but no rule:")
        for expansion_code, collector_number, name in unresolved:
            print(f"  {expansion_code}  {collector_number}  {name}")


def load_catalog(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def save_catalog(path, rows, fieldnames):
    fieldnames = fieldnames + ["lang_codes", "copyright", "new_layout"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["lang_codes"] = "|".join(row["lang_codes"])
            writer.writerow(row)


def run_boot_job():
    print(f"[boot] Loading catalog from {CATALOG_CSV_PATH}")
    rows, fieldnames = load_catalog(CATALOG_CSV_PATH)

    set_config = load_set_config(SET_CONFIG_PATH)

    rows = remove_empty_collector_number(rows)
    remove_25th_suffix(rows)
    strip_en_collector_number_prefix(rows)
    add_lang_codes_column(rows)
    apply_set_config(rows, set_config)
    split_lang_code(rows)

    save_catalog(CATALOG_PROCESSED_CSV_PATH, rows, fieldnames)
    print(f"[boot] Processed catalog written to {CATALOG_PROCESSED_CSV_PATH}, {len(rows)} rows")

    catalog_df = pd.read_csv(CATALOG_PROCESSED_CSV_PATH, dtype=str)
    catalog_df["lang_codes"] = catalog_df["lang_codes"].apply(lambda s: s.split("|") if s else [])
    return catalog_df