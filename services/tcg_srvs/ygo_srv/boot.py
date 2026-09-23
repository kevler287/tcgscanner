import csv
import os

DATA_DIR = os.environ.get("DATA_DIR", "/data")
CATALOG_CSV_PATH = os.path.join(DATA_DIR, "catalog.csv")
CATALOG_PROCESSED_CSV_PATH = os.path.join(DATA_DIR, "catalog_processed.csv")


def remove_25th_suffix(rows):
    for row in rows:
        row["expansionCode"] = row["expansionCode"].replace("-25TH", "")


def add_lang_codes_column(rows):
    for row in rows:
        row["lang_codes"] = []


def split_lang_code(rows):
    for row in rows:
        code = row["expansionCode"]
        if "-" in code:
            base, lang_code = code.split("-", 1)
            row["expansionCode"] = base
            row["lang_codes"].append(lang_code)


def load_catalog(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def save_catalog(path, rows, fieldnames):
    fieldnames = fieldnames + ["lang_codes"]
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

    remove_25th_suffix(rows)
    add_lang_codes_column(rows)
    split_lang_code(rows)

    save_catalog(CATALOG_PROCESSED_CSV_PATH, rows, fieldnames)
    print(f"[boot] Processed catalog written to {CATALOG_PROCESSED_CSV_PATH}, {len(rows)} rows")