from pathlib import Path
import csv
from stylometry import extract_features

BASE_DIR = Path(__file__).resolve().parent / "political_corpus"
CLEAN_DIR = BASE_DIR / "texts" / "clean"
FEATURES_CSV = BASE_DIR / "features.csv"


def main():
    files = list(CLEAN_DIR.rglob("*.txt"))
    rows = []
    for clean_path in files:
        rel = clean_path.relative_to(CLEAN_DIR)
        feats = extract_features(clean_path)
        feats["file"] = str(rel)
        feats["author"] = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        rows.append(feats)
        print(f"✓ {rel}")

    fieldnames = list(rows[0].keys())
    with FEATURES_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nГотово: {FEATURES_CSV}")
    print(f"Текстов: {len(rows)}")


if __name__ == "__main__":
    main()