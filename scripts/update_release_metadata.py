"""Replace repository and Zenodo release metadata across the submission package.

Run after publishing the v2.0.0 GitHub/Zenodo release, then recompile the PDFs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_DOI = "10.5281/zenodo.21813151"
DEFAULT_REPO = "https://github.com/wgarching/lhn-point-gap-topology"
TEXT_SUFFIXES = {".tex", ".txt", ".md", ".cff"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doi", required=True, help="New version DOI, e.g. 10.5281/zenodo.XXXX")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Final public GitHub URL")
    parser.add_argument("--old-doi", default=DEFAULT_OLD_DOI)
    args = parser.parse_args()

    changed = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text.replace(args.old_doi, args.doi).replace(DEFAULT_REPO, args.repo)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(path.relative_to(ROOT))

    print(f"Updated {len(changed)} files:")
    for path in changed:
        print(f"  {path}")
    print("Recompile paper/main.tex and paper/supplementary.tex. Also update the separate cover letter and portal text in the journal-submission package.")


if __name__ == "__main__":
    main()
