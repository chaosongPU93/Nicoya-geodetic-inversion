#!/usr/bin/env python3
"""Git clean filter: strip outputs/execution counts from Jupyter notebooks.

Configured via .gitattributes (`*.ipynb filter=nbstrip`) so that committed
notebook blobs contain code only, while the working-copy notebooks keep their
rendered outputs. Reads a notebook on stdin, writes the stripped version to
stdout.
"""
import sys
import json


def main():
    nb = json.load(sys.stdin)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    nb.get("metadata", {}).pop("widgets", None)
    json.dump(nb, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
