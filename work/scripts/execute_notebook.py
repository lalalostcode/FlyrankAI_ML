"""Execute a notebook's Python code cells sequentially using only the stdlib.

This lightweight fallback is for environments that have an IPython kernel but
do not ship nbformat/nbclient/nbconvert. It supports the plain stream outputs
used by this repository's experiment notebook.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.notebook.resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    execution_count = 0

    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        source = "".join(cell.get("source", []))
        stdout = io.StringIO()
        stderr = io.StringIO()
        cell["execution_count"] = execution_count
        cell["outputs"] = []

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(source, f"{path.name}:cell-{cell_index}", "exec"), namespace)
        except Exception as exc:
            output_text = stdout.getvalue()
            error_text = stderr.getvalue()
            if output_text:
                cell["outputs"].append({"name": "stdout", "output_type": "stream", "text": output_text})
            if error_text:
                cell["outputs"].append({"name": "stderr", "output_type": "stream", "text": error_text})
            cell["outputs"].append(
                {
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "output_type": "error",
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
            path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            raise

        output_text = stdout.getvalue()
        error_text = stderr.getvalue()
        if output_text:
            cell["outputs"].append({"name": "stdout", "output_type": "stream", "text": output_text})
        if error_text:
            cell["outputs"].append({"name": "stderr", "output_type": "stream", "text": error_text})

    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Executed {execution_count} code cells successfully: {path}")


if __name__ == "__main__":
    main()
