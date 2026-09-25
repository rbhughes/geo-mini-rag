"""Point it at a file and see what the pipeline would read out of it.

    python -m geo_mini_rag.ep.inspect data/raw/las/some.las
    python -m geo_mini_rag.ep.inspect --facts data/raw/gis/layer.shp

The handler that claims the file does the work, so this shows exactly what
ingest would store: the text that gets chunked and embedded, and the facts that
get filtered and ranked on. No API key, no database, nothing written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geo_mini_rag import ep, settings
from geo_mini_rag.rag.extract import HEAD_BYTES, Skip
from geo_mini_rag.rag.parse import parse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--facts", action="store_true",
                        help="Print the metadata facts instead of the text.")
    parser.add_argument("--chunks", type=int, default=0, metavar="N",
                        help="Also print the first N segments after the summary.")
    args = parser.parse_args(argv)

    cfg = settings.load_rag_config()
    failed = False
    for n, path in enumerate(args.files):
        if n:
            print("\n" + "=" * 72 + "\n")
        print(f"{path}")
        try:
            with path.open("rb") as f:
                head = f.read(HEAD_BYTES)
            handler = ep.find(path, head)
            print(f"  handler: {handler.name}" if handler else
                  "  (no E&P handler; read as an ordinary document)")
            parsed = parse(path, cfg)
        except (Skip, OSError, ValueError) as exc:
            failed = True
            print(f"  cannot read: {exc}")
            continue

        if args.facts:
            for key, value in sorted(parsed.metadata.items()):
                shown = value if not isinstance(value, list) else \
                    f"{len(value)} values: {', '.join(map(str, value[:6]))}" \
                    + (" ..." if len(value) > 6 else "")
                print(f"  {key:22} {shown}")
        else:
            print()
            print(parsed.segments[0][1] if parsed.segments else "  (no text)")
        for segment in parsed.segments[1 : 1 + args.chunks]:
            print("\n" + "-" * 40 + "\n" + segment[1])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
