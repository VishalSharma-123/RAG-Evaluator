from __future__ import annotations

import warnings

warnings.filterwarnings(
      "ignore",
      category=DeprecationWarning,
      module=r"chromadb\..*",
)

from dotenv import load_dotenv  # noqa: E402

from rag_evaluator.commands.parser import build_parser  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error(f"Unknown command: {getattr(args, 'command', None)}")

    return int(handler(args))
