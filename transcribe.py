import sys
from pathlib import Path


def _bootstrap_import_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    sys.path.insert(0, str(src))


def main() -> None:
    _bootstrap_import_path()
    from transcripto.cli_transcribe import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
