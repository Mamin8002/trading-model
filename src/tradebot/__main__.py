"""`python -m tradebot` — единая точка входа (cli.run)."""

import sys

from .cli.run import main

if __name__ == "__main__":
    sys.exit(main())
