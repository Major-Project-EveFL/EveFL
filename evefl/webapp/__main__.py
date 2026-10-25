"""
    python -m evefl.webapp [--port 8765] [--no-browser]

Starts the demonstration app on the loopback interface and opens it in the default browser.
"""

from __future__ import annotations

import argparse
import logging

from evefl.webapp import server


def main() -> None:
    parser = argparse.ArgumentParser(description="EveFL demonstration app (loopback only)")
    parser.add_argument("--port", type=int, default=server.DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server.serve(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
