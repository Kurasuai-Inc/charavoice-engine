from __future__ import annotations

import argparse

from charavoice_engine.nlp.japanese.pyopenjtalk_worker.worker_common import WORKER_HOST, WORKER_PORT
from charavoice_engine.nlp.japanese.pyopenjtalk_worker.worker_server import WorkerServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=WORKER_HOST)
    parser.add_argument("--port", type=int, default=WORKER_PORT)
    parser.add_argument("--no-client-timeout", type=int, default=120)
    args = parser.parse_args()
    server = WorkerServer()
    server.start_server(host=args.host, port=args.port, no_client_timeout=args.no_client_timeout)


if __name__ == "__main__":
    main()
