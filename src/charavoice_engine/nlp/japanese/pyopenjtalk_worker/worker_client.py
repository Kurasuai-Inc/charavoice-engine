from __future__ import annotations

import logging
import socket
import threading
from typing import Any, cast

from charavoice_engine.nlp.japanese.pyopenjtalk_worker.worker_common import (
    WORKER_HOST,
    RequestType,
    receive_data,
    send_data,
)

logger = logging.getLogger(__name__)


class WorkerClient:
    """pyopenjtalk worker client"""

    def __init__(self, port: int, host: str = WORKER_HOST) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # timeout: 60 seconds
        sock.settimeout(60)
        sock.connect((host, port))
        self.sock = sock
        self._lock = threading.Lock()

    def __enter__(self) -> WorkerClient:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self.sock.close()

    def dispatch_pyopenjtalk(self, func: str, *args: Any, **kwargs: Any) -> Any:
        data = {
            "request-type": RequestType.PYOPENJTALK,
            "func": func,
            "args": args,
            "kwargs": kwargs,
        }
        with self._lock:
            logger.debug(f"client sends request: {data}")
            send_data(self.sock, data)
            logger.debug("client sent request successfully")
            response = receive_data(self.sock)
            logger.debug(f"client received response: {response}")
        return response.get("return")

    def status(self) -> int:
        data = {"request-type": RequestType.STATUS}
        logger.debug(f"client sends request: {data}")
        send_data(self.sock, data)
        logger.debug("client sent request successfully")
        response = receive_data(self.sock)
        logger.debug(f"client received response: {response}")
        return cast(int, response.get("client-count"))

    def quit_server(self) -> None:
        data = {"request-type": RequestType.QUIT_SERVER}
        logger.debug(f"client sends request: {data}")
        send_data(self.sock, data)
        logger.debug("client sent request successfully")
        response = receive_data(self.sock)
        logger.debug(f"client received response: {response}")
