"""Synchronous OpenPI WebSocket client adapted from Apache-2.0 upstream."""

from __future__ import annotations

import logging
import time

from websockets.sync.client import connect

from . import msgpack_numpy


class WebsocketPolicyClient:
    """Call an OpenPI policy server with the official binary protocol."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._api_key = api_key
        self._packer = msgpack_numpy.Packer()
        self._socket, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        """Return metadata sent by the policy server during connection setup."""
        return self._server_metadata

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        """Send one observation and synchronously return one action chunk."""
        self._socket.send(self._packer.pack(observation))
        response = self._socket.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        result = msgpack_numpy.unpackb(response)
        if not isinstance(result, dict):
            raise RuntimeError("OpenPI policy response must be a mapping")
        return result

    def close(self) -> None:
        """Close the policy connection."""
        self._socket.close()

    def _wait_for_server(self):
        logging.info("Waiting for OpenPI policy server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                socket = connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = msgpack_numpy.unpackb(socket.recv())
                return socket, metadata
            except ConnectionRefusedError:
                logging.info("OpenPI server is not ready; retrying in 5 seconds...")
                time.sleep(5)
