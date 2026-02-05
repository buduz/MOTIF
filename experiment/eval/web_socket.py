import asyncio
import websockets.asyncio.server as _server
import websocket
import msgpack_numpy as m
import http
import logging
m.patch()
logger = logging.getLogger(__name__)

class WebSocketInferenceServer:
    def __init__(self, model, host="0.0.0.0", port=5550):
        self.model = model
        self.host = host
        self.port = port

    async def handler(self, websocket):
        async for message in websocket:
            logger.info(f"Connection from {websocket.remote_address} opened")
            try:
                request = m.unpackb(message)
                endpoint = request.get("endpoint")
                data = request.get("data", None)

                if endpoint == "get_action":
                    result = self.model.get_action(data)
                elif endpoint == "get_modality_config":
                    result = self.model.get_modality_config()
                else:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                await websocket.send(m.packb(result))
            except Exception as e:
                print(f"Server error: {e}")
                await websocket.send(b"ERROR")

    async def start(self):
        print(f"[WebSocket] Server running at ws://{self.host}:{self.port}")
        async with _server.serve(
            self.handler, 
            self.host, 
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    def run(self):
        asyncio.run(self.start()) 

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None

class WebSocketInferenceClient:
    def __init__(self, url):
        self.ws = websocket.WebSocket()
        self.ws.connect(url)

    def get_action(self, observation: dict) -> dict:
        request = {"endpoint": "get_action", "data": observation}
        self.ws.send(m.packb(request), opcode=websocket.ABNF.OPCODE_BINARY)
        response = self.ws.recv()
        if response == b"ERROR":
            raise RuntimeError("Inference failed")
        return m.unpackb(response)

    def get_modality_config(self) -> dict:
        request = {"endpoint": "get_modality_config"}
        self.ws.send(m.packb(request), opcode=websocket.ABNF.OPCODE_BINARY)
        response = self.ws.recv()
        return m.unpackb(response)

    def close(self):
        self.ws.close()