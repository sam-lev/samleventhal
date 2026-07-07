#!/usr/bin/env python3
"""Geodesics engine bridge — serves local bots to the web app over WebSocket.

Run:  python3 serve.py          (no dependencies; Python 3.8+)

On startup every bots/*.py module is imported and its module-level BOT object
is registered (see bots/random_bot.py for the contract). The web app connects
to ws://127.0.0.1:8765, receives the list of available bots for its Opponent
menu, and sends stateless `genmove` requests: full stones, legal mask, move
history and board adjacency out — one site index (or -1 for a pass) back.
Legality is always re-checked by the app's own rules engine.

The WebSocket layer below is a minimal RFC 6455 server (handshake, masked
client frames, fragmentation reassembly, ping/pong, close) — enough for a
single trusted localhost client, which is the entire threat model here: the
server binds 127.0.0.1 only.
"""

import base64
import hashlib
import importlib.util
import json
import os
import socketserver
import struct
import sys
import threading
import traceback

HOST, PORT = "127.0.0.1", 8765
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
BOTS = {}          # id -> bot object
BOT_LOCKS = {}     # id -> threading.Lock (bots may not be re-entrant)


# ---------------------------------------------------------------- bot loading

def load_bots():
    here = os.path.dirname(os.path.abspath(__file__))
    bdir = os.path.join(here, "bots")
    if not os.path.isdir(bdir):
        print("no bots/ directory found beside serve.py")
        return
    if bdir not in sys.path:          # let bot variants import their siblings
        sys.path.insert(0, bdir)
    for fname in sorted(os.listdir(bdir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        path = os.path.join(bdir, fname)
        try:
            spec = importlib.util.spec_from_file_location(
                "geobot_" + fname[:-3], path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            bot = getattr(mod, "BOT", None)
            if bot is None:
                print(f"  skip {fname}: no module-level BOT")
                continue
            avail = getattr(bot, "available", None)
            if avail is not None and not avail():
                print(f"  skip {bot.id} ({fname}): not available "
                      f"(see the module for setup)")
                continue
            if ":" in bot.id:
                print(f"  skip {fname}: bot id may not contain ':'")
                continue
            BOTS[bot.id] = bot
            BOT_LOCKS[bot.id] = threading.Lock()
            print(f"  bot '{bot.id}' — {bot.name} "
                  f"(levels: {', '.join(bot.levels)})")
        except Exception:
            print(f"  skip {fname}: import failed")
            traceback.print_exc()


def models_message():
    out = []
    for bot in BOTS.values():
        out.append({
            "id": bot.id,
            "name": bot.name,
            "levels": list(getattr(bot, "levels", ["standard"])),
            "supports": getattr(bot, "supports", None),
        })
    return {"type": "models", "models": out}


# ------------------------------------------------------------ websocket layer

def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def read_message(sock):
    """Read one complete (possibly fragmented) message. Handles ping/close.
    Returns str for text messages, None when the peer closes."""
    parts = []
    while True:
        b1, b2 = _recv_exact(sock, 2)
        fin = b1 & 0x80
        opcode = b1 & 0x0F
        masked = b2 & 0x80
        length = b2 & 0x7F
        if length == 126:
            (length,) = struct.unpack(">H", _recv_exact(sock, 2))
        elif length == 127:
            (length,) = struct.unpack(">Q", _recv_exact(sock, 8))
        mask = _recv_exact(sock, 4) if masked else b""
        payload = _recv_exact(sock, length) if length else b""
        if masked:
            payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
        if opcode == 0x8:                       # close
            try:
                send_frame(sock, payload[:2], opcode=0x8)
            except OSError:
                pass
            return None
        if opcode == 0x9:                       # ping -> pong
            send_frame(sock, payload, opcode=0xA)
            continue
        if opcode == 0xA:                       # unsolicited pong
            continue
        if opcode in (0x1, 0x2, 0x0):           # text / binary / continuation
            parts.append(payload)
            if fin:
                return b"".join(parts).decode("utf-8")
            continue
        raise ConnectionError(f"unsupported opcode {opcode}")


def send_frame(sock, payload, opcode=0x1):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    n = len(payload)
    head = bytes([0x80 | opcode])
    if n < 126:
        head += bytes([n])
    elif n < 65536:
        head += bytes([126]) + struct.pack(">H", n)
    else:
        head += bytes([127]) + struct.pack(">Q", n)
    sock.sendall(head + payload)


def handshake(sock):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("closed during handshake")
        data += chunk
        if len(data) > 65536:
            raise ConnectionError("oversized handshake")
    key = None
    for line in data.split(b"\r\n"):
        if line.lower().startswith(b"sec-websocket-key:"):
            key = line.split(b":", 1)[1].strip().decode()
    if not key:
        raise ConnectionError("not a websocket request")
    accept = base64.b64encode(
        hashlib.sha1((key + GUID).encode()).digest()).decode()
    sock.sendall((
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())


# ----------------------------------------------------------------- dispatcher

def handle_request(msg):
    kind = msg.get("type")
    if kind == "hello":
        return models_message()
    if kind == "genmove":
        rid = msg.get("id")
        bot = BOTS.get(msg.get("model"))
        if bot is None:
            return {"type": "error", "id": rid,
                    "message": f"unknown model '{msg.get('model')}'"}
        try:
            with BOT_LOCKS[bot.id]:
                result = bot.genmove(msg)
            move, info = (result if isinstance(result, tuple)
                          else (result, bot.name))
            return {"type": "move", "id": rid, "move": int(move), "info": info}
        except Exception as e:
            traceback.print_exc()
            return {"type": "error", "id": rid, "message": str(e)}
    return None


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        try:
            handshake(sock)
            print(f"client connected ({self.client_address[0]})")
            while True:
                text = read_message(sock)
                if text is None:
                    break
                try:
                    msg = json.loads(text)
                except ValueError:
                    continue
                reply = handle_request(msg)
                if reply is not None:
                    send_frame(sock, json.dumps(reply))
        except (ConnectionError, OSError):
            pass
        finally:
            print("client disconnected")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print("Geodesics bridge — scanning bots/")
    load_bots()
    if not BOTS:
        print("no bots available; serving an empty engine list")
    print(f"listening on ws://{HOST}:{port}  (Ctrl-C to stop)")
    try:
        Server((HOST, port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
