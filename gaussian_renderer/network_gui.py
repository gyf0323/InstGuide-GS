#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import traceback
import socket
import json
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = None


def _close_socket(sock):
    if sock is None:
        return
    try:
        sock.close()
    except OSError:
        pass


def _read_exact(sock, num_bytes):
    chunks = bytearray()
    while len(chunks) < num_bytes:
        chunk = sock.recv(num_bytes - len(chunks))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data.")
        chunks.extend(chunk)
    return bytes(chunks)

def init(wish_host, wish_port):
    global host, port, listener, conn, addr
    host = wish_host
    port = wish_port
    _close_socket(conn)
    conn = None
    addr = None
    _close_socket(listener)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen()
    # Training polls this socket every iteration.  A blocking timeout here
    # silently caps headless/server training at about 2 it/s, so keep accept()
    # non-blocking unless a viewer is actually connected.
    listener.setblocking(False)

def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except (socket.timeout, BlockingIOError):
        pass
            
def read():
    global conn
    messageLength = _read_exact(conn, 4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = _read_exact(conn, messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    verify_bytes = verify.encode("utf-8")
    conn.sendall(len(verify_bytes).to_bytes(4, 'little'))
    conn.sendall(verify_bytes)

def receive():
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            render_mode = int(message.get("render_mode", 0))
            semantic_query = str(message.get("semantic_query", ""))
            semantic_overlay_alpha = float(message.get("semantic_overlay_alpha", 0.55))
            semantic_temperature = float(message.get("semantic_temperature", 15.0))
            world_view_transform = torch.reshape(
                torch.tensor(message["view_matrix"], dtype=torch.float32, device="cuda"),
                (4, 4),
            )
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(
                torch.tensor(message["view_projection_matrix"], dtype=torch.float32, device="cuda"),
                (4, 4),
            )
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        request = {
            "render_mode": render_mode,
            "semantic_query": semantic_query,
            "semantic_overlay_alpha": semantic_overlay_alpha,
            "semantic_temperature": semantic_temperature,
        }
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier, request
    else:
        return None, None, None, None, None, None, {}
