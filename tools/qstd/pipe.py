"""Q-STD: the engine's logits stream through a Windows named pipe instead of a file (no dump on disk).

The engine opens STRATA_LOGITS_DUMP once (fopen "wb") and flushes each record {int32 magic, int32 T, int32 n_vocab,
int32 record, int64 position} + T*n_vocab FP32 rows. PipeReader is the pipe's server: its thread parses records as they
arrive and hands each row to a callback, tagged with what the runner set before the request (case, phase). After a
request's DONE line the runner calls drain(): every byte the engine wrote for that request is then processed.

Only the reader thread touches the pipe handle, and it never blocks in ReadFile: it looks (PeekNamedPipe) and reads only
what is there. (A synchronous ReadFile pending on the handle makes any other call on it wait until data comes: the first
version's drain() deadlocked on exactly that, 2026-11-13.)
"""
import ctypes
import struct
import threading
import time
from ctypes import wintypes

import numpy as np

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateNamedPipeW.restype = wintypes.HANDLE
k32.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
k32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
k32.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
k32.CloseHandle.argtypes = [wintypes.HANDLE]
INVALID = wintypes.HANDLE(-1).value
HDR = struct.Struct("<iiiiq")
MAGIC = 0x53474F4C
CHUNK = 16 << 20


class PipeReader:
    def __init__(self, name, on_row):
        self.path = r"\\.\pipe" + "\\" + name
        self.on_row = on_row          # on_row(tag, position, row_float32)
        self.tag = None
        self.h = k32.CreateNamedPipeW(self.path, 0x1, 0x0, 1, 0, 64 << 20, 0, None)   # inbound, byte mode, blocking connect
        if self.h in (None, INVALID):
            raise OSError(f"CreateNamedPipe failed: {ctypes.get_last_error()}")
        self.buf = bytearray()
        self.connected = threading.Event()
        self.idle_seq = 0             # +1 each time the thread finds the pipe empty with no record left to process
        self.closed = False
        self.error = None
        self.records = 0
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        try:
            if not k32.ConnectNamedPipe(self.h, None) and ctypes.get_last_error() != 535:   # 535: already connected
                raise OSError(f"ConnectNamedPipe failed: {ctypes.get_last_error()}")
            self.connected.set()
            tmp = ctypes.create_string_buffer(CHUNK)
            got, avail = wintypes.DWORD(0), wintypes.DWORD(0)
            while True:
                self._parse()
                if not k32.PeekNamedPipe(self.h, None, 0, None, ctypes.byref(avail), None):
                    err = ctypes.get_last_error()
                    if err == 109:            # the engine closed its end
                        break
                    raise OSError(f"PeekNamedPipe failed: {err}")
                if avail.value == 0:
                    if len(self.buf) < HDR.size:
                        self.idle_seq += 1
                    time.sleep(0.001)
                    continue
                if not k32.ReadFile(self.h, tmp, min(avail.value, CHUNK), ctypes.byref(got), None):
                    err = ctypes.get_last_error()
                    if err == 109:
                        break
                    raise OSError(f"ReadFile failed: {err}")
                self.buf += tmp.raw[:got.value]
            self._parse()
        except Exception as e:                # noqa: BLE001 - reported to the runner, which marks the arm invalid
            self.error = repr(e)
        finally:
            self.closed = True

    def _parse(self):
        while len(self.buf) >= HDR.size:
            magic, T, nv, rec, pos = HDR.unpack_from(self.buf, 0)
            if magic != MAGIC:
                raise ValueError(f"bad record header at record {self.records}")
            need = HDR.size + T * nv * 4
            if len(self.buf) < need:
                return
            rows = np.frombuffer(bytes(self.buf[HDR.size:need]), dtype=np.float32).reshape(T, nv)
            del self.buf[:need]
            with self.lock:
                tag = self.tag
            for t in range(T):
                self.on_row(tag, int(pos) + t, rows[t])
            self.records += 1

    def set_tag(self, tag):
        with self.lock:
            self.tag = tag

    def drain(self, timeout=600):
        """After a request's DONE: wait until the thread has found the pipe empty twice with nothing left to process."""
        t0 = time.time()
        while not self.connected.is_set():
            if self.error or self.closed or time.time() - t0 > 10:
                self.error = self.error or "the engine never opened the logits pipe (no record for this request)"
                return
            time.sleep(0.01)
        s0 = self.idle_seq
        while time.time() - t0 < timeout:
            if self.error or self.closed:
                return
            if self.idle_seq >= s0 + 2:
                return
            time.sleep(0.002)
        self.error = self.error or "drain timed out"

    def close(self):
        self.t.join(timeout=60)
        k32.CloseHandle(self.h)
