"""
RDS / RT+ (RadioText Plus) Soundcard Encoder (Tkinter) — main.py (STABLE UI VERSION)
-----------------------------------------------------------------------------------

✅ Keeps the same UI structure/features we built (no random removals)
✅ PS fixed (Group 0A PS chars are in Block D)
✅ Manual RT field restored (checkbox + entry)
✅ nowplaying.txt watcher auto-updates RT/RT+
✅ Config JSON Load/Save/Save As
✅ Autostart option
✅ Tooltips (mouse hover descriptions)
✅ ECC (Group 1A) + AF (Group 0A Block C) + UI fields for both
✅ NEW: Subcarrier Hz field (57k real RDS; 1000 Hz audible debug)

IMPORTANT:
- 57 kHz is real RDS. If you set subcarrier to 1 kHz you WILL hear the data,
  but it will NOT decode as RDS on radios. (debug/listening only)

Install:
  pip install sounddevice numpy
"""

from __future__ import annotations

import os
import json
import math
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -----------------------------
# Constants
# -----------------------------
CONFIG_PATH_DEFAULT = "rds_rtplus_encoder_config.json"
CONFIG_VERSION = 1

RDS_BITRATE = 1187.5
RDS_CHIPRATE = 2 * RDS_BITRATE  # 2375 Hz

RDS_CRC_POLY = 0x5B9  # CRC-10 poly

OFFSET_A = 0x0FC
OFFSET_B = 0x198
OFFSET_C = 0x168
OFFSET_D = 0x1B4

RTPLUS_AID = 0x4BD7
RTPLUS_GROUP_TYPE = 11  # 11A tags

# RT+ content-type codes (common)
CT_TITLE = 1
CT_ARTIST = 4


# -----------------------------
# Helpers
# -----------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def u16(x: int) -> int:
    return x & 0xFFFF


def bits_msb_first(value: int, nbits: int):
    for i in range(nbits):
        yield 1 if (value >> (nbits - 1 - i)) & 1 else 0


def crc10_rds(data16: int) -> int:
    reg = (data16 & 0xFFFF) << 10
    for bit in range(25, 9, -1):
        if (reg >> bit) & 1:
            reg ^= (RDS_CRC_POLY << (bit - 10))
    return reg & 0x3FF


def block26(data16: int, offset10: int) -> int:
    c = crc10_rds(data16) ^ (offset10 & 0x3FF)
    return ((data16 & 0xFFFF) << 10) | (c & 0x3FF)


def safe_latin1(s: str) -> str:
    return (s or "").encode("latin-1", errors="replace").decode("latin-1", errors="replace")


def ru_transliterate(s: str) -> str:
    m = {
        "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "Yo", "Ж": "Zh", "З": "Z", "И": "I", "Й": "Y",
        "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U", "Ф": "F",
        "Х": "Kh", "Ц": "Ts", "Ч": "Ch", "Ш": "Sh", "Щ": "Sch", "Ъ": "", "Ы": "Y", "Ь": "", "Э": "E", "Ю": "Yu", "Я": "Ya",
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "i", "й": "y",
        "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
        "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
    return "".join(m.get(ch, ch) for ch in (s or ""))


def parse_nowplaying_text(text: str) -> Dict[str, str]:
    """
    Accepts:
      - Artist=...; Title=...; Album=...
      - lines: Artist: ... / Title: ... / Album: ...
      - fallback: "Artist - Title" or whole line as Title
    """
    out = {"artist": "", "title": "", "album": ""}
    t = (text or "").strip()
    if not t:
        return out

    if "=" in t and (";" in t or "\n" not in t):
        parts = [p.strip() for p in t.split(";") if p.strip()]
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if k in ("artist", "исполнитель", "артист"):
                out["artist"] = v
            elif k in ("title", "song", "track", "название"):
                out["title"] = v
            elif k in ("album", "альбом"):
                out["album"] = v
        if out["artist"] or out["title"]:
            return out

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in lines:
        low = ln.lower()
        if low.startswith("artist:") or low.startswith("исполнитель:") or low.startswith("артист:"):
            out["artist"] = ln.split(":", 1)[1].strip()
        elif low.startswith("title:") or low.startswith("track:") or low.startswith("song:") or low.startswith("название:"):
            out["title"] = ln.split(":", 1)[1].strip()
        elif low.startswith("album:") or low.startswith("альбом:"):
            out["album"] = ln.split(":", 1)[1].strip()

    if out["artist"] or out["title"]:
        return out

    if " - " in t:
        a, b = t.split(" - ", 1)
        out["artist"] = a.strip()
        out["title"] = b.strip()
        return out

    out["title"] = t
    return out


def af_code_from_mhz(freq_mhz: float) -> int:
    """
    RDS AF frequency code:
      code = round((freq_mhz - 87.5)/0.1)
      valid: 1..204 (87.6..107.9)
    """
    try:
        code = int(round((float(freq_mhz) - 87.5) / 0.1))
    except Exception:
        return 0
    if 1 <= code <= 204:
        return code
    return 0


# -----------------------------
# Tooltip
# -----------------------------
class ToolTip:
    def __init__(self, widget, text: str, delay_ms: int = 450, wrap: int = 520):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wrap = wrap
        self._after = None
        self._tip = None
        self._xy = (0, 0)

        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        widget.bind("<ButtonPress>", self._leave, add="+")
        widget.bind("<Motion>", self._motion, add="+")

    def _motion(self, e):
        self._xy = (e.x_root, e.y_root)

    def _enter(self, _e=None):
        self._cancel()
        self._after = self.widget.after(self.delay_ms, self._show)

    def _leave(self, _e=None):
        self._cancel()
        self._hide()

    def _cancel(self):
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def _show(self):
        if self._tip is not None:
            return
        x, y = self._xy
        if x == 0 and y == 0:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        else:
            x += 14
            y += 18

        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        try:
            self._tip.wm_attributes("-topmost", True)
        except Exception:
            pass

        frm = ttk.Frame(self._tip, padding=8)
        frm.pack(fill="both", expand=True)
        lbl = ttk.Label(frm, text=self.text, justify="left", wraplength=self.wrap)
        lbl.pack(fill="both", expand=True)

        try:
            self._tip.wm_geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _hide(self):
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


# -----------------------------
# Config
# -----------------------------
@dataclass
class AppConfig:
    version: int = CONFIG_VERSION
    device_id: Optional[int] = None
    device_label: str = ""

    sample_rate: int = 192000
    amplitude: float = 0.10
    carrier_leak: float = 0.02
    swap_01: bool = True

    # NEW: subcarrier Hz (57k = real RDS, 1k = audible debug)
    subcarrier_hz: float = 57_000.0

    # station identity
    pi: int = 0x3377  # KNLF
    ps: str = "KNLF FM "  # 8 chars static
    pty: int = 10
    tp: bool = True
    ta: bool = False
    ms: bool = True

    # ECC + AF
    ecc_enabled: bool = True
    ecc: int = 0xA0  # USA RBDS ECC
    ecc_interval_sec: float = 1.5

    af_enabled: bool = True
    af_mhz: float = 90.9

    # RT
    rt_format: str = "{title} - {artist}"
    translit_ru: bool = True

    # manual RT field
    manual_rt_enabled: bool = False
    manual_rt_text: str = ""

    # RT+
    enable_rtplus: bool = True

    # scheduling
    ps_every_n_groups: int = 2

    # nowplaying
    nowplaying_path: str = ""
    watch_enabled: bool = False
    watch_interval: float = 0.6
    auto_apply_live: bool = True

    # autostart
    autostart: bool = False


def load_config(path: str) -> AppConfig:
    if not os.path.exists(path):
        return AppConfig()
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        base = asdict(AppConfig())
        base.update(d or {})
        return AppConfig(**base)
    except Exception:
        return AppConfig()


def save_config(path: str, cfg: AppConfig) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


# -----------------------------
# Differential Manchester (with swap)
# -----------------------------
class DifferentialManchester:
    def __init__(self, initial_level: int = 1, swap_01: bool = True):
        self.level = 1.0 if initial_level else -1.0
        self.swap_01 = bool(swap_01)

    def set_swap(self, swap_01: bool):
        self.swap_01 = bool(swap_01)

    def bit_to_chips(self, bit: int) -> Tuple[float, float]:
        b = int(bit) & 1
        if self.swap_01:
            b ^= 1

        # Known-good mapping style from earlier working versions
        if b == 1:
            first = -self.level
            second = self.level
        else:
            first = self.level
            second = -self.level

        self.level = second
        return first, second


# -----------------------------
# RDS Builder
# -----------------------------
class RDSBuilder:
    def __init__(self, cfg: AppConfig):
        self._lock = threading.Lock()
        self.cfg = cfg

        self.ps_index = 0
        self.rt_index = 0
        self.group_counter = 0

        self.rt_ab = 0
        self.rtplus_item_toggle = 0

        self._rt64 = " " * 64
        self._last_item_sig = ""

        self._span_title = (0, 0)
        self._span_artist = (0, 0)

        self._last_3a = 0.0
        self._last_11a = 0.0
        self._last_1a = 0.0

    def update_config(self, cfg: AppConfig):
        with self._lock:
            self.cfg = cfg

    def _get_cfg(self) -> AppConfig:
        with self._lock:
            return self.cfg

    def set_rt_from_meta(self, meta: Dict[str, str]):
        cfg = self._get_cfg()
        artist = meta.get("artist", "") or ""
        title = meta.get("title", "") or ""
        album = meta.get("album", "") or ""

        try:
            rt_fmt = (cfg.rt_format or "{title} - {artist}").format(
                artist=artist, title=title, album=album
            ).strip()
        except Exception:
            rt_fmt = f"{title} - {artist}".strip()

        if not rt_fmt:
            rt_fmt = title or artist or ""

        rt_src = (cfg.manual_rt_text or "").strip() if cfg.manual_rt_enabled else rt_fmt

        rt_src = safe_latin1(rt_src)
        if cfg.translit_ru:
            rt_src = safe_latin1(ru_transliterate(rt_src))

        rt = rt_src[:64].ljust(64)

        item_sig = f"{artist}|{title}|{album}"
        if item_sig != self._last_item_sig:
            self._last_item_sig = item_sig
            self.rt_ab ^= 1
            self.rtplus_item_toggle ^= 1

        self._rt64 = rt

        def find_span(hay: str, needle: str) -> Tuple[int, int]:
            if not needle:
                return (0, 0)
            nd = needle
            if cfg.translit_ru:
                nd = ru_transliterate(nd)
            nd = safe_latin1(nd)
            idx = hay.find(nd)
            if idx < 0:
                return (0, 0)
            return (idx, len(nd))

        self._span_title = find_span(self._rt64, title)
        self._span_artist = find_span(self._rt64, artist)

        if (not cfg.manual_rt_enabled) and title and self._span_title == (0, 0):
            alt = f"{title} - {artist}".strip()
            if cfg.translit_ru:
                alt = ru_transliterate(alt)
            alt = safe_latin1(alt)[:64].ljust(64)
            self._rt64 = alt
            self._span_title = find_span(self._rt64, title)
            self._span_artist = find_span(self._rt64, artist)

    def _block_words(self, b1: int, b2: int, b3: int, b4: int) -> Tuple[int, int, int, int]:
        return (
            block26(u16(b1), OFFSET_A),
            block26(u16(b2), OFFSET_B),
            block26(u16(b3), OFFSET_C),
            block26(u16(b4), OFFSET_D),
        )

    def _group_header_b2(self, group_type: int, versionA: int, tp: int, pty: int) -> int:
        return (
            ((group_type & 0xF) << 12)
            | ((0 if versionA else 1) << 11)
            | ((tp & 1) << 10)
            | ((pty & 0x1F) << 5)
        )

    # Group 0A (PS + AF) — PS chars are in Block D
    def group_0A(self, seg: int) -> Tuple[int, int, int, int]:
        cfg = self._get_cfg()
        ps = (cfg.ps or "")[:8].ljust(8)

        c1 = ord(ps[2 * (seg & 3)]) & 0xFF
        c2 = ord(ps[2 * (seg & 3) + 1]) & 0xFF

        b2 = self._group_header_b2(0, 1, int(cfg.tp), int(cfg.pty))
        b2 |= (int(cfg.ta) & 1) << 4
        b2 |= (int(cfg.ms) & 1) << 3
        b2 |= (seg & 3)

        # Block C: AF + DI (we do AF only; DI=0)
        if cfg.af_enabled:
            code = af_code_from_mhz(cfg.af_mhz)
            if code:
                b3 = (0xE1 << 8) | (code & 0xFF)  # "1 AF follows" + AF code
            else:
                b3 = 0x0000
        else:
            b3 = 0x0000

        b4 = (c1 << 8) | c2
        return self._block_words(int(cfg.pi), b2, b3, b4)

    # Group 1A (ECC)
    def group_1A_ecc(self) -> Tuple[int, int, int, int]:
        cfg = self._get_cfg()
        b2 = self._group_header_b2(1, 1, int(cfg.tp), int(cfg.pty))
        ecc = int(cfg.ecc) & 0xFF
        b3 = (ecc << 8) | 0x00
        b4 = 0x0000
        return self._block_words(int(cfg.pi), b2, b3, b4)

    # Group 2A (RT)
    def group_2A(self, seg: int) -> Tuple[int, int, int, int]:
        cfg = self._get_cfg()
        rt = self._rt64
        i = 4 * (seg & 0xF)
        c = [ord(rt[i + k]) & 0xFF for k in range(4)]

        b2 = self._group_header_b2(2, 1, int(cfg.tp), int(cfg.pty))
        b2 |= (self.rt_ab & 1) << 4
        b2 |= (seg & 0xF)

        b3 = (c[0] << 8) | c[1]
        b4 = (c[2] << 8) | c[3]
        return self._block_words(int(cfg.pi), b2, b3, b4)

    # RT+ announce (3A)
    def group_3A_rtplus_announce(self) -> Tuple[int, int, int, int]:
        cfg = self._get_cfg()
        b2 = self._group_header_b2(3, 1, int(cfg.tp), int(cfg.pty))
        b3 = RTPLUS_AID
        b4 = (RTPLUS_GROUP_TYPE & 0xF) << 12
        return self._block_words(int(cfg.pi), b2, b3, b4)

    # RT+ tags payload (37 bits)
    def _pack_rtplus_payload_37(self) -> Tuple[int, int, int]:
        t_start, t_len = self._span_title
        a_start, a_len = self._span_artist

        t_lm = 0 if t_len <= 0 else int(clamp(t_len - 1, 0, 63))
        a_lm = 0 if a_len <= 0 else int(clamp(a_len - 1, 0, 63))

        a_lm_5 = int(clamp(a_lm, 0, 31))
        t_lm_5 = int(clamp(t_lm, 0, 31))

        tag1_type, tag1_start, tag1_lenm = CT_TITLE, int(clamp(t_start, 0, 63)), t_lm
        tag2_type, tag2_start, tag2_lenm = CT_ARTIST, int(clamp(a_start, 0, 63)), a_lm_5

        if a_len > 32 and t_len <= 32:
            tag1_type, tag1_start, tag1_lenm = CT_ARTIST, int(clamp(a_start, 0, 63)), a_lm
            tag2_type, tag2_start, tag2_lenm = CT_TITLE, int(clamp(t_start, 0, 63)), t_lm_5

        item_toggle = self.rtplus_item_toggle & 1
        item_running = 1

        stream = 0

        def push(val: int, bits: int):
            nonlocal stream
            stream = (stream << bits) | (val & ((1 << bits) - 1))

        push(item_toggle, 1)
        push(tag1_type, 6)
        push(tag1_start, 6)
        push(tag1_lenm, 6)
        push(tag2_type, 6)
        push(tag2_start, 6)
        push(tag2_lenm, 5)
        push(item_running, 1)

        top5 = (stream >> 32) & 0x1F
        mid16 = (stream >> 16) & 0xFFFF
        low16 = stream & 0xFFFF

        top5 &= 0x1E
        return top5, mid16, low16

    # Group 11A (RT+ tags)
    def group_11A_rtplus_tags(self) -> Tuple[int, int, int, int]:
        cfg = self._get_cfg()
        msg5, b3, b4 = self._pack_rtplus_payload_37()
        b2 = self._group_header_b2(11, 1, int(cfg.tp), int(cfg.pty))
        b2 |= (msg5 & 0x1F)
        return self._block_words(int(cfg.pi), b2, b3, b4)

    def next_group(self) -> Tuple[int, int, int, int]:
        cfg = self._get_cfg()
        now = time.monotonic()

        # ECC scheduling
        if cfg.ecc_enabled:
            interval = float(cfg.ecc_interval_sec) if cfg.ecc_interval_sec > 0.2 else 0.2
            if (now - self._last_1a) >= interval:
                self._last_1a = now
                return self.group_1A_ecc()

        # RT+ scheduling
        if cfg.enable_rtplus:
            if (now - self._last_3a) >= 10.0:
                self._last_3a = now
                return self.group_3A_rtplus_announce()
            if (now - self._last_11a) >= 2.0:
                self._last_11a = now
                return self.group_11A_rtplus_tags()

        # PS/RT scheduling
        self.group_counter = (self.group_counter + 1) & 0xFFFFFFFF
        n = max(1, int(cfg.ps_every_n_groups))

        if (self.group_counter % n) == 0:
            g = self.group_0A(self.ps_index)
            self.ps_index = (self.ps_index + 1) % 4
            return g

        g = self.group_2A(self.rt_index)
        self.rt_index = (self.rt_index + 1) % 16
        return g

    # Infinite bit generator (never StopIteration)
    def next_bits(self):
        while True:
            wA, wB, wC, wD = self.next_group()
            for w in (wA, wB, wC, wD):
                yield from bits_msb_first(w, 26)


# -----------------------------
# Wave generator (chip_phase + shaping)
# -----------------------------
class RDSWaveGenerator:
    def __init__(self, cfg: AppConfig, builder: RDSBuilder):
        self._lock = threading.Lock()
        self.cfg = cfg
        self.builder = builder

        self.fs = float(cfg.sample_rate)

        self.phase = 0.0
        self.w = 2.0 * math.pi * (float(cfg.subcarrier_hz) / self.fs)

        self.dm = DifferentialManchester(initial_level=1, swap_01=cfg.swap_01)

        self.chip_phase = 0.0
        self.chip_inc = float(RDS_CHIPRATE / self.fs)

        self._chip_pair = (1.0, -1.0)
        self._chip_pair_index = 0
        self._current_chip = 1.0

        self._bit_iter = self.builder.next_bits()

        # simple 1-pole shaping to soften edges
        self._shaped = 0.0
        self._alpha = self._calc_alpha(fc_hz=2400.0)

        self._load_next_pair()

    def _calc_alpha(self, fc_hz: float) -> float:
        x = 2.0 * math.pi * float(fc_hz) / self.fs
        return x / (1.0 + x)

    def update_cfg(self, cfg: AppConfig):
        with self._lock:
            self.cfg = cfg
            self.dm.set_swap(cfg.swap_01)
            # allow live subcarrier changes (debug)
            self.w = 2.0 * math.pi * (float(cfg.subcarrier_hz) / self.fs)

    def _load_next_pair(self):
        try:
            bit = next(self._bit_iter)
        except StopIteration:
            self._bit_iter = self.builder.next_bits()
            bit = next(self._bit_iter)

        c1, c2 = self.dm.bit_to_chips(bit)
        self._chip_pair = (float(c1), float(c2))
        self._chip_pair_index = 0
        self._current_chip = self._chip_pair[0]

    def _advance_chip(self):
        self._chip_pair_index += 1
        if self._chip_pair_index >= 2:
            self._load_next_pair()
        else:
            self._current_chip = self._chip_pair[self._chip_pair_index]

    def render(self, nframes: int) -> np.ndarray:
        out = np.zeros(nframes, dtype=np.float32)

        with self._lock:
            amp = float(self.cfg.amplitude)
            leak = float(self.cfg.carrier_leak)
            alpha = float(self._alpha)

        ph = self.phase
        w = self.w
        chip_phase = self.chip_phase
        shaped = self._shaped
        current_chip = self._current_chip

        for i in range(nframes):
            chip_phase += self.chip_inc
            while chip_phase >= 1.0:
                chip_phase -= 1.0
                self._advance_chip()
                current_chip = self._current_chip

            shaped = shaped + alpha * (current_chip - shaped)
            out[i] = float(amp * ((shaped + leak) * math.sin(ph)))

            ph += w
            if ph >= 2.0 * math.pi:
                ph -= 2.0 * math.pi

        self.phase = ph
        self.chip_phase = chip_phase
        self._shaped = shaped
        self._current_chip = current_chip
        return out


# -----------------------------
# Audio engine
# -----------------------------
class AudioEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = None
        self._running = False

        self.builder: Optional[RDSBuilder] = None
        self.gen: Optional[RDSWaveGenerator] = None
        self.last_error: Optional[str] = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self, cfg: AppConfig, meta: Dict[str, str]):
        with self._lock:
            if self._running:
                return
            self.last_error = None

            self.builder = RDSBuilder(cfg)
            self.builder.set_rt_from_meta(meta)
            self.gen = RDSWaveGenerator(cfg, self.builder)

            def callback(outdata, frames, _time_info, _status):
                g = self.gen
                if g is None:
                    outdata[:] = 0
                    return
                outdata[:, 0] = g.render(frames)

            try:
                self._stream = sd.OutputStream(
                    device=cfg.device_id,
                    samplerate=int(cfg.sample_rate),
                    channels=1,
                    dtype="float32",
                    callback=callback,
                    blocksize=0,
                    latency="low",
                )
                self._stream.start()
            except Exception as e:
                self._stream = None
                self.builder = None
                self.gen = None
                self.last_error = str(e)
                raise

            self._running = True

    def stop(self):
        with self._lock:
            st = self._stream
            self._stream = None
            self._running = False
            self.builder = None
            self.gen = None

        if st is not None:
            try:
                st.abort()
            except Exception:
                pass
            try:
                st.stop()
            except Exception:
                pass
            try:
                st.close()
            except Exception:
                pass

    def apply_live(self, cfg: AppConfig):
        with self._lock:
            if not self._running:
                return
            if self.builder is not None:
                self.builder.update_config(cfg)
            if self.gen is not None:
                self.gen.update_cfg(cfg)

    def update_meta_live(self, meta: Dict[str, str]):
        with self._lock:
            if not self._running:
                return
            if self.builder is not None:
                self.builder.set_rt_from_meta(meta)


# -----------------------------
# GUI App
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RDS / RT+ Encoder (Soundcard)")
        self.geometry("1180x1010")
        self.minsize(1180, 1010)

        self.cfg_path = os.path.abspath(CONFIG_PATH_DEFAULT)
        self.cfg = load_config(self.cfg_path)

        self.engine = AudioEngine()
        self.meta = {"artist": "", "title": "", "album": ""}

        self._watch_thread: Optional[threading.Thread] = None
        self._watch_stop = threading.Event()
        self._last_mtime = 0.0

        self._devices = []

        self._build_ui()
        self._load_devices()
        self._cfg_to_ui()

        if self.cfg.nowplaying_path and os.path.exists(self.cfg.nowplaying_path):
            self._read_nowplaying(force=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(250, self._status_loop)
        self.after(350, self._maybe_autostart)

    # -------- UI build --------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 8}

        cfgrow = ttk.Frame(self)
        cfgrow.pack(fill="x", **pad)

        self.lbl_cfg = ttk.Label(cfgrow, text=f"Config: {self.cfg_path}")
        self.lbl_cfg.pack(side="left")

        btn_load = ttk.Button(cfgrow, text="Load config...", command=self._ui_load_config)
        btn_save = ttk.Button(cfgrow, text="Save config", command=self._ui_save_config)
        btn_saveas = ttk.Button(cfgrow, text="Save config as...", command=self._ui_saveas_config)
        btn_load.pack(side="right")
        btn_saveas.pack(side="right", padx=(8, 0))
        btn_save.pack(side="right", padx=(8, 0))

        ToolTip(btn_load, "Load settings from a JSON file.")
        ToolTip(btn_save, "Save settings to the current config file.")
        ToolTip(btn_saveas, "Save settings to a new JSON file.")

        # Audio box
        box_audio = ttk.LabelFrame(self, text="Audio Output", padding=10)
        box_audio.pack(fill="x", **pad)

        row1 = ttk.Frame(box_audio)
        row1.pack(fill="x", pady=6)

        ttk.Label(row1, text="Output device:").pack(side="left")
        self.dev_combo = ttk.Combobox(row1, state="readonly", width=80)
        self.dev_combo.pack(side="left", padx=8)
        btn_refresh = ttk.Button(row1, text="Refresh", command=self._load_devices_keep_selection)
        btn_refresh.pack(side="left")

        ToolTip(self.dev_combo, "Select the sound card output device.\nUse one that supports 192 kHz.")
        ToolTip(btn_refresh, "Re-scan audio devices.")

        row2 = ttk.Frame(box_audio)
        row2.pack(fill="x", pady=6)

        ttk.Label(row2, text="Sample rate:").pack(side="left")
        self.sr_var = tk.IntVar(value=192000)
        self.sr_combo = ttk.Combobox(row2, state="readonly", width=10, textvariable=self.sr_var)
        self.sr_combo["values"] = [48000, 96000, 192000]
        self.sr_combo.pack(side="left", padx=8)
        ToolTip(self.sr_combo, "192000 is recommended.\nChanging sample rate requires Stop + Start.")

        # NEW: subcarrier Hz
        ttk.Label(row2, text="Subcarrier Hz:").pack(side="left", padx=(16, 0))
        self.subc_var = tk.DoubleVar(value=57000.0)
        ent_subc = ttk.Entry(row2, width=10, textvariable=self.subc_var)
        ent_subc.pack(side="left", padx=8)
        ToolTip(ent_subc, "57kHz = real RDS.\n1000Hz = audible debug (won't decode as RDS).")

        ttk.Label(row2, text="Amplitude:").pack(side="left", padx=(16, 0))
        self.amp_var = tk.DoubleVar(value=0.10)
        self.amp_scale = ttk.Scale(row2, from_=0.0, to=0.30, variable=self.amp_var,
                                   command=lambda _e: self._on_amp())
        self.amp_scale.pack(side="left", fill="x", expand=True, padx=8)
        self.amp_lbl = ttk.Label(row2, text="0.10")
        self.amp_lbl.pack(side="left", padx=(0, 12))
        ToolTip(self.amp_scale, "RDS level. Too high can distort.\nTry 0.06–0.15.")

        ttk.Label(row2, text="Carrier leak:").pack(side="left")
        self.leak_var = tk.DoubleVar(value=0.02)
        self.leak_scale = ttk.Scale(row2, from_=0.0, to=0.20, variable=self.leak_var,
                                    command=lambda _e: self._on_leak())
        self.leak_scale.pack(side="left", fill="x", expand=True, padx=8)
        self.leak_lbl = ttk.Label(row2, text="0.02")
        self.leak_lbl.pack(side="left")
        ToolTip(self.leak_scale,
                "Adds a tiny unmodulated subcarrier.\n"
                "If spectrum is too narrow or decode fails, try 0.00.\n"
                "Typical: 0.00–0.03.\n"
                "For 1kHz audible debug, set this to 0.00 to hear data better.")

        # RDS box
        box_rds = ttk.LabelFrame(self, text="RDS Settings (PS + RT)", padding=10)
        box_rds.pack(fill="x", **pad)

        r = ttk.Frame(box_rds)
        r.pack(fill="x", pady=6)

        ttk.Label(r, text="PI (hex):").pack(side="left")
        self.pi_var = tk.StringVar(value="0x3377")
        ent_pi = ttk.Entry(r, width=10, textvariable=self.pi_var)
        ent_pi.pack(side="left", padx=8)
        ToolTip(ent_pi, "Program Identification (PI) code.\nFor callsign KNLF: 0x3377")

        ttk.Label(r, text="PTY:").pack(side="left", padx=(12, 0))
        self.pty_var = tk.IntVar(value=10)
        sp_pty = ttk.Spinbox(r, from_=0, to=31, width=5, textvariable=self.pty_var)
        sp_pty.pack(side="left", padx=8)
        ToolTip(sp_pty, "Program Type (0..31).")

        self.tp_var = tk.IntVar(value=1)
        self.ta_var = tk.IntVar(value=0)
        self.ms_var = tk.IntVar(value=1)
        cb_tp = ttk.Checkbutton(r, text="TP", variable=self.tp_var)
        cb_ta = ttk.Checkbutton(r, text="TA", variable=self.ta_var)
        cb_ms = ttk.Checkbutton(r, text="Music (MS)", variable=self.ms_var)
        cb_tp.pack(side="left", padx=(12, 0))
        cb_ta.pack(side="left")
        cb_ms.pack(side="left", padx=(8, 0))
        ToolTip(cb_tp, "TP flag (traffic program).")
        ToolTip(cb_ta, "TA flag (traffic announcement).")
        ToolTip(cb_ms, "MS flag: Music/Speech.")

        self.swap_var = tk.IntVar(value=1)
        cb_swap = ttk.Checkbutton(r, text="Swap 0/1 mapping", variable=self.swap_var, command=self._apply_if_live)
        cb_swap.pack(side="left", padx=(18, 0))
        ToolTip(cb_swap, "If decoding fails, toggle this.\nFlips differential Manchester mapping.")

        r2 = ttk.Frame(box_rds)
        r2.pack(fill="x", pady=6)

        ttk.Label(r2, text="PS (8 chars, static):").pack(side="left")
        self.ps_var = tk.StringVar(value="KNLF FM ")
        ent_ps = ttk.Entry(r2, width=14, textvariable=self.ps_var)
        ent_ps.pack(side="left", padx=8)
        ToolTip(ent_ps, "Program Service name (8 chars).\nKeep static = no scroll.\nPads with spaces.")

        ttk.Label(r2, text="PS every N groups:").pack(side="left", padx=(12, 0))
        self.psn_var = tk.IntVar(value=2)
        psn = ttk.Combobox(r2, state="readonly", width=6, textvariable=self.psn_var)
        psn["values"] = [1, 2, 3, 4, 6, 8, 12, 16]
        psn.pack(side="left", padx=8)
        psn.bind("<<ComboboxSelected>>", lambda _e: self._apply_if_live())
        ToolTip(psn, "Lower = PS repeats more often.\nHigher = more bandwidth for RT/RT+.\nTry 2 or 4.")

        # ECC + AF row
        r3 = ttk.Frame(box_rds)
        r3.pack(fill="x", pady=6)

        self.ecc_en_var = tk.IntVar(value=1)
        cb_ecc = ttk.Checkbutton(r3, text="Send ECC (1A)", variable=self.ecc_en_var, command=self._apply_if_live)
        cb_ecc.pack(side="left")
        ToolTip(cb_ecc, "Sends ECC using Group 1A.\nHelps some car radios trust station identity.\nUSA ECC = 0xA0")

        ttk.Label(r3, text="ECC (hex):").pack(side="left", padx=(12, 0))
        self.ecc_var = tk.StringVar(value="0xA0")
        ent_ecc = ttk.Entry(r3, width=8, textvariable=self.ecc_var)
        ent_ecc.pack(side="left", padx=8)
        ToolTip(ent_ecc, "Extended Country Code (ECC).\nUSA RBDS: 0xA0")

        ttk.Label(r3, text="ECC interval (s):").pack(side="left", padx=(12, 0))
        self.ecc_int_var = tk.DoubleVar(value=1.5)
        sp_ecci = ttk.Spinbox(r3, from_=0.3, to=10.0, increment=0.1, width=6, textvariable=self.ecc_int_var)
        sp_ecci.pack(side="left", padx=8)
        ToolTip(sp_ecci, "How often to transmit Group 1A with ECC.\nTry 1.0–2.0 seconds.")

        self.af_en_var = tk.IntVar(value=1)
        cb_af = ttk.Checkbutton(r3, text="Send AF (0A)", variable=self.af_en_var, command=self._apply_if_live)
        cb_af.pack(side="left", padx=(18, 0))
        ToolTip(cb_af, "Sends a simple AF list in Group 0A Block C.\nSome radios expect AF for name display.")

        ttk.Label(r3, text="AF MHz:").pack(side="left", padx=(12, 0))
        self.af_mhz_var = tk.DoubleVar(value=90.9)
        ent_af = ttk.Entry(r3, width=8, textvariable=self.af_mhz_var)
        ent_af.pack(side="left", padx=8)
        ToolTip(ent_af, "Alternative Frequency (AF) in MHz.\nOften set to your own frequency (e.g., 90.9).")

        # RT / RT+
        box_rt = ttk.LabelFrame(self, text="RT + RT+ (Title / Artist tags)", padding=10)
        box_rt.pack(fill="x", **pad)

        rr = ttk.Frame(box_rt)
        rr.pack(fill="x", pady=6)

        self.rtplus_var = tk.IntVar(value=1)
        cb_rtplus = ttk.Checkbutton(rr, text="Enable RT+ (3A + 11A)", variable=self.rtplus_var, command=self._apply_if_live)
        cb_rtplus.pack(side="left")
        ToolTip(cb_rtplus, "Sends RT+ ODA announce (3A) and tags (11A).\nSome receivers ignore it; RT still works.")

        self.tr_var = tk.IntVar(value=1)
        cb_tr = ttk.Checkbutton(rr, text="Transliterate Russian", variable=self.tr_var, command=self._apply_if_live)
        cb_tr.pack(side="left", padx=(18, 0))
        ToolTip(cb_tr, "Converts Cyrillic to Latin so most radios display RT correctly.")

        self.autostart_var = tk.IntVar(value=0)
        cb_as = ttk.Checkbutton(rr, text="Autostart", variable=self.autostart_var)
        cb_as.pack(side="left", padx=(18, 0))
        ToolTip(cb_as, "Automatically starts transmission when you launch the app.")

        # Manual RT row
        rr_manual = ttk.Frame(box_rt)
        rr_manual.pack(fill="x", pady=6)

        self.manual_rt_var = tk.IntVar(value=0)
        cb_manual = ttk.Checkbutton(rr_manual, text="Use manual RT", variable=self.manual_rt_var, command=self._apply_if_live)
        cb_manual.pack(side="left")
        ToolTip(cb_manual, "When enabled, RT is taken from the Manual RT field.\nRT+ tags still follow the nowplaying meta.")

        ttk.Label(rr_manual, text="Manual RT:").pack(side="left", padx=(12, 0))
        self.manual_rt_text_var = tk.StringVar(value="")
        ent_manual_rt = ttk.Entry(rr_manual, width=80, textvariable=self.manual_rt_text_var)
        ent_manual_rt.pack(side="left", padx=8, fill="x", expand=True)
        ToolTip(ent_manual_rt, "Manual RadioText (up to 64 chars).\nIf transliteration is enabled, Cyrillic becomes Latin.")

        rr2 = ttk.Frame(box_rt)
        rr2.pack(fill="x", pady=6)

        ttk.Label(rr2, text="RT format:").pack(side="left")
        self.rtfmt_var = tk.StringVar(value="{title} - {artist}")
        ent_fmt = ttk.Entry(rr2, width=70, textvariable=self.rtfmt_var)
        ent_fmt.pack(side="left", padx=8, fill="x", expand=True)
        ToolTip(ent_fmt, "Template used when manual RT is OFF.\nUse {artist} {title} {album}.\nMax 64 chars.")

        # Watcher
        box_watch = ttk.LabelFrame(self, text="Auto-update from nowplaying.txt", padding=10)
        box_watch.pack(fill="x", **pad)

        w1 = ttk.Frame(box_watch)
        w1.pack(fill="x", pady=6)

        self.np_path_var = tk.StringVar(value="")
        ent_np = ttk.Entry(w1, textvariable=self.np_path_var, width=90)
        ent_np.pack(side="left", padx=(0, 8), fill="x", expand=True)
        btn_browse = ttk.Button(w1, text="Browse...", command=self._browse_nowplaying)
        btn_browse.pack(side="left")

        ToolTip(ent_np, "Path to nowplaying.txt.\nFormats:\nArtist=...; Title=...; Album=...\nOR lines 'Artist:' 'Title:'")
        ToolTip(btn_browse, "Select a text file with now playing info.")

        w2 = ttk.Frame(box_watch)
        w2.pack(fill="x", pady=6)

        self.watch_var = tk.IntVar(value=0)
        cb_watch = ttk.Checkbutton(w2, text="Enable watcher", variable=self.watch_var, command=self._toggle_watch)
        cb_watch.pack(side="left")
        ToolTip(cb_watch, "When enabled, checks the file for changes and updates RT/RT+ automatically.")

        ttk.Label(w2, text="Interval (sec):").pack(side="left", padx=(12, 0))
        self.wint_var = tk.DoubleVar(value=0.6)
        sp_int = ttk.Spinbox(w2, from_=0.2, to=5.0, increment=0.1, width=6, textvariable=self.wint_var)
        sp_int.pack(side="left", padx=8)
        ToolTip(sp_int, "How often to check the file for changes.\n0.5–1.0 sec is good.")

        self.autoapply_var = tk.IntVar(value=1)
        cb_autoapply = ttk.Checkbutton(w2, text="Auto-apply while running", variable=self.autoapply_var)
        cb_autoapply.pack(side="left", padx=(18, 0))
        ToolTip(cb_autoapply, "If ON, file changes update the running encoder immediately.")

        # Controls
        box_ctl = ttk.Frame(self)
        box_ctl.pack(fill="x", **pad)

        self.btn_start = ttk.Button(box_ctl, text="Start", command=self._start)
        self.btn_stop = ttk.Button(box_ctl, text="Stop", command=self._stop, state="disabled")
        self.btn_apply = ttk.Button(box_ctl, text="Apply (live)", command=self._apply_if_live, state="disabled")

        self.btn_start.pack(side="left")
        self.btn_stop.pack(side="left", padx=8)
        self.btn_apply.pack(side="left", padx=8)

        ToolTip(self.btn_start, "Start transmitting the RDS subcarrier.")
        ToolTip(self.btn_stop, "Stop transmitting.")
        ToolTip(self.btn_apply, "Apply changed settings live (while running).")

        self.status_lbl = ttk.Label(self, text="Status: idle")
        self.status_lbl.pack(fill="x", padx=12, pady=(0, 10))

    # -------- Devices --------
    def _load_devices(self):
        devs = sd.query_devices()
        outs = []
        for i, d in enumerate(devs):
            if d.get("max_output_channels", 0) > 0:
                host = sd.query_hostapis(d.get("hostapi", 0)).get("name", "")
                outs.append((i, f"[{i}] {d.get('name','Device')} ({host})"))
        self._devices = outs
        self.dev_combo["values"] = [label for _, label in outs]

    def _load_devices_keep_selection(self):
        old = self.dev_combo.get()
        self._load_devices()
        if old:
            try:
                self.dev_combo.set(old)
            except Exception:
                pass

    # -------- Config <-> UI --------
    def _cfg_to_ui(self):
        self._load_devices()
        if self._devices:
            sel = None
            if self.cfg.device_id is not None:
                for i, (did, _label) in enumerate(self._devices):
                    if did == int(self.cfg.device_id):
                        sel = i
                        break
            if sel is None:
                sel = 0
            self.dev_combo.current(sel)

        self.sr_var.set(int(self.cfg.sample_rate))
        self.subc_var.set(float(self.cfg.subcarrier_hz))
        self.amp_var.set(float(self.cfg.amplitude))
        self.leak_var.set(float(self.cfg.carrier_leak))
        self.swap_var.set(1 if self.cfg.swap_01 else 0)

        self.pi_var.set(f"0x{int(self.cfg.pi) & 0xFFFF:04X}")
        self.ps_var.set(self.cfg.ps)
        self.pty_var.set(int(self.cfg.pty))
        self.tp_var.set(1 if self.cfg.tp else 0)
        self.ta_var.set(1 if self.cfg.ta else 0)
        self.ms_var.set(1 if self.cfg.ms else 0)

        self.psn_var.set(int(self.cfg.ps_every_n_groups))

        self.ecc_en_var.set(1 if self.cfg.ecc_enabled else 0)
        self.ecc_var.set(f"0x{int(self.cfg.ecc) & 0xFF:02X}")
        self.ecc_int_var.set(float(self.cfg.ecc_interval_sec))

        self.af_en_var.set(1 if self.cfg.af_enabled else 0)
        self.af_mhz_var.set(float(self.cfg.af_mhz))

        self.rtfmt_var.set(self.cfg.rt_format)
        self.tr_var.set(1 if self.cfg.translit_ru else 0)
        self.rtplus_var.set(1 if self.cfg.enable_rtplus else 0)
        self.autostart_var.set(1 if self.cfg.autostart else 0)

        self.manual_rt_var.set(1 if self.cfg.manual_rt_enabled else 0)
        self.manual_rt_text_var.set(self.cfg.manual_rt_text or "")

        self.np_path_var.set(self.cfg.nowplaying_path or "")
        self.watch_var.set(1 if self.cfg.watch_enabled else 0)
        self.wint_var.set(float(self.cfg.watch_interval))
        self.autoapply_var.set(1 if self.cfg.auto_apply_live else 0)

        self._on_amp()
        self._on_leak()

        if self.watch_var.get() and self.np_path_var.get().strip():
            self._start_watch()
        else:
            self._stop_watch()

    def _ui_to_cfg(self) -> AppConfig:
        cfg = self.cfg

        device_label = self.dev_combo.get()
        device_id = None
        if device_label.startswith("["):
            try:
                device_id = int(device_label.split("]")[0][1:])
            except Exception:
                device_id = None

        cfg.device_id = device_id
        cfg.device_label = device_label

        cfg.sample_rate = int(self.sr_var.get())
        cfg.subcarrier_hz = float(self.subc_var.get())
        cfg.amplitude = float(self.amp_var.get())
        cfg.carrier_leak = float(self.leak_var.get())
        cfg.swap_01 = bool(self.swap_var.get())

        cfg.pi = int(self.pi_var.get().strip(), 0) & 0xFFFF
        cfg.ps = (self.ps_var.get() or "")[:8].ljust(8)
        cfg.pty = int(self.pty_var.get()) & 0x1F
        cfg.tp = bool(self.tp_var.get())
        cfg.ta = bool(self.ta_var.get())
        cfg.ms = bool(self.ms_var.get())

        cfg.ps_every_n_groups = max(1, int(self.psn_var.get()))

        cfg.ecc_enabled = bool(self.ecc_en_var.get())
        cfg.ecc = int(self.ecc_var.get().strip(), 0) & 0xFF
        cfg.ecc_interval_sec = float(self.ecc_int_var.get())

        cfg.af_enabled = bool(self.af_en_var.get())
        cfg.af_mhz = float(self.af_mhz_var.get())

        cfg.rt_format = self.rtfmt_var.get().strip() or "{title} - {artist}"
        cfg.translit_ru = bool(self.tr_var.get())
        cfg.enable_rtplus = bool(self.rtplus_var.get())
        cfg.autostart = bool(self.autostart_var.get())

        cfg.manual_rt_enabled = bool(self.manual_rt_var.get())
        cfg.manual_rt_text = (self.manual_rt_text_var.get() or "").strip()

        cfg.nowplaying_path = self.np_path_var.get().strip()
        cfg.watch_enabled = bool(self.watch_var.get())
        cfg.watch_interval = float(self.wint_var.get())
        cfg.auto_apply_live = bool(self.autoapply_var.get())

        return cfg

    # -------- Config buttons --------
    def _ui_save_config(self):
        try:
            self.cfg = self._ui_to_cfg()
            save_config(self.cfg_path, self.cfg)
            self.lbl_cfg.config(text=f"Config: {self.cfg_path}")
            self.status_lbl.config(text=f"Status: saved config to {self.cfg_path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _ui_load_config(self):
        path = filedialog.askopenfilename(
            title="Load config JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.cfg_path = os.path.abspath(path)
            self.cfg = load_config(self.cfg_path)
            self.lbl_cfg.config(text=f"Config: {self.cfg_path}")
            self._cfg_to_ui()
            self.status_lbl.config(text=f"Status: loaded config from {self.cfg_path}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _ui_saveas_config(self):
        path = filedialog.asksaveasfilename(
            title="Save config as",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.cfg = self._ui_to_cfg()
            self.cfg_path = os.path.abspath(path)
            save_config(self.cfg_path, self.cfg)
            self.lbl_cfg.config(text=f"Config: {self.cfg_path}")
            self.status_lbl.config(text=f"Status: saved config to {self.cfg_path}")
        except Exception as e:
            messagebox.showerror("Save As failed", str(e))

    # -------- Watcher --------
    def _browse_nowplaying(self):
        path = filedialog.askopenfilename(
            title="Select nowplaying.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        self.np_path_var.set(path)
        self._last_mtime = 0.0
        self._read_nowplaying(force=True)
        if self.watch_var.get():
            self._start_watch()

    def _toggle_watch(self):
        if self.watch_var.get():
            if not self.np_path_var.get().strip():
                messagebox.showwarning("No file", "Select nowplaying.txt first.")
                self.watch_var.set(0)
                return
            self._start_watch()
        else:
            self._stop_watch()

    def _start_watch(self):
        self._stop_watch()
        self._watch_stop.clear()

        def loop():
            while not self._watch_stop.is_set():
                try:
                    interval = float(self.wint_var.get() or 0.6)
                    self._read_nowplaying(force=False)
                    time.sleep(max(0.1, interval))
                except Exception:
                    time.sleep(0.6)

        self._watch_thread = threading.Thread(target=loop, daemon=True)
        self._watch_thread.start()

    def _stop_watch(self):
        self._watch_stop.set()

    def _read_nowplaying(self, force: bool):
        path = self.np_path_var.get().strip()
        if not path or not os.path.exists(path):
            return
        try:
            mtime = os.path.getmtime(path)
            if not force and mtime == self._last_mtime:
                return
            self._last_mtime = mtime

            time.sleep(0.08)  # debounce

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()

            self.meta = parse_nowplaying_text(txt)

            if self.engine.running and self.autoapply_var.get():
                self.engine.update_meta_live(self.meta)

            artist = self.meta.get("artist", "")
            title = self.meta.get("title", "")
            self.status_lbl.config(text=f"Status: nowplaying -> {artist} / {title}")
        except Exception:
            pass

    # -------- Live apply --------
    def _apply_if_live(self):
        self.cfg = self._ui_to_cfg()
        self._on_amp()
        self._on_leak()
        if self.engine.running:
            self.engine.apply_live(self.cfg)
            self.engine.update_meta_live(self.meta)

    def _on_amp(self):
        self.amp_lbl.config(text=f"{float(self.amp_var.get()):.2f}")

    def _on_leak(self):
        self.leak_lbl.config(text=f"{float(self.leak_var.get()):.02f}")

    # -------- Start/Stop --------
    def _start(self):
        if self.engine.running:
            return

        try:
            self.cfg = self._ui_to_cfg()
        except Exception as e:
            messagebox.showerror("Config error", str(e))
            return

        if self.cfg.device_id is None:
            messagebox.showerror("No device", "Select an output device.")
            return

        # For real RDS at 57k, recommend high sample rate
        if float(self.cfg.subcarrier_hz) >= 40_000.0 and int(self.cfg.sample_rate) < 152000:
            messagebox.showerror(
                "Sample rate too low",
                "For 57 kHz RDS, use at least 152000 Hz (recommended 192000)."
            )
            return

        try:
            self.engine.start(self.cfg, self.meta)
        except Exception as e:
            messagebox.showerror("Start failed", str(e))
            return

        if self.watch_var.get() and self.np_path_var.get().strip():
            self._start_watch()

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_apply.config(state="normal")
        self.status_lbl.config(text="Status: RUNNING")

    def _stop(self):
        if not self.engine.running:
            return
        self.status_lbl.config(text="Status: stopping...")
        self.engine.stop()

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_apply.config(state="disabled")
        self.status_lbl.config(text="Status: stopped")

    def _status_loop(self):
        if self.engine.running:
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.btn_apply.config(state="normal")
        else:
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.btn_apply.config(state="disabled")
        self.after(250, self._status_loop)

    def _maybe_autostart(self):
        if self.autostart_var.get() and not self.engine.running:
            try:
                self._start()
            except Exception:
                pass

    def _on_close(self):
        try:
            self.cfg = self._ui_to_cfg()
            save_config(self.cfg_path, self.cfg)
        except Exception:
            pass
        try:
            self._stop_watch()
        except Exception:
            pass
        try:
            self.engine.stop()
        except Exception:
            pass
        self.destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
