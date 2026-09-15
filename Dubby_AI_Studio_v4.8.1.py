#!/usr/bin/env python3
"""
Dubby AI Studio v4.8.0 Turbo Export
==================
A Windows-friendly PyQt6 AI video dubbing/editor foundation with OpenAI/ChatGPT translation support.

Features
--------
- Modern 3-panel editor UI
- Video preview
- Upload up to 20 videos
- Whisper transcription
- Speaker labels + manual/automatic gender assignment
- Translation with GoogleTranslator, OpenAI/ChatGPT, or Gemini
- Edge-TTS dubbing
- Per-speaker voice selection
- Separate subtitle text and TTS text
- Subtitle/title/logo layers
- Live visual preview of title/logo/subtitles
- Timeline with editable subtitle rows
- BGM preservation with improved stereo center reduction + FFmpeg fallback
- Audio ducking while dubbed speech is playing
- TTS caching
- Whisper model caching
- GPU/CPU automatic selection
- Export MP4 + SRT
- Optional Gemini API key settings (translation helper hook)
- Project save/load
- Undo/redo for editor text settings
- Auto aspect ratio presets: Original, 16:9, 9:16, 1:1
- OpenAI/ChatGPT translation support
- Faster export with optimized FFmpeg parameters
- Gemini long-video chunked transcription (5–10 min, batch of 20)
- Improved TTS frame-sync (stretch + rate match to subtitle windows)

Install
-------
Python 3.12 recommended.

pip install PyQt6 numpy pandas Pillow librosa requests pydub faster-whisper deep-translator edge-tts srt imageio-ffmpeg soundfile transformers torch pyannote.audio voxcpm

Speaker diarization requires a Hugging Face token and acceptance of the
pyannote/speaker-diarization-community-1 model terms. Put the token in the
AI / API Settings dialog or set HF_TOKEN in the environment.

VoxCPM2 requires Python >=3.10,<3.13 and PyTorch >=2.5; its official
implementation documents CUDA >=12 and about 8GB VRAM for the 2B model.
On lower-VRAM GPU configurations the app automatically chooses the safer compute path for VoxCPM2.

FFmpeg
-------
Install FFmpeg and make sure "ffmpeg" is available in PATH.
The app also works with imageio-ffmpeg's bundled executable when available.

Run
---
python "E:\\AI Video Translator and App_dubby.py"
"""
import platform
import uuid
from datetime import datetime, timezone, timedelta
import sqlite3
import sys
import os
import re
import json
import math
import time
import asyncio
import shutil
import hashlib
import subprocess
import tempfile
import base64
import wave
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

# ------------------------- Optional imports -------------------------

try:
    import srt
except Exception:
    srt = None

try:
    import requests
except Exception:
    requests = None

try:
    from google import genai as google_genai
except Exception:
    google_genai = None

try:
    from google.genai import types as google_genai_types
except Exception:
    google_genai_types = None

try:
    import librosa
except Exception:
    librosa = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    import edge_tts
except Exception:
    edge_tts = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except Exception:
    PyannotePipeline = None

try:
    from pyannote.audio import Audio as PyannoteAudio
except Exception:
    PyannoteAudio = None

try:
    from voxcpm import VoxCPM
except Exception:
    VoxCPM = None

try:
    from proglog import ProgressBarLogger
except Exception:
    ProgressBarLogger = None

# MoviePy compatibility: support both old and newer package layouts.
try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, ColorClip
    )
except Exception:
    try:
        from moviepy import (
            VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, ColorClip
        )
    except Exception:
        VideoFileClip = AudioFileClip = ImageClip = CompositeVideoClip = ColorClip = None

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QComboBox, QSlider, QCheckBox, QRadioButton,
        QButtonGroup, QLineEdit, QFileDialog, QProgressBar, QTextEdit,
        QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QGroupBox,
        QScrollArea, QMessageBox, QColorDialog, QSpinBox, QDoubleSpinBox,
        QFrame, QSizePolicy, QAbstractItemView, QTabWidget, QListWidget,
        QListWidgetItem, QToolButton, QFormLayout, QDialog, QDialogButtonBox,
        QStatusBar, QMenuBar, QMenu
    )
    from PyQt6.QtCore import QEvent, Qt, QThread, pyqtSignal, QUrl, QTimer, QSize, QPoint
    from PyQt6.QtGui import (
        QFont, QColor, QPixmap, QImage, QPainter, QIcon, QPen, QBrush, QShortcut, QKeySequence, QAction,
        QDesktopServices
    )
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
    from PyQt6.QtMultimediaWidgets import QVideoWidget
except Exception as exc:
    raise SystemExit(
        "PyQt6 is required. Install it with: python -m pip install PyQt6\n"
        f"Import error: {exc}"
    )

# ------------------------- Paths / config -------------------------

APP_NAME = "Dubby AI Studio"
APP_VERSION = "4.8.1"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
CACHE_DIR = BASE_DIR / "cache"
TTS_CACHE_DIR = CACHE_DIR / "tts"
MODEL_CACHE_DIR = CACHE_DIR / "whisper"
DIARIZATION_CACHE_DIR = CACHE_DIR / "diarization"
GENDER_CACHE_DIR = CACHE_DIR / "gender"
VOXCPM_CACHE_DIR = CACHE_DIR / "voxcpm"

FONT_DIR = BASE_DIR / "fonts"
PROJECT_DIR = BASE_DIR / "projects"
# Persistent local config so API keys survive the next day / app restart.
SETTINGS_FILE = BASE_DIR / "user_settings.json"

# Manager update feed (JSON). Change this URL when you host your own version file.
# Example JSON:
# {
#   "version": "4.9.0",
#   "download_url": "https://your-server.com/dubby/Dubby_AI_Studio_v4.9.0.py",
#   "changelog": "- New feature\n- Bug fix",
#   "mandatory": false,
#   "release_notes_url": "https://your-server.com/dubby/changelog"
# }
UPDATE_INFO_URL = (
    "https://raw.githubusercontent.com/YOUR_ORG/dubby-ai-studio/main/update.json"
)
# Set False to disable silent startup checks; users can still use Help → Check for Updates.
AUTO_CHECK_UPDATES_ON_START = True
UPDATE_CHECK_TIMEOUT_SEC = 12

PREVIEW_PROXY_DIR = CACHE_DIR / "preview_proxy"

for d in (OUTPUT_DIR, TEMP_DIR, CACHE_DIR, TTS_CACHE_DIR, MODEL_CACHE_DIR, DIARIZATION_CACHE_DIR, GENDER_CACHE_DIR, VOXCPM_CACHE_DIR, FONT_DIR, PROJECT_DIR, PREVIEW_PROXY_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_VIDEOS = 20

# Keys written to user_settings.json (including secrets).
PERSISTENT_SETTING_KEYS = (
    "gemini_api_key", "gemini_api_keys", "gemini_transcribe_model", "gemini_activity_model",
    "openai_api_key", "openai_model", "translation_prompt", "translation_provider",
    "tts_provider", "voxcpm_device", "voxcpm_model", "voxcpm_style",
    "voxcpm_reference_wav", "voxcpm_prompt_text",
    "huggingface_token", "min_speakers", "max_speakers",
    "gemini_tts_model", "gemini_default_voice", "gemini_tts_style",
    "gemini_request_interval", "gpu_mode", "transcribe_engine",
    "shutdown_when_finished", "output_folder", "export_mode",
)

LANGUAGES = {
    "Khmer": "km",
    "English": "en",
    "Chinese (Simplified)": "zh-CN",
    "Thai": "th",
    "Vietnamese": "vi",
    "Japanese": "ja",
    "Korean": "ko",
    "French": "fr",
    "Spanish": "es",
    "Indonesian": "id",
    "Russian": "ru",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Dutch": "nl",
    "Polish": "pl",
}

VOICES = {
    "Khmer": {
        "Male": ["km-KH-PisethNeural"],
        "Female": ["km-KH-SreymomNeural"],
    },
    "English": {
        "Male": ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-US-EricNeural"],
        "Female": ["en-US-JennyNeural", "en-US-AriaNeural", "en-US-SaraNeural"],
    },
    "Chinese": {
        "Male": ["zh-CN-YunxiNeural", "zh-CN-YunjianNeural"],
        "Female": ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural"],
    },
    "Thai": {
        "Male": ["th-TH-NiwatNeural"],
        "Female": ["th-TH-PremwadeeNeural"],
    },
    "Vietnamese": {
        "Male": ["vi-VN-NamMinhNeural"],
        "Female": ["vi-VN-HoaiMyNeural"],
    },
    "Japanese": {
        "Male": ["ja-JP-KeitaNeural"],
        "Female": ["ja-JP-NanamiNeural"],
    },
    "Korean": {
        "Male": ["ko-KR-InJoonNeural"],
        "Female": ["ko-KR-SunHiNeural"],
    },
    "French": {
        "Male": ["fr-FR-HenriNeural"],
        "Female": ["fr-FR-DeniseNeural"],
    },
    "Spanish": {
        "Male": ["es-ES-AlvaroNeural"],
        "Female": ["es-ES-ElviraNeural"],
    },
    "Indonesian": {
        "Male": ["id-ID-ArdiNeural"],
        "Female": ["id-ID-GadisNeural"],
    },
    "Russian": {
        "Male": ["ru-RU-DmitryNeural"],
        "Female": ["ru-RU-SvetlanaNeural"],
    },
    "German": {
        "Male": ["de-DE-ConradNeural"],
        "Female": ["de-DE-KatjaNeural"],
    },
    "Italian": {
        "Male": ["it-IT-DiegoNeural"],
        "Female": ["it-IT-ElsaNeural"],
    },
    "Portuguese": {
        "Male": ["pt-BR-AntonioNeural"],
        "Female": ["pt-BR-FranciscaNeural"],
    },
    "Arabic": {
        "Male": ["ar-SA-HamedNeural"],
        "Female": ["ar-SA-ZariyahNeural"],
    },
    "Hindi": {
        "Male": ["hi-IN-MadhurNeural"],
        "Female": ["hi-IN-SwaraNeural"],
    },
    "Turkish": {
        "Male": ["tr-TR-AhmetNeural"],
        "Female": ["tr-TR-EmelNeural"],
    },
    "Dutch": {
        "Male": ["nl-NL-MaartenNeural"],
        "Female": ["nl-NL-ColetteNeural"],
    },
    "Polish": {
        "Male": ["pl-PL-MarekNeural"],
        "Female": ["pl-PL-ZofiaNeural"],
    },
}

FONT_CANDIDATES = {
    "Khmer Nida": [
        FONT_DIR / "KhmerNida.ttf",
        FONT_DIR / "KhmerNida-Regular.ttf",
        Path("C:/Windows/Fonts/KhmerNida.ttf"),
        Path("C:/Windows/Fonts/KhmerNida-Regular.ttf"),
    ],
    "Noto Sans Khmer": [
        FONT_DIR / "NotoSansKhmer-Regular.ttf",
        FONT_DIR / "NotoSansKhmerUI-Regular.ttf",
        Path("C:/Windows/Fonts/NotoSansKhmer-Regular.ttf"),
        Path("C:/Windows/Fonts/NotoSansKhmer-Bold.ttf"),
        Path("C:/Windows/Fonts/NotoSansKhmerUI-Regular.ttf"),
        Path("C:/Windows/Fonts/NotoSansKhmerUI-Bold.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansKhmer-Regular.ttf"),
    ],
    "Khmer UI": [Path("C:/Windows/Fonts/KhmerUI.ttf")],
    "Leelawadee": [
        Path("C:/Windows/Fonts/LeelawadeeUI.ttf"),
        Path("C:/Windows/Fonts/Leelawadee.ttf"),
    ],
    "System Default": [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ],
}

# ------------------------- Utility functions -------------------------


def resolve_output_dir(settings=None) -> Path:
    """Return the folder where exported MP4/SRT files should be written.

    Uses the user-chosen output folder when set and writable; otherwise the
    app's default OUTPUT_DIR next to the script.
    """
    settings = settings or {}
    custom = str(settings.get("output_folder") or "").strip()
    if custom:
        try:
            p = Path(custom).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            # Quick write probe so we fail early with a clear message.
            probe = p / ".dubby_write_test"
            try:
                probe.write_text("ok", encoding="utf-8")
                probe.unlink(missing_ok=True)
            except Exception as exc:
                raise RuntimeError(f"Output folder is not writable: {p}\n{exc}")
            return p
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Invalid output folder: {custom}\n{exc}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR

def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    return name.strip() or "untitled"

def ffmpeg_executable() -> str:
    custom = shutil.which("ffmpeg")
    if custom:
        return custom
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

def nvidia_gpu_name() -> str:
    """Return the first NVIDIA GPU name when nvidia-smi is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=5
        )
        if result.returncode == 0:
            return (result.stdout.strip().splitlines() or [""])[0].strip()
    except Exception:
        pass
    return ""

def nvidia_available() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def cuda_safe_for_whisper() -> bool:
    """Return True only when CUDA is present AND safe for faster-whisper.

    Avoids hard process crashes on old legacy NVIDIA architecture cards / broken CUDA drivers by
    probing torch.cuda without loading heavy models. Returns False on any
    doubt so the UI never kills the whole app when the user picks Auto/CUDA.
    """
    if not nvidia_available():
        return False
    if torch is None:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        # Tiny allocation probe — catches driver/toolkit mismatches early.
        _ = torch.zeros(1, device="cuda")
        del _
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        name = nvidia_gpu_name().lower()
        # Some older or low-VRAM GPU configurations can be unstable with CTranslate2; prefer the safer compute path.
        if any(t in name for t in ("gtx 10", "gtx 1050", "gtx 1060", "quadro p", "mx150", "mx250", "gt 7", "gt 9")):
            # Still allow CUDA for Whisper (int8) but caller must use safe compute types.
            return True
        return True
    except Exception:
        return False

def nvenc_available() -> bool:
    """Check that the installed FFmpeg exposes NVIDIA H.264 NVENC."""
    try:
        result = subprocess.run(
            [ffmpeg_executable(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=False, timeout=8
        )
        return result.returncode == 0 and "h264_nvenc" in (result.stdout + result.stderr)
    except Exception:
        return False

def recommended_cuda_compute() -> List[str]:
    """Pick safer CTranslate2 compute types for older/legacy GPU architectures."""
    name = nvidia_gpu_name().lower()
    if any(token in name for token in ("gtx 10", "gtx 1050", "gtx 1060", "gtx 1070", "gtx 1080", "quadro p", "tesla p")):
        return ["int8", "float32", "float16"]
    return ["float16", "int8_float16", "int8", "float32"]

def run_ffmpeg(
    args: List[str],
    quiet: bool = True,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Run FFmpeg without allowing an export job to wait forever.

    Export calls can legitimately take many minutes, so timeout is opt-in.
    When enabled, TimeoutExpired is raised and the worker reports a clean error
    instead of leaving the UI looking permanently stuck.
    """
    cmd = [ffmpeg_executable()] + args
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.DEVNULL if quiet else None,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"FFmpeg timed out after {int(timeout or 0)} seconds. "
            "The export was stopped safely; the source video is unchanged."
        ) from exc

def get_video_info(path: str) -> Tuple[float, int, int]:
    """Return duration, width, height using MoviePy when available."""
    if VideoFileClip is None:
        return 0.0, 0, 0
    clip = None
    try:
        clip = VideoFileClip(path, audio=False)
        return float(clip.duration or 0), int(clip.w), int(clip.h)
    except Exception:
        return 0.0, 0, 0
    finally:
        try:
            if clip:
                clip.close()
        except Exception:
            pass


def ffprobe_duration(path: str) -> float:
    """Reliable media duration via ffprobe (preferred over MoviePy for long/VFR files)."""
    try:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            # imageio-ffmpeg only ships ffmpeg; try sibling name
            exe = ffmpeg_executable()
            candidate = Path(exe).with_name("ffprobe")
            if candidate.exists():
                ffprobe = str(candidate)
            else:
                ffprobe = "ffprobe"
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=False, timeout=20,
        )
        if result.returncode == 0:
            val = float((result.stdout or "").strip().splitlines()[0])
            if val > 0.05:
                return val
    except Exception:
        pass
    # Fallback: ffmpeg -i parse
    try:
        result = subprocess.run(
            [ffmpeg_executable(), "-i", str(path)],
            capture_output=True, text=True, check=False, timeout=20,
        )
        err = result.stderr or ""
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            return h * 3600 + mi * 60 + s
    except Exception:
        pass
    return 0.0

KHMER_FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-fonts/main/"
    "hinted/ttf/NotoSansKhmer/NotoSansKhmer-Regular.ttf"
)

def ensure_khmer_font() -> Optional[Path]:
    """Ensure a real Noto Sans Khmer Unicode font exists locally.

    The app first uses a bundled/system font. If none is available, it makes
    one small download from the official Noto Fonts repository and caches it
    in the app's fonts/ directory. This prevents the common failure where
    the Qt preview shows Khmer correctly but Pillow export uses tofu/boxes.
    """
    candidates = [
        FONT_DIR / "NotoSansKhmer-Regular.ttf",
        Path("C:/Windows/Fonts/NotoSansKhmer-Regular.ttf"),
        Path("C:/Windows/Fonts/NotoSansKhmerUI-Regular.ttf"),
        Path("C:/Windows/Fonts/KhmerUI.ttf"),
        Path("C:/Windows/Fonts/KhmerOS.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 50000:
            return candidate

    target = FONT_DIR / "NotoSansKhmer-Regular.ttf"
    if requests is not None:
        try:
            response = requests.get(KHMER_FONT_URL, timeout=30)
            if response.ok and len(response.content) > 50000:
                target.write_bytes(response.content)
                return target
        except Exception:
            pass
    return None

def get_font(size: int = 48, family: str = "Noto Sans Khmer"):
    """Load a real Unicode/OpenType font for subtitles.

    Khmer must never silently fall back to Pillow's tiny bitmap font because
    that fallback does not contain Khmer glyphs.  We therefore search the
    app fonts folder, Windows Fonts, Linux Noto folders, and Windows' common
    Khmer faces before allowing a non-Khmer fallback for Latin text.
    """
    if ImageFont is None:
        return None
    size = max(10, int(size))

    ordered = list(FONT_CANDIDATES.get(family, []))
    for key, paths in FONT_CANDIDATES.items():
        if key != family:
            ordered.extend(paths)

    ordered.extend([
        FONT_DIR / "NotoSansKhmer-Regular.ttf",
        FONT_DIR / "NotoSansKhmer-Bold.ttf",
        FONT_DIR / "NotoSansKhmerUI-Regular.ttf",
        FONT_DIR / "NotoSansKhmerUI-Bold.ttf",
        FONT_DIR / "KhmerNida.ttf",
        FONT_DIR / "KhmerNida-Regular.ttf",
        Path("C:/Windows/Fonts/KhmerUI.ttf"),
        Path("C:/Windows/Fonts/KhmerOS.ttf"),
        Path("C:/Windows/Fonts/KhmerOSSystem.ttf"),
        Path("C:/Windows/Fonts/NotoSansKhmer-Regular.ttf"),
        Path("C:/Windows/Fonts/NotoSansKhmer-Bold.ttf"),
        Path("C:/Windows/Fonts/NotoSansKhmerUI-Regular.ttf"),
        Path("C:/Windows/Fonts/NotoSansKhmerUI-Bold.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansKhmer-Regular.ttf"),
    ])

    # Windows can register fonts with a different filename. Scan the Fonts
    # directory for Khmer/Noto filenames as a final local fallback.
    win_fonts = Path("C:/Windows/Fonts")
    if win_fonts.exists():
        try:
            for p in sorted(win_fonts.glob("*")):
                name = p.name.lower()
                if p.suffix.lower() in (".ttf", ".otf", ".ttc") and (
                    "khmer" in name or "noto" in name or "leelawadee" in name
                ):
                    ordered.append(p)
        except Exception:
            pass

    seen = set()
    for p in ordered:
        p = Path(p)
        key = str(p).lower()
        if key in seen or not p.exists():
            continue
        seen.add(key)
        try:
            return ImageFont.truetype(str(p), size)
        except Exception:
            continue

    # Last local/online fallback: download the official Noto Sans Khmer font.
    downloaded = ensure_khmer_font()
    if downloaded is not None:
        try:
            return ImageFont.truetype(str(downloaded), size)
        except Exception:
            pass

    # Never return PIL's bitmap font for Khmer. It produces missing boxes in
    # the final MP4 even when the editor preview looks correct.
    return None

def text_image(text: str, width: int, height: int, size: int,
               family: str, color: str = "#FFFFFF",
               stroke: int = 2, align: str = "center") -> Optional[object]:
    """Render subtitle/title text. Optimized for Khmer complex-script fonts.

    Khmer (and similar scripts) need:
    - a real OpenType Khmer face (Nida / Noto / Khmer UI)
    - slightly larger line spacing
    - character-level wrapping when there are no spaces
    - strong outline so glyphs stay readable on busy video
    """
    if Image is None:
        return None
    img = Image.new("RGBA", (max(10, width), max(10, height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Detect Khmer / complex script (no spaces, high Unicode range).
    is_khmer = any("\u1780" <= ch <= "\u17FF" for ch in text)
    font = get_font(size, family)
    if not font:
        if is_khmer:
            raise RuntimeError(
                "No Khmer Unicode font was found. Put NotoSansKhmer-Regular.ttf "
                "or KhmerUI.ttf in the app fonts folder and export again."
            )
        return img
    words = text.split()
    lines, cur = [], ""
    max_w = max(20, width - 40)

    # Khmer does not always have whitespace — wrap by character clusters.
    if (len(words) <= 1 and len(text) > 12) or (is_khmer and " " not in text):
        # Prefer wrapping on spaces when present; otherwise character units.
        units = words if len(words) > 1 else list(text)
    else:
        units = words

    for unit in units:
        sep = "" if (is_khmer and " " not in text) else " "
        test = (cur + sep + unit).strip() if cur else unit
        try:
            bbox = draw.textbbox((0, 0), test, font=font, stroke_width=stroke)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(test) * size * 0.6
        if tw <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = unit
    if cur:
        lines.append(cur)
    if not lines:
        lines = [""]

    # Khmer needs a bit more vertical room for stacking marks.
    line_h = size + (14 if is_khmer else 8)
    total_h = line_h * len(lines)
    y = max(0, (height - total_h) // 2)

    for line in lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(line) * size * 0.55
        if align == "left":
            x = 10
        elif align == "right":
            x = width - tw - 10
        else:
            x = (width - tw) // 2
        # Double-pass stroke for clearer Khmer outlines on video.
        if is_khmer and stroke >= 2:
            draw.text(
                (x, y), line, font=font,
                fill=color, stroke_width=stroke + 1, stroke_fill="#000000"
            )
        draw.text(
            (x, y), line, font=font,
            fill=color, stroke_width=stroke, stroke_fill="#000000"
        )
        y += line_h
    return img


def gaussian_blur_bar(width: int, height: int, opacity: float = 0.65,
                      radius: int = 18, tint=(0, 0, 0)) -> Optional[object]:
    """Premiere-style soft Gaussian-blurred subtitle background bar.

    Creates a semi-transparent rounded-ish dark plate with Gaussian blur so
    subtitle text remains readable without a hard rectangle edge.
    radius controls blur strength (similar to Premiere Gaussian Blur).
    """
    if Image is None:
        return None
    try:
        from PIL import ImageFilter
    except Exception:
        ImageFilter = None
    w = max(8, int(width))
    h = max(8, int(height))
    # Larger canvas so blur does not clip hard at edges.
    pad = max(8, int(radius) * 2)
    canvas = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    alpha = int(max(0, min(255, opacity * 255)))
    color = (int(tint[0]), int(tint[1]), int(tint[2]), alpha)
    # Soft rounded rectangle
    margin = pad // 2
    draw.rounded_rectangle(
        [margin, margin, w + pad * 2 - margin - 1, h + pad * 2 - margin - 1],
        radius=min(h // 2, 24),
        fill=color,
    )
    if ImageFilter is not None and radius > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=max(1, int(radius))))
    # Crop back to content size with a little soft edge kept.
    return canvas

def extract_audio_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """Extract a clean mono PCM WAV for diarization and speaker analysis."""
    result = run_ffmpeg([
        "-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(output_path)
    ])
    if result.returncode != 0 or not Path(output_path).exists():
        raise RuntimeError("FFmpeg could not extract the audio track.")
    return str(output_path)


_DIARIZATION_PIPELINE = None
_GENDER_PIPELINE = None
_VOXCPM_MODEL = None


def _hf_token(settings=None) -> str:
    settings = settings or {}
    return str(settings.get("huggingface_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()


def _safe_torch_device(prefer_gpu=True):
    if prefer_gpu and torch is not None and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") if torch is not None else "cpu"


def load_diarization_pipeline(use_gpu=True, token=""):
    global _DIARIZATION_PIPELINE
    if PyannotePipeline is None:
        raise RuntimeError("pyannote.audio is not installed. Run: python -m pip install pyannote.audio")
    if not token:
        raise RuntimeError("Hugging Face token is required for speaker diarization. Add HF_TOKEN in AI / API Settings and accept the Community-1 model terms.")
    if _DIARIZATION_PIPELINE is None:
        _DIARIZATION_PIPELINE = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=token
        )
    try:
        device = _safe_torch_device(use_gpu)
        if hasattr(_DIARIZATION_PIPELINE, "to"):
            _DIARIZATION_PIPELINE.to(device)
    except Exception:
        # CPU fallback is intentional for low-VRAM GPUs such as legacy low-VRAM GPU.
        try:
            if hasattr(_DIARIZATION_PIPELINE, "to") and torch is not None:
                _DIARIZATION_PIPELINE.to(torch.device("cpu"))
        except Exception:
            pass
    return _DIARIZATION_PIPELINE


def diarize_audio(audio_path: str, use_gpu=True, token="", min_speakers=None, max_speakers=None):
    """Return [(start, end, speaker_label)] using pyannote Community-1.

    Low-VRAM NVIDIA cards get a second chance on CPU if CUDA inference
    fails or runs out of memory.
    """
    kwargs = {}
    if min_speakers:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers:
        kwargs["max_speakers"] = int(max_speakers)
    try:
        pipeline = load_diarization_pipeline(use_gpu=use_gpu, token=token)
        output = pipeline(audio_path, **kwargs)
    except Exception:
        if not use_gpu:
            raise
        pipeline = load_diarization_pipeline(use_gpu=False, token=token)
        output = pipeline(audio_path, **kwargs)
    annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(output, "speaker_diarization", output)
    turns = []
    try:
        iterator = annotation.itertracks(yield_label=True)
    except Exception:
        iterator = []
    for turn, _, speaker in iterator:
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    turns.sort(key=lambda x: x[0])
    return turns


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker(start, end, turns):
    if not turns:
        return "Speaker 1"
    scores = {}
    for t0, t1, speaker in turns:
        ov = _overlap(start, end, t0, t1)
        if ov > 0:
            scores[speaker] = scores.get(speaker, 0.0) + ov
    if scores:
        return max(scores, key=scores.get)
    mid = (start + end) / 2.0
    nearest = min(turns, key=lambda t: abs(((t[0] + t[1]) / 2.0) - mid))
    return nearest[2]


def _canonical_speaker_name(label, mapping):
    if label not in mapping:
        mapping[label] = f"Speaker {len(mapping) + 1}"
    return mapping[label]


def load_gender_classifier(use_gpu=True):
    """Load an audio gender classifier once. It returns probabilities, not certainty."""
    global _GENDER_PIPELINE
    if hf_pipeline is None:
        raise RuntimeError("transformers is not installed. Run: python -m pip install transformers torch")
    if _GENDER_PIPELINE is None:
        # Prefer CPU for stability on low-VRAM / broken CUDA hosts.
        safe = bool(use_gpu) and cuda_safe_for_whisper()
        device = 0 if safe else -1
        try:
            _GENDER_PIPELINE = hf_pipeline(
                "audio-classification",
                model="7wolf/wav2vec2-base-gender-classification",
                device=device,
            )
        except Exception:
            if device == 0:
                _GENDER_PIPELINE = hf_pipeline(
                    "audio-classification",
                    model="7wolf/wav2vec2-base-gender-classification",
                    device=-1,
                )
            else:
                raise
    return _GENDER_PIPELINE


def _gender_from_predictions(predictions, pipe=None):
    male = female = 0.0

    # Some Transformers versions/models return LABEL_0/LABEL_1. Resolve
    # those labels from the model config instead of treating them as unknown.
    id2label = {}
    try:
        id2label = dict(getattr(getattr(pipe, "model", None), "config", None).id2label or {})
    except Exception:
        id2label = {}

    for item in predictions or []:
        raw_label = str(item.get("label", "")).strip()
        label = raw_label.lower()
        if label.startswith("label_"):
            try:
                idx = int(label.split("_", 1)[1])
                label = str(id2label.get(idx, id2label.get(str(idx), raw_label))).lower()
            except Exception:
                pass
        score = float(item.get("score", 0.0) or 0.0)
        if any(x in label for x in ("female", "woman", "girl", "жен", "femme")) or label in ("f",):
            female += score
        elif any(x in label for x in ("male", "man", "boy", "муж", "homme")) or label in ("m",):
            male += score
    total = male + female
    if total <= 0:
        return "Unknown", 0.0, 0.0, 0.0
    male /= total
    female /= total
    if abs(male - female) < 0.12:
        return "Unknown", max(male, female), male, female
    gender = "Male" if male > female else "Female"
    return gender, max(male, female), male, female


def _pitch_gender_hint(y, sr):
    """Estimate gender from median fundamental frequency (F0).

    Typical adult male F0 ≈ 85–160 Hz, adult female ≈ 165–255 Hz.
    Asian female speech often sits around 180–220 Hz; male around 100–140.
    Overlap zone 150–175 is treated as uncertain.
    Returns (gender, confidence) or ("Unknown", 0).
    """
    if librosa is None or y is None or len(y) < int(0.4 * sr):
        return "Unknown", 0.0
    try:
        f0 = librosa.yin(
            y.astype(np.float32),
            fmin=65,
            fmax=350,
            sr=sr,
            frame_length=2048,
        )
        f0 = f0[np.isfinite(f0) & (f0 > 65) & (f0 < 350)]
        if len(f0) < 6:
            return "Unknown", 0.0
        med = float(np.median(f0))
        # Clear bands — tightened for better female recall.
        if med < 150:
            conf = min(0.97, 0.60 + (150 - med) / 100.0)
            return "Male", conf
        if med >= 175:
            conf = min(0.97, 0.60 + (med - 175) / 90.0)
            return "Female", conf
        # Ambiguous mid band — slight lean by distance.
        if med < 162:
            return "Male", 0.42
        return "Female", 0.42
    except Exception:
        return "Unknown", 0.0


def detect_gender_for_speaker(audio_path: str, turns, speaker_label: str, use_gpu=True):
    """Aggregate several clean speech windows for one diarized speaker.

    Combines wav2vec2 classifier probabilities with F0 pitch analysis so
    results track the real speaker more reliably than either method alone.

    Accuracy improvements vs earlier builds:
    - Longer / more speech windows (up to 20)
    - Energy-based rejection of near-silent frames
    - Weighted median of classifier scores
    - Stronger pitch vote when F0 is decisive
    - Prefer a Male/Female decision whenever confidence >= 0.52
    """
    if librosa is None:
        return {"gender": "Unknown", "confidence": 0.0, "male": 0.0, "female": 0.0}
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        windows = []
        for t0, t1, spk in turns:
            if spk != speaker_label:
                continue
            dur = t1 - t0
            if dur < 0.35:
                continue
            # Prefer the middle of each turn to avoid boundary crosstalk.
            w0 = t0 + min(0.15, dur * 0.10)
            w1 = t1 - min(0.15, dur * 0.10)
            if w1 - w0 < 0.35:
                continue
            a = max(0, int(w0 * sr))
            b = min(len(y), int(w1 * sr))
            clip = y[a:b]
            if len(clip) < int(0.35 * sr):
                continue
            # Reject near-silent windows (music beds / pauses).
            rms = float(np.sqrt(np.mean(np.square(clip.astype(np.float32)))))
            if rms < 0.008:
                continue
            windows.append(clip[: int(6.0 * sr)])
            if len(windows) >= 20:
                break
        if not windows:
            if len(y) >= int(0.8 * sr):
                windows = [y[: int(min(len(y), 8.0 * sr))]]
            else:
                return {"gender": "Unknown", "confidence": 0.0, "male": 0.0, "female": 0.0}

        male_scores, female_scores = [], []
        try:
            # Gender classifier prefers CPU on unstable CUDA setups.
            safe_gpu = bool(use_gpu) and cuda_safe_for_whisper()
            pipe = load_gender_classifier(use_gpu=safe_gpu)
            for clip in windows:
                try:
                    pred = pipe(
                        {"raw": clip.astype(np.float32), "sampling_rate": sr},
                        top_k=5,
                    )
                    g, conf, male, female = _gender_from_predictions(pred, pipe)
                    male_scores.append(male)
                    female_scores.append(female)
                except Exception:
                    continue
        except Exception:
            pass

        # Pitch (F0) vote across the same windows.
        pitch_male = pitch_female = 0
        pitch_confs = []
        for clip in windows:
            pg, pc = _pitch_gender_hint(clip, sr)
            if pg == "Male":
                pitch_male += 1
                pitch_confs.append(pc)
            elif pg == "Female":
                pitch_female += 1
                pitch_confs.append(pc)

        if male_scores:
            male = float(np.median(male_scores))
            female = float(np.median(female_scores))
        else:
            male = female = 0.0

        # Blend classifier + pitch when both are available.
        pitch_total = pitch_male + pitch_female
        if pitch_total > 0:
            pitch_male_p = pitch_male / pitch_total
            pitch_female_p = pitch_female / pitch_total
            # Stronger pitch weight when many windows agree and F0 is clear.
            avg_pc = float(np.mean(pitch_confs)) if pitch_confs else 0.4
            pitch_w = 0.55 if (not male_scores) else min(0.50, 0.30 + 0.25 * avg_pc)
            cls_w = 1.0 - pitch_w
            male = cls_w * male + pitch_w * pitch_male_p
            female = cls_w * female + pitch_w * pitch_female_p

        total = male + female
        if total <= 0:
            return {"gender": "Unknown", "confidence": 0.0, "male": 0.0, "female": 0.0}
        male /= total
        female /= total
        confidence = max(male, female)
        gender = "Male" if male > female else "Female"

        # Use the aggregated pitch vote as a tie-breaker instead of returning
        # Unknown for most ordinary voices. Keep genuinely ambiguous voices
        # as Auto so the user can review them.
        if abs(male - female) < 0.055:
            if pitch_total > 0 and pitch_male != pitch_female:
                gender = "Male" if pitch_male > pitch_female else "Female"
            elif confidence < 0.54:
                gender = "Unknown"
        return {
            "gender": gender,
            "confidence": float(confidence),
            "male": float(male),
            "female": float(female),
        }
    except Exception:
        return {"gender": "Unknown", "confidence": 0.0, "male": 0.0, "female": 0.0}


def detect_gender_from_audio(audio_path: str) -> str:
    """Backward-compatible wrapper; uses the ML classifier instead of pitch thresholds."""
    try:
        result = detect_gender_for_speaker(audio_path, [(0, 999999, "Speaker 1")], "Speaker 1", use_gpu=False)
        return result.get("gender", "Unknown")
    except Exception:
        return "Unknown"

def audio_duration_ms(path: str) -> int:
    if AudioSegment is None:
        return 0
    try:
        return len(AudioSegment.from_file(path))
    except Exception:
        return 0

def trim_tts_silence(audio_seg, keep_ms: int = 45):
    """Remove excessive leading/trailing silence without touching the voice."""
    if AudioSegment is None or not audio_seg or len(audio_seg) < 80:
        return audio_seg
    try:
        # Conservative threshold: only remove clear digital/room silence.
        ranges = audio_seg.detect_nonsilent(min_silence_len=35, silence_thresh=-48)
        if not ranges:
            return audio_seg
        start = max(0, ranges[0][0] - keep_ms)
        end = min(len(audio_seg), ranges[-1][1] + keep_ms)
        return audio_seg[start:end]
    except Exception:
        return audio_seg


def natural_time_stretch(audio_seg, target_ms: int, max_change: float = 0.05):
    """Legacy helper — prefer fit_tts_to_slot() for export sync."""
    return fit_tts_to_slot(audio_seg, target_ms, max_stretch=max_change, allow_trim=False)


def _librosa_time_stretch_segment(audio_seg, rate: float):
    """Time-stretch a pydub AudioSegment with librosa. rate>1 shortens."""
    if AudioSegment is None or audio_seg is None or librosa is None:
        return audio_seg
    rate = float(rate)
    if abs(rate - 1.0) < 0.01:
        return audio_seg
    try:
        samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
        scale = float(1 << (8 * audio_seg.sample_width - 1))
        if audio_seg.channels == 2:
            samples = samples.reshape((-1, 2)).T / scale
            stretched = [librosa.effects.time_stretch(ch, rate=rate) for ch in samples]
            y = np.asarray(stretched).T.reshape(-1)
        else:
            y = librosa.effects.time_stretch(samples / scale, rate=rate)
        y = np.clip(y * scale, -scale, scale - 1).astype(np.int16)
        return audio_seg._spawn(
            y.tobytes(),
            overrides={"frame_rate": audio_seg.frame_rate, "sample_width": 2},
        )
    except Exception:
        return audio_seg


def _fit_audio_precisely_to_slot(audio_seg, target_ms: int, max_speed: float = 2.0):
    """Fit speech to the subtitle window with pitch-preserving FFmpeg atempo."""
    if AudioSegment is None or audio_seg is None:
        return audio_seg
    target_ms=max(120,int(target_ms)); audio_seg=trim_tts_silence(audio_seg, keep_ms=18)
    cur=len(audio_seg)
    if cur<=0: return audio_seg
    if cur<=target_ms:
        if cur<target_ms: audio_seg += AudioSegment.silent(duration=target_ms-cur, frame_rate=audio_seg.frame_rate)
        return audio_seg[:target_ms]
    ratio=cur/float(target_ms); speed=min(max_speed,max(1.01,ratio))
    try:
        filters=[]; remaining=speed
        while remaining>2.0:
            filters.append("atempo=2.0"); remaining/=2.0
        filters.append(f"atempo={remaining:.6f}")
        src=Path(tempfile.mktemp(suffix='.wav')); out=Path(tempfile.mktemp(suffix='.wav'))
        try:
            audio_seg.export(str(src),format='wav')
            r=run_ffmpeg(['-y','-i',str(src),'-af',','.join(filters),'-ar','48000','-ac','2','-c:a','pcm_s16le',str(out)])
            if r.returncode==0 and out.exists() and out.stat().st_size>1000:
                fitted=AudioSegment.from_file(str(out),format='wav')
                fitted=fitted[:target_ms]
                if len(fitted)<target_ms: fitted += AudioSegment.silent(duration=target_ms-len(fitted),frame_rate=fitted.frame_rate)
                return fitted
        finally:
            for q in (src,out):
                try:q.unlink()
                except Exception:pass
    except Exception: pass
    return audio_seg[:target_ms]


def fit_tts_to_slot(audio_seg, target_ms: int, max_stretch: float = 0.80,
                    allow_trim: bool = True, pad_short: bool = True):
    """Frame-lock speech to the original subtitle/activity window."""
    if AudioSegment is None or not audio_seg or target_ms<=0: return audio_seg
    target_ms=max(120,int(target_ms)); audio_seg=trim_tts_silence(audio_seg,keep_ms=18)
    if len(audio_seg)>target_ms: audio_seg=_fit_audio_precisely_to_slot(audio_seg,target_ms,2.0)
    elif pad_short and len(audio_seg)<target_ms-35: audio_seg += AudioSegment.silent(duration=target_ms-len(audio_seg),frame_rate=audio_seg.frame_rate)
    return audio_seg[:target_ms] if allow_trim else audio_seg


def improve_audio_segment(seg):
    """Clean and clarify TTS while preserving a natural voice.

    Processing is intentionally gentle:
    - remove leading/trailing digital silence
    - convert to stable stereo PCM
    - high-pass very low rumble
    - low-pass harsh ultrasonic/high-frequency noise
    - gentle compression for consistent speech level
    - safe normalization with headroom
    - tiny fades to avoid clicks at segment boundaries

    This is applied to each TTS segment BEFORE it is placed on the master
    timeline, so the BGM is not compressed or filtered by the voice cleanup.
    """
    if AudioSegment is None or seg is None:
        return seg
    try:
        seg = trim_tts_silence(seg, keep_ms=25)

        # Stable format for all voices. 48 kHz is used by the final export.
        if seg.frame_rate != 48000:
            seg = seg.set_frame_rate(48000)
        if seg.channels == 1:
            seg = seg.set_channels(2)
        elif seg.channels > 2:
            seg = seg.set_channels(2)
        if seg.sample_width != 2:
            seg = seg.set_sample_width(2)

        # Gentle voice cleanup. Do not use aggressive denoising on TTS:
        # it can create metallic/robotic artifacts.
        try:
            seg = seg.high_pass_filter(70)
            seg = seg.low_pass_filter(12000)
        except Exception:
            pass

        # Do NOT heavily compress TTS. Strong compression can create the
        # metallic/robotic character that is especially noticeable on short
        # activity narration clips. Keep dynamics mostly intact.
        try:
            seg = seg.compress_dynamic_range(
                threshold=-20.0,
                ratio=1.15,
                attack=12,
                release=180,
            )
        except Exception:
            pass

        # Normalize safely; keep headroom so AAC encoding will not clip.
        try:
            seg = seg.normalize(headroom=1.5)
        except Exception:
            pass

        # Tiny fades prevent hard clicks when many segments are overlaid.
        fade_in = min(4, max(1, len(seg) // 100))
        fade_out = min(18, max(4, len(seg) // 45))
        if len(seg) > fade_in + fade_out + 20:
            seg = seg.fade_in(fade_in).fade_out(fade_out)

        return seg
    except Exception:
        return seg

def extract_bgm_only(input_audio_path: str, output_path: str,
                     strength: str = "strong") -> bool:
    """
    Improved non-AI fallback:
    - stereo center/side reduction for common centered dialogue
    - gentle EQ
    - FFmpeg fallback
    strength "isolate" zeros the mid (dialogue) channel aggressively for
    cleaner BGM-only beds (Premiere-style Isolate).
    For highest-quality music separation, replace this function with a
    dedicated Demucs/UVR model in a future backend.
    """
    if AudioSegment is None:
        return False
    try:
        audio = AudioSegment.from_file(input_audio_path)
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape(-1, 2).astype(np.float32)
            left, right = samples[:, 0], samples[:, 1]
            center = (left + right) / 2.0
            side = (left - right) / 2.0

            center_gain = {
                "light": 0.45,
                "medium": 0.25,
                "strong": 0.10,
                "isolate": 0.0,   # pure side-channel = isolate BGM
            }.get(strength, 0.10)
            new_left = side + center * center_gain
            new_right = -side + center * center_gain

            peak = max(float(np.max(np.abs(new_left))),
                       float(np.max(np.abs(new_right))), 1.0)
            scale = min(1.0, 32767.0 / peak)
            new_left *= scale
            new_right *= scale

            stereo = np.column_stack([
                new_left.astype(np.int16),
                new_right.astype(np.int16)
            ])
            bgm = audio._spawn(
                stereo.tobytes(),
                overrides={"channels": 2, "sample_width": 2}
            )
            bgm = bgm.high_pass_filter(45).low_pass_filter(15000)
            if strength in ("strong", "isolate"):
                bgm = bgm.low_pass_filter(9000)
            if strength == "isolate":
                # Extra dialogue band cut for cleaner isolation
                bgm = bgm.low_pass_filter(7500)
            bgm.export(output_path, format="wav")
            return Path(output_path).exists()
    except Exception:
        pass

    filters = {
        "isolate": "highpass=f=50,lowpass=f=8000,equalizer=f=200:t=q:w=1.4:g=-18,equalizer=f=400:t=q:w=1.3:g=-20,equalizer=f=800:t=q:w=1.2:g=-18,equalizer=f=1500:t=q:w=1.1:g=-14,equalizer=f=2500:t=q:w=1:g=-10",
        "strong": "highpass=f=45,lowpass=f=10000,equalizer=f=250:t=q:w=1.3:g=-12,equalizer=f=500:t=q:w=1.2:g=-14,equalizer=f=1000:t=q:w=1.1:g=-12,equalizer=f=2000:t=q:w=1:g=-9",
        "medium": "highpass=f=45,lowpass=f=12000,equalizer=f=300:t=q:w=1.2:g=-9,equalizer=f=700:t=q:w=1.1:g=-10,equalizer=f=1500:t=q:w=1:g=-7",
        "light": "highpass=f=40,lowpass=f=14000,equalizer=f=500:t=q:w=1.2:g=-5,equalizer=f=1200:t=q:w=1:g=-5",
    }
    result = run_ffmpeg([
        "-y", "-i", input_audio_path, "-af", filters.get(strength, filters["strong"]),
        "-ac", "2", output_path
    ])
    return result.returncode == 0 and Path(output_path).exists()



def preview_proxy_path(source_path: str) -> Path:
    """Stable cache path for a lower-resolution preview proxy of a source video."""
    src = Path(source_path)
    try:
        stamp = int(src.stat().st_mtime)
        size = int(src.stat().st_size)
    except Exception:
        stamp, size = 0, 0
    key = hashlib.sha256(f"{src.resolve()}|{stamp}|{size}".encode("utf-8", errors="ignore")).hexdigest()[:20]
    return PREVIEW_PROXY_DIR / f"proxy_{key}.mp4"


def ensure_preview_proxy(source_path: str, max_width: int = 1280, progress_cb=None) -> str:
    """Build (or reuse) an H.264 720p/1280w proxy so QMediaPlayer can preview 2K/4K reliably.

    High-resolution H.265 / 10-bit / VFR sources often fail or stutter in Qt's
    Windows Media Foundation backend. A lightweight H.264 proxy keeps the editor
    preview responsive while export still uses the original master file.
    """
    src = Path(source_path)
    if not src.exists():
        return str(src)
    proxy = preview_proxy_path(str(src))
    if proxy.exists() and proxy.stat().st_size > 50_000:
        return str(proxy)

    # Probe resolution; skip proxy when already modest.
    w = h = 0
    try:
        _, w, h = get_video_info(str(src))
    except Exception:
        w = h = 0
    if 0 < w <= max_width and 0 < h <= 1080:
        return str(src)

    if progress_cb:
        progress_cb(f"Building preview proxy for {src.name} ({w}x{h})…")

    # Fast H.264 proxy: scale longest side to max_width, 30 fps cap, stereo AAC.
    # ultrafast + CRF 28 is intentional — preview only, not export quality.
    scale = f"scale='min({max_width},iw)':-2:flags=fast_bilinear"
    args = [
        "-y", "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", scale,
        "-r", "30",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-movflags", "+faststart",
        str(proxy),
    ]
    result = run_ffmpeg(args, quiet=True)
    if result.returncode == 0 and proxy.exists() and proxy.stat().st_size > 50_000:
        return str(proxy)
    # Fallback: try without audio map quirks
    try:
        if proxy.exists():
            proxy.unlink()
    except Exception:
        pass
    args2 = [
        "-y", "-i", str(src),
        "-vf", scale,
        "-r", "30",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        str(proxy),
    ]
    result2 = run_ffmpeg(args2, quiet=True)
    if result2.returncode == 0 and proxy.exists() and proxy.stat().st_size > 50_000:
        return str(proxy)
    return str(src)


def hash_tts(text: str, voice: str, rate: str = "+0%", provider: str = "edge", model: str = "", pitch: str = "+0Hz", volume: str = "+0%") -> str:
    return hashlib.sha256(
        f"{provider}|{model}|{voice}|{rate}|{pitch}|{volume}|{text}".encode("utf-8")
    ).hexdigest()


GEMINI_TTS_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus",
    "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel",
    "Algieba", "Despina", "Erinome", "Laomedeia", "Achird",
    "Zubenelgenubi", "Sadachbia", "Sadaltager", "Sulafat",
]

def split_api_keys(value) -> List[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split(r"[\r\n,;]+", str(value or ""))
    out = []
    seen = set()
    for item in raw:
        key = str(item).strip()
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def load_user_settings() -> dict:
    """Load API / AI settings from disk so keys survive the next day."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_user_settings(settings: dict) -> None:
    """Persist API keys and AI preferences locally (UTF-8 JSON beside the app)."""
    try:
        payload = {}
        for key in PERSISTENT_SETTING_KEYS:
            if key in settings:
                payload[key] = settings[key]
        keys = split_api_keys(payload.get("gemini_api_keys") or payload.get("gemini_api_key") or "")
        payload["gemini_api_keys"] = keys
        payload["gemini_api_key"] = keys[0] if keys else ""
        SETTINGS_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[Dubby] Could not save user_settings.json: {exc}")


def gemini_tts_cache_path(text: str, voice: str, speed: float, model: str,
                           style_prompt: str = "", target_ms: int = 0) -> Path:
    # Include the quality profile and approximate target duration so an older,
    # more aggressive voice file is never reused after changing the voice mode.
    profile = hashlib.sha256(str(style_prompt or "").encode("utf-8")).hexdigest()[:12]
    duration_bucket = int(round(max(0, int(target_ms or 0)) / 250.0) * 250)
    cache_key = hash_tts(
        text, voice,
        f"{float(speed):.3f}|dur={duration_bucket}|profile={profile}",
        "gemini", model
    )
    return TTS_CACHE_DIR / f"{cache_key}.wav"

def _write_pcm_wav(path: Path, pcm: bytes, sample_rate: int = 24000) -> None:
    if not pcm or len(pcm) < 1000:
        raise RuntimeError("Google Gemini TTS returned empty audio.")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)

def _gemini_error_text(response) -> str:
    """Return a compact, useful Gemini API error message."""
    try:
        data = response.json()
        err = data.get("error") or {}
        msg = err.get("message") or err.get("status") or ""
        if msg:
            return str(msg).replace("\n", " ")[:280]
    except Exception:
        pass
    try:
        text = response.text.strip().replace("\n", " ")
        if text:
            return text[:280]
    except Exception:
        pass
    return f"HTTP {getattr(response, 'status_code', '?')}"


def _retry_after_seconds(response, default=8.0) -> float:
    """Read Retry-After when Google supplies it, otherwise use a safe delay."""
    try:
        value = response.headers.get("Retry-After")
        if value:
            return max(1.0, min(90.0, float(value)))
    except Exception:
        pass
    return float(default)


def _pace_instruction(speed: float) -> str:
    speed = float(speed or 1.0)
    if speed <= 0.88:
        return "Speak slowly and clearly, with comfortable pauses."
    if speed <= 0.96:
        return "Speak slightly slower than normal, with natural pauses."
    if speed >= 1.12:
        return "Speak slightly faster than normal, while remaining clear and natural."
    if speed >= 1.05:
        return "Speak at a moderately brisk but natural pace."
    return "Speak at a natural conversational pace."


def save_gemini_tts_with_retry(
    text: str,
    voice: str,
    target: Path,
    api_keys,
    model: str = "gemini-2.5-flash-preview-tts",
    style_prompt: str = "Natural, clear, warm narration. Speak the text exactly as written.",
    speed: float = 1.0,
    target_duration_sec: float = 0.0,
    retries_per_key: int = 2,
    min_request_interval: float = 0.65,
    key_state: Optional[dict] = None,
) -> str:
    """Generate Gemini TTS using the CURRENT Google API schema.

    Gemini 3.1 Flash TTS uses the Interactions API with response_format={"type":"audio"}.
    Gemini 2.5 Flash/Pro TTS use generateContent. Both return 24-kHz mono PCM.
    """
    if requests is None:
        raise RuntimeError("requests is not installed. Run: python -m pip install requests")

    keys = split_api_keys(api_keys)
    if not keys:
        raise RuntimeError("No Gemini API key is configured. Open AI / API Settings.")

    model = str(model or "gemini-3.1-flash-tts-preview").strip()
    if not model:
        model = "gemini-3.1-flash-tts-preview"

    if key_state is None:
        key_state = {"next": 0, "cooldown_until": [0.0] * len(keys), "last_request": 0.0}
    if len(key_state.get("cooldown_until", [])) != len(keys):
        key_state["cooldown_until"] = [0.0] * len(keys)
    key_state.setdefault("next", 0)
    key_state.setdefault("last_request", 0.0)

    clean_style = (style_prompt or "").strip() or (
        "Professional human voice-over. Clear, warm, natural conversational narration. "
        "Natural breathing and pauses. Do not sound robotic, theatrical, breathy, or synthetic."
    )
    speed = max(0.70, min(1.55, float(speed or 1.0)))
    pace = _pace_instruction(speed)
    duration_note = ""
    if float(target_duration_sec or 0) >= 0.75:
        duration_note = (
            f" The spoken line should naturally fit about {float(target_duration_sec):.2f} seconds."
        )

    # Google recommends a clear TTS preamble and explicit transcript boundary.
    prompt = (
        "SPEECH SYNTHESIS TASK. Output audio only.\n"
        f"{clean_style}\n"
        f"{pace}{duration_note}\n"
        "Speak ONLY the text between BEGIN TRANSCRIPT and END TRANSCRIPT. "
        "Do not speak the instructions or labels. Preserve every word exactly. "
        "Use crisp pronunciation, clean consonants, natural breaths, and short "
        "pauses only where punctuation indicates them. Do not add dramatic acting.\n\n"
        "BEGIN TRANSCRIPT\n"
        f"{text}\n"
        "END TRANSCRIPT"
    )

    while tried < attempts:
        now = time.monotonic()
        cooldowns = key_state["cooldown_until"]
        available = [i for i, until in enumerate(cooldowns) if until <= now]
        if not available:
            time.sleep(min(90.0, max(0.1, min(cooldowns) - now)))
            continue

        start_index = int(key_state.get("next", 0)) % len(keys)
        key_index = next(
            ((start_index + off) % len(keys)
             for off in range(len(keys))
             if ((start_index + off) % len(keys)) in available),
            None,
        )
        if key_index is None:
            continue
        key_state["next"] = (key_index + 1) % len(keys)

        gap = float(min_request_interval or 0.0)
        since_last = time.monotonic() - float(key_state.get("last_request", 0.0))
        if gap > since_last:
            time.sleep(gap - since_last)
        key_state["last_request"] = time.monotonic()
        tried += 1
        key_label = f"key {key_index + 1}/{len(keys)}"

        try:
            if is_31:
                # Modern Interactions API (required by the May 2026 schema).
                url = "https://generativelanguage.googleapis.com/v1beta/interactions"
                payload = {
                    "model": model,
                    "input": prompt,
                    "response_format": {"type": "audio"},
                    "generation_config": {
                        "speech_config": [{"voice": voice}]
                    },
                }
                headers = {
                    "x-goog-api-key": keys[key_index],
                    "Content-Type": "application/json",
                    "Api-Revision": "2026-05-20",
                }
            else:
                # Gemini 2.5 TTS uses the GenerateContent endpoint.
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {"voiceName": voice}
                            }
                        },
                    },
                }
                headers = {
                    "x-goog-api-key": keys[key_index],
                    "Content-Type": "application/json",
                }

            response = requests.post(
                url, headers=headers, json=payload, timeout=120
            )
            status = int(response.status_code)

            if status == 429:
                delay = _retry_after_seconds(response, default=8.0)
                cooldowns[key_index] = time.monotonic() + delay
                errors.append(f"{key_label}: HTTP 429; waiting {delay:.0f}s")
                continue
            if status in (408, 409, 425) or status >= 500:
                delay = min(30.0, 2.0 * (1 + tried // max(1, len(keys))))
                cooldowns[key_index] = time.monotonic() + delay
                errors.append(f"{key_label}: HTTP {status}: {_gemini_error_text(response)}")
                continue
            if status in (401, 403):
                cooldowns[key_index] = time.monotonic() + 300.0
                errors.append(f"{key_label}: HTTP {status}: {_gemini_error_text(response)}")
                continue
            if status >= 400:
                errors.append(f"{key_label}: HTTP {status}: {_gemini_error_text(response)}")
                continue

            data = response.json()
            audio_b64 = None

            if is_31:
                # Interactions response: output_audio.data
                out_audio = data.get("output_audio") or {}
                audio_b64 = out_audio.get("data")
                if not audio_b64:
                    # Be tolerant of SDK/REST response variations.
                    for item in data.get("output", []) or []:
                        if not isinstance(item, dict):
                            continue
                        oa = item.get("audio") or item.get("output_audio") or {}
                        if isinstance(oa, dict) and oa.get("data"):
                            audio_b64 = oa["data"]
                            break
            else:
                candidates = data.get("candidates") or []
                parts = (
                    ((candidates[0].get("content") or {}).get("parts")
                     if candidates else None) or []
                )
                for part in parts:
                    inline = part.get("inlineData") or part.get("inline_data") or {}
                    if inline.get("data"):
                        audio_b64 = inline["data"]
                        break

            if not audio_b64:
                raise RuntimeError(
                    "Gemini returned no audio data. "
                    f"Response: {json.dumps(data, ensure_ascii=False)[:500]}"
                )

            try:
                pcm = base64.b64decode(audio_b64, validate=True)
            except Exception as exc:
                raise RuntimeError(f"Invalid Gemini audio data: {exc}")

            if len(pcm) < 1000:
                raise RuntimeError(f"Gemini returned too little audio ({len(pcm)} bytes).")

            tmp = target.with_suffix(target.suffix + ".tmp")
            _write_pcm_wav(tmp, pcm, sample_rate=24000)
            if tmp.stat().st_size < 1200:
                try:
                    tmp.unlink()
                except Exception:
                    pass
                raise RuntimeError("Gemini returned an empty/too-small WAV.")
            tmp.replace(target)
            return f"{key_label} OK"

        except Exception as exc:
            errors.append(f"{key_label}: {exc}")
            cooldowns[key_index] = time.monotonic() + min(
                15.0, 1.5 * (1 + tried // max(1, len(keys)))
            )

    detail = " | ".join(errors[-10:])
    raise RuntimeError(
        f"Google Gemini TTS failed after {tried} attempts. {detail}"
    )

def is_server_error_text(text: str) -> bool:
    """Return True when a translation/API response looks like an error."""
    if not text:
        return True
    normalized = re.sub(r"\s+", " ", str(text)).strip().lower()
    patterns = (
        "server error 500",
        "internal server error",
        "http 500",
        "status code 500",
        "error 500",
        "service unavailable",
        "temporarily unavailable",
        "too many requests",
        "rate limit",
        "429",
        "502 bad gateway",
        "503 service unavailable",
    )
    return any(p in normalized for p in patterns)


async def save_tts_with_retry(text: str, voice: str, rate: str,
                              target: Path, retries: int = 5,
                              pitch: str = "+0Hz", volume: str = "+0%") -> None:
    """
    Generate Edge-TTS audio with retry/backoff.

    A transient HTTP 500/429 or websocket/network failure should not
    poison the whole dubbing job. The partial cache file is removed before
    each retry so a corrupt MP3 is never reused.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            # Never allow a partial/corrupt cache file to be mistaken for
            # successful TTS output.
            if target.exists():
                try:
                    if target.stat().st_size < 1024:
                        target.unlink()
                except Exception:
                    pass

            communicate = edge_tts.Communicate(
                text,
                voice,
                rate=rate,
                pitch=pitch,
                volume=volume,
            )
            await communicate.save(str(target))

            if not target.exists() or target.stat().st_size < 1024:
                raise RuntimeError("Edge-TTS returned an empty audio file.")

            return

        except Exception as exc:
            last_error = exc

            try:
                if target.exists():
                    target.unlink()
            except Exception:
                pass

            if attempt < retries:
                # Keep retries responsive; normal successful requests never wait.
                delay = min(8.0, 0.8 * (2 ** (attempt - 1)))
                await asyncio.sleep(delay)

    raise RuntimeError(
        f"Edge-TTS failed after {retries} attempts: {last_error}"
    )

def tts_cache_path(text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz", volume: str = "+0%") -> Path:
    return TTS_CACHE_DIR / f"{hash_tts(text, voice, rate, 'edge', '', pitch, volume)}.mp3"

def language_family(display_name: str) -> str:
    if display_name.startswith("Chinese"):
        return "Chinese"
    return display_name.split()[0]

def default_voice(lang_name: str, gender: str) -> str:
    family = language_family(lang_name)
    options = VOICES.get(family, {})
    vals = options.get(gender) or options.get("Male") or []
    if vals:
        return vals[0]
    return "en-US-GuyNeural"

# ===================== LICENSE SYSTEM (Request Code + Serial) =====================

import platform
import uuid
import base64
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

LOCAL_LICENSE_FILE = Path.home() / ".dubby" / "license.json"
MACHINE_ID_FILE = Path.home() / ".dubby" / "machine.id"

# Secret used only for simple validation (change this to your own secret)
LICENSE_SECRET = "DubbySecretKey2026_ChangeMe"


def get_machine_id() -> str:
    if MACHINE_ID_FILE.exists():
        try:
            return MACHINE_ID_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    raw = f"{platform.node()}|{platform.system()}|{platform.machine()}|{uuid.getnode()}"
    mid = hashlib.sha256(raw.encode()).hexdigest()[:32]
    try:
        MACHINE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        MACHINE_ID_FILE.write_text(mid, encoding="utf-8")
    except Exception:
        pass
    return mid


def get_request_code() -> str:
    """Friendly Request Code shown to the customer."""
    mid = get_machine_id()
    # Make it shorter and easier to copy
    short = mid[:16].upper()
    return f"REQ-{short[:4]}-{short[4:8]}-{short[8:12]}-{short[12:16]}"


def save_local_license(data: dict):
    try:
        LOCAL_LICENSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_LICENSE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[License] Save error: {e}")


def load_local_license() -> Optional[dict]:
    if not LOCAL_LICENSE_FILE.exists():
        return None
    try:
        return json.loads(LOCAL_LICENSE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_local_license():
    try:
        if LOCAL_LICENSE_FILE.exists():
            LOCAL_LICENSE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def validate_serial(serial: str) -> dict:
    serial = (serial or "").strip().upper()
    if not serial:
        return {"valid": False, "message": "Empty serial number"}

    machine_id = get_machine_id()
    request_code = get_request_code()

    # Expected format: DUBY-1-L-00000000-XXXXXXXX-XXXXXXXX-XXXX
    #             or: DUBY-1-D-20261015-XXXXXXXX-XXXXXXXX-XXXX
    parts = serial.split("-")
    if len(parts) != 7 or parts[0] != "DUBY" or parts[1] != "1":
        return {"valid": False, "message": "Invalid serial format"}

    type_code = parts[2]          # L or D
    expiry_code = parts[3]       # 00000000 or YYYYMMDD
    req_hash = parts[4]
    random_part = parts[5]
    checksum = parts[6]

    # Verify checksum
    data = f"{type_code}|{expiry_code}|{req_hash}|{random_part}|{LICENSE_SECRET}"
    expected_checksum = hashlib.sha256(data.encode()).hexdigest()[:4].upper()
    if checksum != expected_checksum:
        return {"valid": False, "message": "Invalid serial (checksum failed)"}

    # Verify Request Code binding
    clean_req = request_code.replace("REQ-", "").replace("-", "")
    expected_req_hash = hashlib.sha256((clean_req + LICENSE_SECRET).encode()).hexdigest()[:8].upper()
    if req_hash != expected_req_hash:
        return {"valid": False, "message": "This serial is not for this computer"}

    # Determine type and expiry
    if type_code == "L":
        lic_type = "lifetime"
        expires_at = None
        days_left = None
    elif type_code == "D":
        lic_type = "days"
        try:
            exp_dt = datetime.strptime(expiry_code, "%Y%m%d").replace(tzinfo=timezone.utc)
            # Set to end of that day
            exp_dt = exp_dt.replace(hour=23, minute=59, second=59)
            now = datetime.now(timezone.utc)
            if now > exp_dt:
                return {"valid": False, "message": "License has expired"}
            expires_at = exp_dt.isoformat()
            days_left = (exp_dt - now).days
        except Exception:
            return {"valid": False, "message": "Invalid expiry date in serial"}
    else:
        return {"valid": False, "message": "Unknown license type"}

    # Save locally
    data = {
        "serial": serial,
        "type": lic_type,
        "expires_at": expires_at,
        "machine_id": machine_id,
        "request_code": request_code,
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "message": "License activated"
    }
    save_local_license(data)

    return {
        "valid": True,
        "type": lic_type,
        "expires_at": expires_at,
        "days_left": days_left,
        "message": "License activated successfully"
    }


# ===================== SOFTWARE UPDATE CHECK =====================

def _parse_version_tuple(version: str) -> Tuple[int, ...]:
    """Convert '4.8.1' / 'v4.8.1-beta' into a comparable integer tuple."""
    raw = str(version or "").strip().lstrip("vV")
    parts = re.split(r"[^\d]+", raw)
    nums = []
    for p in parts:
        if p.isdigit():
            nums.append(int(p))
        if len(nums) >= 4:
            break
    return tuple(nums) if nums else (0,)


def is_newer_version(remote: str, local: str = APP_VERSION) -> bool:
    """True when remote version is strictly greater than the installed version."""
    try:
        return _parse_version_tuple(remote) > _parse_version_tuple(local)
    except Exception:
        return False


def fetch_update_info(url: str = "", timeout: float = UPDATE_CHECK_TIMEOUT_SEC) -> dict:
    """Download and parse the manager's update.json feed.

    Expected keys:
      version, download_url, changelog, mandatory (bool), release_notes_url
    """
    if requests is None:
        raise RuntimeError("requests is not installed. Run: python -m pip install requests")
    feed = (url or UPDATE_INFO_URL or "").strip()
    if not feed or "YOUR_ORG" in feed:
        raise RuntimeError(
            "Update URL is not configured yet.\n\n"
            "Manager: set UPDATE_INFO_URL near the top of the script to your "
            "hosted update.json (GitHub raw, your website, etc.)."
        )
    response = requests.get(feed, timeout=max(4.0, float(timeout or 12)))
    if response.status_code != 200:
        raise RuntimeError(f"Update server returned HTTP {response.status_code}")
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Update feed is not a JSON object.")
    version = str(data.get("version") or "").strip()
    if not version:
        raise RuntimeError("Update feed is missing the 'version' field.")
    return {
        "version": version,
        "download_url": str(data.get("download_url") or data.get("url") or "").strip(),
        "changelog": str(data.get("changelog") or data.get("notes") or "").strip(),
        "mandatory": bool(data.get("mandatory") or data.get("force") or False),
        "release_notes_url": str(data.get("release_notes_url") or data.get("notes_url") or "").strip(),
        "raw": data,
    }


class UpdateCheckWorker(QThread):
    """Background check so the UI does not freeze while contacting the server."""
    finished_ok = pyqtSignal(dict)
    finished_error = pyqtSignal(str)

    def __init__(self, url: str = "", parent=None):
        super().__init__(parent)
        self.url = url or UPDATE_INFO_URL

    def run(self):
        try:
            info = fetch_update_info(self.url)
            self.finished_ok.emit(info)
        except Exception as exc:
            self.finished_error.emit(str(exc))


class UpdateAvailableDialog(QDialog):
    """Show remote version, changelog, and actions to open/download the update."""

    def __init__(self, info: dict, parent=None):
        super().__init__(parent)
        self.info = info or {}
        self.setWindowTitle(f"Update Available — {APP_NAME}")
        self.setMinimumWidth(520)
        self.setMinimumHeight(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)

        remote = str(self.info.get("version") or "?")
        title = QLabel(
            f"<b>A newer version is available</b><br>"
            f"Installed: <b>v{APP_VERSION}</b> &nbsp;→&nbsp; Latest: <b>v{remote}</b>"
        )
        title.setWordWrap(True)
        layout.addWidget(title)

        if self.info.get("mandatory"):
            warn = QLabel("⚠ This update is marked as important by the manager.")
            warn.setStyleSheet("color: #c0392b; font-weight: 600;")
            layout.addWidget(warn)

        layout.addWidget(QLabel("<b>What's new</b>"))
        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setPlainText(self.info.get("changelog") or "(No changelog provided.)")
        notes.setMinimumHeight(160)
        layout.addWidget(notes)

        btn_row = QHBoxLayout()
        self.btn_download = QPushButton("Download Update…")
        self.btn_download.setMinimumHeight(36)
        self.btn_download.clicked.connect(self._download_or_open)
        btn_row.addWidget(self.btn_download)

        self.btn_notes = QPushButton("Release Notes")
        self.btn_notes.setMinimumHeight(36)
        self.btn_notes.setEnabled(bool(self.info.get("release_notes_url")))
        self.btn_notes.clicked.connect(self._open_notes)
        btn_row.addWidget(self.btn_notes)

        self.btn_later = QPushButton("Later")
        self.btn_later.setMinimumHeight(36)
        self.btn_later.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_later)
        layout.addLayout(btn_row)

        tip = QLabel(
            "Tip: After downloading the new .py file, replace the old script "
            "and restart Dubby AI Studio. Your user_settings.json and license stay intact."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(tip)

    def _open_url(self, url: str):
        url = (url or "").strip()
        if not url:
            return
        try:
            QDesktopServices.openUrl(QUrl(url))
        except Exception:
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception as exc:
                QMessageBox.warning(self, "Open URL", f"Could not open link:\n{url}\n\n{exc}")

    def _open_notes(self):
        self._open_url(self.info.get("release_notes_url"))

    def _download_or_open(self):
        url = str(self.info.get("download_url") or "").strip()
        if not url:
            QMessageBox.information(
                self, "Download",
                "No download_url was provided in the update feed.\n"
                "Ask the manager for the new installer / script."
            )
            return
        # Prefer saving the file next to the app when it looks like a direct file link.
        lower = url.lower()
        looks_like_file = any(lower.endswith(ext) for ext in (".py", ".zip", ".exe", ".msi"))
        if looks_like_file and requests is not None:
            try:
                suggested = Path(url).name or f"Dubby_AI_Studio_v{self.info.get('version', 'new')}.py"
                path, _ = QFileDialog.getSaveFileName(
                    self, "Save Update",
                    str(BASE_DIR / suggested),
                    "All Files (*.*)"
                )
                if not path:
                    return
                self.btn_download.setEnabled(False)
                self.btn_download.setText("Downloading…")
                QApplication.processEvents()
                r = requests.get(url, timeout=120, stream=True)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            f.write(chunk)
                self.btn_download.setText("Download Update…")
                self.btn_download.setEnabled(True)
                QMessageBox.information(
                    self, "Download Complete",
                    f"Saved:\n{path}\n\n"
                    "Close this app, replace the old script with the new file, then restart."
                )
                self.accept()
                return
            except Exception as exc:
                self.btn_download.setText("Download Update…")
                self.btn_download.setEnabled(True)
                QMessageBox.warning(
                    self, "Download Failed",
                    f"Could not download automatically:\n{exc}\n\nOpening the link in your browser instead."
                )
        self._open_url(url)


# ------------------------- Worker classes -------------------------

def _gemini_api_keys(settings):
    keys = settings.get("gemini_api_keys") or settings.get("gemini_api_key", "")
    if isinstance(keys, str):
        keys = split_api_keys(keys)
    return [str(k).strip() for k in (keys or []) if str(k).strip()]


def _gemini_transcription_schema():
    return {
        "type": "object",
        "properties": {
            "language": {"type": "string"},
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "text": {"type": "string"},
                        "gender": {"type": "string", "enum": ["Male", "Female", "Unknown"]},
                        "gender_confidence": {"type": "number"}
                    },
                    "required": ["speaker", "start", "end", "text", "gender", "gender_confidence"]
                }
            }
        },
        "required": ["language", "segments"]
    }


def _gemini_json_response(response):
    text = getattr(response, "text", None)
    if text:
        return json.loads(text)
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError("Gemini returned an empty transcription response.")


def gemini_transcribe_audio(audio_path: str, settings: dict, progress_cb=None):
    """Transcribe audio with Gemini using the Google GenAI Python SDK.

    Supports 5–10+ minute videos by splitting long audio into overlapping
    chunks, transcribing each chunk, then merging timestamps/speakers so a
    batch of ~20 medium-length videos stays reliable.

    Hardened against:
    - uploaded file still processing (wait for ACTIVE)
    - long blocking generate_content (threaded timeout per chunk)
    - schema / empty response failures with a clear error instead of a silent crash
    """
    if google_genai is None:
        raise RuntimeError("Google GenAI SDK is not installed. Run: python -m pip install -U google-genai")
    keys = _gemini_api_keys(settings)
    if not keys:
        raise RuntimeError("No Gemini API key configured. Open AI / API Settings and add a Gemini API key.")

    model_name = str(settings.get("gemini_transcribe_model", "gemini-3.7-flash") or "gemini-3.7-flash").strip()
    if not model_name:
        model_name = "gemini-3.6-flash"

    audio_path = str(audio_path)
    if not Path(audio_path).exists():
        raise RuntimeError(f"Audio file missing for Gemini transcription: {audio_path}")

    # Chunk settings — keep each request within Gemini's comfortable window.
    # 150s chunks with 8s overlap work well for 5–10 min videos and still
    # scale to ~20 videos in a sequential batch.
    chunk_sec = float(settings.get("gemini_chunk_sec", 150) or 150)
    overlap_sec = float(settings.get("gemini_chunk_overlap", 8) or 8)
    chunk_sec = max(60.0, min(240.0, chunk_sec))
    overlap_sec = max(2.0, min(20.0, overlap_sec))

    try:
        size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
        if progress_cb:
            progress_cb(f"Gemini: audio ready ({size_mb:.1f} MB)...")
        # 16 kHz mono PCM ≈ 1.9 MB/min. 20 min ≈ 38 MB; allow up to ~25 min raw.
        if size_mb > 120:
            raise RuntimeError(
                f"Audio is too large for Gemini upload ({size_mb:.0f} MB). "
                "Use a shorter clip or Local Whisper + Analyze."
            )
    except RuntimeError:
        raise
    except Exception:
        pass

    total_dur = ffprobe_duration(audio_path)
    if total_dur <= 0.05 and AudioSegment is not None:
        try:
            total_dur = len(AudioSegment.from_file(audio_path)) / 1000.0
        except Exception:
            total_dur = 0.0

    prompt = """
You are the transcription and speaker-analysis engine for a video dubbing application.
Analyze the attached audio carefully.

Return ONLY the requested structured JSON.
Requirements:
1. Transcribe the actual spoken words accurately in the original spoken language.
2. Split the transcript into natural speech segments with START and END times in seconds relative to THIS audio file (start at 0.0).
3. Identify distinct speakers consistently. Use stable labels such as Speaker 1, Speaker 2.
4. Do NOT create a new speaker for every sentence. Keep the same speaker label whenever the voice is the same.
5. For each segment, estimate Male/Female/Unknown ONLY from the speaker's vocal characteristics. Do not infer gender from names, words, role, or context.
6. Use Unknown when the audio is too short, noisy, overlapping, or ambiguous.
7. gender_confidence must be between 0.0 and 1.0. Be conservative; do not output high confidence when uncertain.
8. Keep timestamps accurate to the actual speech. Prefer short natural phrases (typically 1–8 seconds).
9. Preserve repeated words, names, and spoken wording; do not summarize.
10. The application will aggregate gender across segments for each speaker, so keep gender consistent for the same voice.
""".strip()

    def _file_state_name(file_obj) -> str:
        state = getattr(file_obj, "state", None)
        if state is None:
            return "ACTIVE"
        name = getattr(state, "name", None) or str(state)
        return str(name).split(".")[-1].upper()

    def _wait_file_active(client, uploaded, progress_cb=None, timeout_sec=120):
        name = getattr(uploaded, "name", None) or getattr(uploaded, "uri", None)
        if not name:
            return uploaded
        deadline = time.monotonic() + timeout_sec
        last = _file_state_name(uploaded)
        while time.monotonic() < deadline:
            state = _file_state_name(uploaded)
            last = state
            if state in ("ACTIVE", "SUCCEEDED", "OK", ""):
                return uploaded
            if state in ("FAILED", "ERROR"):
                raise RuntimeError(f"Gemini file processing failed (state={state}).")
            if progress_cb:
                progress_cb(f"Gemini: waiting for audio processing ({state})...")
            time.sleep(1.2)
            try:
                uploaded = client.files.get(name=name)
            except TypeError:
                try:
                    uploaded = client.files.get(name)
                except Exception:
                    break
            except Exception:
                break
        if last not in ("ACTIVE", "SUCCEEDED", "OK", ""):
            if progress_cb:
                progress_cb(f"Gemini: proceeding after wait (last state={last})...")
        return uploaded

    def _call_generate(client, model_name, prompt, uploaded, config):
        return client.models.generate_content(
            model=model_name,
            contents=[prompt, uploaded],
            config=config,
        )

    def _build_chunks(duration: float):
        """Return list of (offset_sec, chunk_wav_path) covering the full file."""
        if duration <= 0 or duration <= chunk_sec + 15:
            return [(0.0, audio_path)]
        chunks = []
        step = max(30.0, chunk_sec - overlap_sec)
        t0 = 0.0
        idx = 0
        while t0 < duration - 0.5:
            t1 = min(duration, t0 + chunk_sec)
            # Last chunk: extend to end if remaining is short.
            if duration - t1 < overlap_sec + 5 and t1 < duration:
                t1 = duration
            out = TEMP_DIR / f"gemini_chunk_{hashlib.md5((audio_path + str(idx)).encode()).hexdigest()[:10]}_{idx}.wav"
            # Re-encode slice with FFmpeg for clean PCM boundaries.
            result = run_ffmpeg([
                "-y", "-ss", f"{t0:.3f}", "-t", f"{max(0.5, t1 - t0):.3f}",
                "-i", audio_path,
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                str(out),
            ])
            if result.returncode != 0 or not out.exists() or out.stat().st_size < 2000:
                # Fallback: full file once if chunking fails.
                if not chunks:
                    return [(0.0, audio_path)]
                break
            chunks.append((t0, str(out)))
            idx += 1
            if t1 >= duration - 0.05:
                break
            t0 += step
        return chunks or [(0.0, audio_path)]

    def _transcribe_one_file(path: str, key_index: int, chunk_label: str):
        nonlocal last_error
        client = None
        uploaded = None
        key = keys[key_index % len(keys)]
        try:
            if progress_cb:
                progress_cb(f"Gemini: uploading {chunk_label} (key {key_index % len(keys) + 1}/{len(keys)})...")
            client = google_genai.Client(api_key=key)
            upload_source = _gemini_ascii_upload_copy(path, "gemini_transcribe_upload")
            uploaded = client.files.upload(file=str(upload_source))
            uploaded = _wait_file_active(client, uploaded, progress_cb=progress_cb, timeout_sec=90)

            if progress_cb:
                progress_cb(f"Gemini: transcribing {chunk_label} with {model_name}...")

            schema = _gemini_transcription_schema()
            if google_genai_types is not None:
                config = google_genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                )
            else:
                config = {
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "temperature": 0.1,
                }

            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
            # Per-chunk timeout scales with content; keep batch of 20 videos responsive.
            timeout_sec = 240 if total_dur and total_dur > 180 else 180
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_call_generate, client, model_name, prompt, uploaded, config)
                try:
                    response = fut.result(timeout=timeout_sec)
                except FuturesTimeout:
                    raise RuntimeError(
                        f"Gemini transcription timed out after {timeout_sec}s on {chunk_label}. "
                        "Try another model or Local Whisper."
                    )

            data = _gemini_json_response(response)
            if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
                raise RuntimeError("Gemini returned invalid transcription JSON.")
            return data
        finally:
            try:
                if "upload_source" in locals() and upload_source is not None:
                    try:
                        Path(upload_source).unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if client is not None and uploaded is not None:
                    name = getattr(uploaded, "name", None)
                    if name:
                        try:
                            client.files.delete(name=name)
                        except TypeError:
                            try:
                                client.files.delete(name)
                            except Exception:
                                pass
            except Exception:
                pass

    def _merge_chunk_segments(chunk_results):
        """Merge overlapping chunk segments with absolute timestamps."""
        merged = []
        language = "unknown"
        for offset, data in chunk_results:
            if not data:
                continue
            language = str(data.get("language") or language) or language
            for row in data.get("segments") or []:
                try:
                    start = max(0.0, float(row.get("start", 0))) + offset
                    end = max(start + 0.05, float(row.get("end", start + 0.1)) + offset)
                except Exception:
                    continue
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                # Drop tiny fragments in overlap tails that duplicate the next chunk.
                item = dict(row)
                item["start"] = round(start, 3)
                item["end"] = round(end, 3)
                item["text"] = text
                merged.append(item)

        if not merged:
            return {"language": language, "segments": []}

        merged.sort(key=lambda r: (float(r["start"]), float(r["end"])))

        # Deduplicate near-identical overlapping rows from adjacent chunks.
        deduped = []
        for row in merged:
            if not deduped:
                deduped.append(row)
                continue
            prev = deduped[-1]
            ov = _overlap(float(prev["start"]), float(prev["end"]),
                          float(row["start"]), float(row["end"]))
            prev_dur = max(0.05, float(prev["end"]) - float(prev["start"]))
            row_dur = max(0.05, float(row["end"]) - float(row["start"]))
            same_text = (
                str(prev.get("text", "")).strip().lower()
                == str(row.get("text", "")).strip().lower()
            )
            # High overlap + same text → keep the longer / earlier segment.
            if same_text and ov > 0.45 * min(prev_dur, row_dur):
                if row_dur > prev_dur * 1.15:
                    deduped[-1] = row
                continue
            # Near-identical start within 0.35s and similar text prefix.
            if abs(float(row["start"]) - float(prev["start"])) < 0.35:
                a = str(prev.get("text", "")).strip().lower()[:24]
                b = str(row.get("text", "")).strip().lower()[:24]
                if a and b and (a in b or b in a or a == b):
                    if row_dur > prev_dur:
                        deduped[-1] = row
                    continue
            deduped.append(row)

        # Normalize speaker labels across chunks (Speaker 1 stays consistent
        # when Gemini restarts numbering in a later chunk — best-effort by
        # temporal continuity of the same gender + adjacent turns).
        return {"language": language, "segments": deduped}

    last_error = None
    chunks = _build_chunks(total_dur)
    if progress_cb:
        if len(chunks) > 1:
            progress_cb(
                f"Gemini: long audio ({total_dur:.0f}s) → {len(chunks)} chunks "
                f"(~{chunk_sec:.0f}s each) for reliable 5–10 min transcription..."
            )
        else:
            progress_cb(f"Gemini: single-pass transcription ({total_dur:.0f}s)...")

    chunk_results = []
    key_cursor = 0
    for ci, (offset, path) in enumerate(chunks):
        label = f"chunk {ci + 1}/{len(chunks)} @ {offset:.0f}s"
        success = False
        # Rotate through keys on failure so batch jobs of 20 videos stay resilient.
        for attempt in range(max(1, len(keys))):
            try:
                data = _transcribe_one_file(path, key_cursor + attempt, label)
                chunk_results.append((offset, data))
                success = True
                key_cursor = (key_cursor + attempt + 1) % max(1, len(keys))
                break
            except Exception as exc:
                last_error = exc
                if progress_cb:
                    progress_cb(f"Gemini {label} failed: {str(exc)[:160]}")
                time.sleep(min(4.0, 0.8 * (attempt + 1)))
        if not success:
            # Cleanup temporary chunk files before raising.
            for off, cpath in chunks:
                if cpath != audio_path:
                    try:
                        Path(cpath).unlink(missing_ok=True)
                    except Exception:
                        pass
            raise RuntimeError(f"Gemini transcription failed on {label}: {last_error}")

    # Cleanup temporary chunk WAVs (never delete the original audio_path).
    for off, cpath in chunks:
        if cpath != audio_path:
            try:
                Path(cpath).unlink(missing_ok=True)
            except Exception:
                pass

    if not chunk_results:
        raise RuntimeError(f"Gemini transcription failed: {last_error}")

    merged = _merge_chunk_segments(chunk_results)
    if progress_cb:
        progress_cb(
            f"Gemini: merged {len(merged.get('segments') or [])} segments "
            f"from {len(chunk_results)} chunk(s)."
        )
    return merged


def aggregate_gemini_gender(rows):
    buckets = {}
    for row in rows:
        sp = str(row.get("speaker") or "Speaker 1")
        g = str(row.get("gender") or "Unknown")
        c = max(0.0, min(1.0, float(row.get("gender_confidence", 0) or 0)))
        dur = max(0.05, float(row.get("end", 0) or 0) - float(row.get("start", 0) or 0))
        b = buckets.setdefault(sp, {"Male": 0.0, "Female": 0.0, "Unknown": 0.0})
        b[g if g in b else "Unknown"] += dur * max(c, 0.15)
    result = {}
    for sp, b in buckets.items():
        known = {k:v for k,v in b.items() if k in ("Male", "Female")}
        if not known or max(known.values()) <= 0:
            result[sp] = {"gender":"Unknown", "confidence":0.0, "male":0.0, "female":0.0}
            continue
        gender = max(known, key=known.get)
        total = max(1e-9, known.get("Male",0)+known.get("Female",0))
        conf = known[gender] / total
        result[sp] = {
            "gender": gender,
            "confidence": min(0.99, conf),
            "male": known.get("Male",0)/total,
            "female": known.get("Female",0)/total,
        }
    return result


def _activity_json_response_text(text: str):
    """Parse Gemini activity JSON even if the model wraps it in markdown."""
    raw = str(text or "").strip()
    if not raw:
        raise RuntimeError("Gemini returned an empty activity analysis response.")
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Remove common markdown fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Extract the largest JSON object in the response.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            pass
    raise RuntimeError("Gemini returned invalid activity JSON. Try again or use a different Gemini model.")


def _gemini_ascii_upload_copy(file_path, prefix="gemini_upload"):
    """Stage a local file under an ASCII-only Windows path before Gemini upload.

    The Google GenAI SDK can fail on Windows when the original path/filename
    contains non-ASCII characters (for example Khmer text or U+FF5C '｜').
    """
    source = Path(file_path)
    if not source.exists():
        raise FileNotFoundError(f"Upload source not found: {source}")

    suffix = source.suffix.lower()
    if not suffix or not suffix.isascii():
        suffix = ".bin"

    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
    staged = TEMP_DIR / f"{prefix}_{digest}{suffix}"
    shutil.copy2(str(source), str(staged))
    return staged


def gemini_analyze_video_activities(video_path: str, settings: dict, progress_cb=None):
    """Analyze actual video frames + audio and return timestamped activities.

    This is intentionally separate from speech transcription. Gemini receives
    the original video through the Files API, allowing it to understand what
    is visibly happening even when there is little or no dialogue.
    """
    if google_genai is None:
        raise RuntimeError("Google GenAI SDK is not installed. Run: python -m pip install -U google-genai")
    keys = _gemini_api_keys(settings)
    if not keys:
        raise RuntimeError("No Gemini API key configured. Open AI / API Settings and add a Gemini API key.")
    path = Path(video_path)
    if not path.exists():
        raise RuntimeError(f"Video file not found: {video_path}")

    model_name = str(
        settings.get("gemini_activity_model")
        or settings.get("gemini_transcribe_model")
        or "gemini-3.7-flash"
    ).strip()
    target_language = str(settings.get("target_lang_name") or "Khmer").strip()

    prompt = f"""
You are the visual activity analysis engine for a professional video dubbing application.
Analyze the ENTIRE attached video using both visual information and audio context.
The main goal is to identify what the people/subjects are actually doing on screen.

Return ONLY valid JSON with this exact top-level shape:
{{
  "activities": [
    {{
      "start": 0.0,
      "end": 5.0,
      "activity": "Short English action label",
      "description": "Concise factual description of what is visibly happening.",
      "narration": "Natural {target_language} narration describing the action."
    }}
  ]
}}

Rules:
1. Cover the entire video with meaningful events, but DO NOT create one row for every second/frame.
2. Group continuous actions into sensible events, normally 3–20 seconds each depending on the scene.
3. Use accurate START and END timestamps in seconds from the beginning of the video.
4. Focus on visible actions: walking, climbing, collecting wood, preparing food, lighting a fire,
   cooking, washing, cutting, carrying, entering/exiting, talking, looking, building, etc.
5. Mention important objects and locations when they help explain the action.
6. If nothing meaningful changes, keep one continuous activity instead of creating repetitive rows.
7. Do NOT invent actions that cannot reasonably be seen or inferred from the video.
8. Do not describe camera movements unless they are necessary to understand the action.
9. "activity" and "description" must be concise English. "narration" must be natural {target_language}.
10. Narration should sound like a human video storyteller, not a literal word-for-word translation.
11. Keep each narration short enough to be spoken naturally inside its own START/END window. Prefer one simple sentence; do not pack multiple clauses into a short activity.
12. Use natural punctuation for breathing, but avoid excessive commas, ellipses, or dramatic pauses.
13. Do not include markdown, comments, or extra text outside the JSON object.
""".strip()

    last_error = None
    for key_index, key in enumerate(keys):
        client = None
        uploaded = None
        try:
            if progress_cb:
                progress_cb(f"Activity AI: uploading video with Gemini key {key_index + 1}/{len(keys)}...")
            client = google_genai.Client(api_key=key)
            upload_source = _gemini_ascii_upload_copy(path, "gemini_activity_upload")
            uploaded = client.files.upload(file=str(upload_source))

            # Poll File API processing until ACTIVE.
            name = getattr(uploaded, "name", None) or getattr(uploaded, "uri", None)
            deadline = time.monotonic() + 300
            while True:
                state = getattr(getattr(uploaded, "state", None), "name", None) or str(getattr(uploaded, "state", "ACTIVE"))
                state = str(state).split(".")[-1].upper()
                if state in ("ACTIVE", "SUCCEEDED", "OK", ""):
                    break
                if state in ("FAILED", "ERROR"):
                    raise RuntimeError(f"Gemini video processing failed (state={state}).")
                if time.monotonic() > deadline:
                    raise RuntimeError("Timed out waiting for Gemini to finish processing the video.")
                if progress_cb:
                    progress_cb(f"Activity AI: Gemini processing video ({state})...")
                time.sleep(2.0)
                if name:
                    try:
                        uploaded = client.files.get(name=name)
                    except TypeError:
                        uploaded = client.files.get(name)

            if progress_cb:
                progress_cb("Activity AI: analyzing visual actions + timestamps...")

            # Use the current Interactions API for direct video understanding.
            interaction = client.interactions.create(
                model=model_name,
                input=[
                    {"type": "video", "uri": uploaded.uri, "mime_type": getattr(uploaded, "mime_type", "video/mp4")},
                    {"type": "text", "text": prompt},
                ],
            )
            output_text = getattr(interaction, "output_text", None)
            if not output_text:
                # SDK versions may expose the text through output items.
                output = getattr(interaction, "output", None) or []
                pieces = []
                for item in output:
                    for content in getattr(item, "content", None) or []:
                        txt = getattr(content, "text", None)
                        if txt:
                            pieces.append(str(txt))
                output_text = "\n".join(pieces)
            data = _activity_json_response_text(output_text)
            rows = data.get("activities") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                raise RuntimeError("Gemini activity response did not contain an 'activities' list.")

            result = []
            duration = ffprobe_duration(str(path))
            for i, row in enumerate(rows):
                try:
                    start = max(0.0, float(row.get("start", 0)))
                    end = max(start + 0.10, float(row.get("end", start + 1.0)))
                except Exception:
                    continue
                if duration > 0.05:
                    start = min(start, duration)
                    end = min(max(start + 0.10, end), duration)
                activity = str(row.get("activity") or "Activity").strip()
                description = str(row.get("description") or activity).strip()
                narration = str(row.get("narration") or description).strip()
                if not activity or not narration:
                    continue
                result.append({
                    "index": len(result) + 1,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "activity": activity,
                    "description": description,
                    "narration": narration,
                })
            result.sort(key=lambda x: (x["start"], x["end"]))
            if not result:
                raise RuntimeError("Gemini found no usable activities in this video.")
            if progress_cb:
                progress_cb(f"Activity AI complete: {len(result)} timestamped activities found.")
            return result
        except Exception as exc:
            last_error = exc
            if progress_cb:
                progress_cb(f"Activity AI key {key_index + 1}/{len(keys)} failed: {str(exc)[:180]}")
        finally:
            try:
                if "upload_source" in locals() and upload_source is not None:
                    try:
                        Path(upload_source).unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if client is not None and uploaded is not None:
                    name = getattr(uploaded, "name", None)
                    if name:
                        try:
                            client.files.delete(name=name)
                        except TypeError:
                            client.files.delete(name)
                        except Exception:
                            pass
            except Exception:
                pass

    raise RuntimeError(f"Gemini activity analysis failed: {last_error}")

class LicenseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Activate {APP_NAME}")
        self.setMinimumWidth(520)
        self.result = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel(f"<b>Activate {APP_NAME}</b>")
        title.setStyleSheet("font-size: 16px;")
        layout.addWidget(title)

        # Request Code section
        layout.addWidget(QLabel("<b>1. Your Request Code</b> (send this to the seller):"))
        self.txt_request = QLineEdit()
        self.txt_request.setReadOnly(True)
        self.txt_request.setText(get_request_code())
        self.txt_request.setMinimumHeight(36)
        self.txt_request.setStyleSheet("font-size: 14px; font-weight: bold; background: #f0f0f0;")
        layout.addWidget(self.txt_request)

        btn_copy = QPushButton("Copy Request Code")
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(self.txt_request.text()))
        layout.addWidget(btn_copy)

        layout.addSpacing(10)

        # Serial section
        layout.addWidget(QLabel("<b>2. Enter Serial Number</b> (received from seller):"))
        self.txt_serial = QLineEdit()
        self.txt_serial.setPlaceholderText("DUBY-XXXX-XXXX-XXXX-XXXX")
        self.txt_serial.setMinimumHeight(36)
        layout.addWidget(self.txt_serial)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        layout.addWidget(self.status_lbl)

        btn_row = QHBoxLayout()
        self.btn_activate = QPushButton("Activate")
        self.btn_activate.setMinimumHeight(38)
        self.btn_activate.clicked.connect(self._activate)
        btn_row.addWidget(self.btn_activate)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setMinimumHeight(38)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        info = QLabel(
            "Send the Request Code to the seller.\n"
            "After you receive the Serial Number, paste it above and click Activate."
        )
        info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info)

    def _activate(self):
        serial = self.txt_serial.text().strip()
        if not serial:
            self.status_lbl.setText("Please enter the Serial Number.")
            self.status_lbl.setStyleSheet("color: #e74c3c;")
            return

        self.btn_activate.setEnabled(False)
        self.status_lbl.setText("Validating...")
        self.status_lbl.setStyleSheet("color: #3498db;")
        QApplication.processEvents()

        result = validate_serial(serial)
        if result.get("valid"):
            self.result = result
            self.status_lbl.setText(result.get("message", "Success"))
            self.status_lbl.setStyleSheet("color: #27ae60;")
            QMessageBox.information(self, "Activated", result.get("message", "License activated successfully."))
            self.accept()
        else:
            self.status_lbl.setText(result.get("message", "Invalid serial"))
            self.status_lbl.setStyleSheet("color: #e74c3c;")
        self.btn_activate.setEnabled(True)



class GeminiTranscribeWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, list, str, dict)
    error = pyqtSignal(str)

    def __init__(self, video_path: str, video_name: str, default_gender: str = "Auto", settings=None):
        super().__init__()
        self.video_path = video_path
        self.video_name = video_name
        self.default_gender = default_gender
        self.settings = settings or {}

    def run(self):
        audio_tmp = TEMP_DIR / f"gemini_{hashlib.md5(self.video_name.encode()).hexdigest()}.wav"
        try:
            extract_audio_wav(self.video_path, str(audio_tmp), 16000)
            self.progress.emit("Gemini: preparing audio...")
            data = gemini_transcribe_audio(str(audio_tmp), self.settings, self.progress.emit)
            raw_rows = data.get("segments", [])
            gender_summary = aggregate_gemini_gender(raw_rows)
            speaker_names = {}
            result = []
            for i, row in enumerate(raw_rows):
                try:
                    start = max(0.0, float(row.get("start", 0)))
                    end = max(start + 0.05, float(row.get("end", start + 0.1)))
                except Exception:
                    continue
                original = str(row.get("text") or "").strip()
                if not original:
                    continue
                raw_sp = str(row.get("speaker") or "Speaker 1").strip()
                m = re.search(r"(\d+)$", raw_sp)
                if m:
                    raw = f"SPEAKER_{int(m.group(1))-1:02d}"
                    label = f"Speaker {int(m.group(1))}"
                else:
                    raw = raw_sp.upper().replace(" ", "_")
                    label = raw_sp
                speaker_names[raw] = label
                gs = gender_summary.get(raw_sp) or {"gender":"Unknown", "confidence":0, "male":0, "female":0}
                detected = gs.get("gender", "Unknown")
                chosen = self.default_gender if self.default_gender in ("Male", "Female") else detected
                if chosen not in ("Male", "Female"):
                    chosen = "Auto"
                result.append({
                    "index": len(result)+1, "start": round(start,3), "end": round(end,3),
                    "speaker": label, "speaker_raw": raw,
                    "gender": chosen, "detected_gender": detected,
                    "gender_confidence": float(gs.get("confidence",0) or 0),
                    "gender_male": float(gs.get("male",0) or 0),
                    "gender_female": float(gs.get("female",0) or 0),
                    "original": original, "translated": "", "tts_text": "", "voice": "", "audio_path": ""
                })
            hardware = {
                "device": "Google Gemini API", "compute": "cloud",
                "diarization": "Gemini", "speaker_count": len(set(s["speaker"] for s in result)),
                "diarization_error": "",
                "gender_model": "Gemini audio analysis",
                "gender_summary": {speaker_names.get(raw, raw): val for raw, val in []},
            }
            # Expose summary keyed by stable speaker label.
            hardware["gender_summary"] = {}
            for row in result:
                if row["speaker"] not in hardware["gender_summary"]:
                    hardware["gender_summary"][row["speaker"]] = {
                        "gender": row["detected_gender"],
                        "confidence": row["gender_confidence"]
                    }
            self.finished.emit(self.video_name, result, str(data.get("language") or "unknown"), hardware)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            try:
                if audio_tmp.exists(): audio_tmp.unlink()
            except Exception:
                pass


class TranscribeWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, list, str, dict)
    error = pyqtSignal(str)

    def __init__(self, video_path: str, video_name: str, model_size: str,
                 use_gpu: bool, default_gender: str = "Auto", settings=None):
        super().__init__()
        self.video_path = video_path
        self.video_name = video_name
        self.model_size = model_size
        self.use_gpu = use_gpu
        self.default_gender = default_gender
        self.settings = settings or {}

    def run(self):
        audio_tmp = TEMP_DIR / f"diarize_{hashlib.md5(self.video_name.encode()).hexdigest()}.wav"
        try:
            if WhisperModel is None:
                raise RuntimeError("faster-whisper is not installed. Run: python -m pip install faster-whisper")
            # Defensive GPU gate: never force CUDA if the probe failed.
            want_gpu = bool(self.use_gpu) and cuda_safe_for_whisper()
            device = "cuda" if want_gpu else "cpu"
            compute_options = recommended_cuda_compute() if device == "cuda" else ["int8"]
            model = None
            compute = None
            last_error = None
            for candidate in compute_options:
                try:
                    self.progress.emit(f"Loading Whisper {self.model_size} ({device}/{candidate})...")
                    model = WhisperModel(
                        self.model_size,
                        device=device,
                        compute_type=candidate,
                        download_root=str(MODEL_CACHE_DIR),
                    )
                    compute = candidate
                    break
                except Exception as exc:
                    last_error = exc
                    model = None
            if model is None and device == "cuda":
                device, compute = "cpu", "int8"
                self.progress.emit("Whisper CUDA failed; falling back to CPU/int8 (safe)...")
                try:
                    model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                        download_root=str(MODEL_CACHE_DIR),
                    )
                except Exception as exc:
                    last_error = exc
                    model = None
            if model is None:
                raise RuntimeError(f"Unable to load Whisper model: {last_error}")

            self.progress.emit(f"Whisper ready: {device}/{compute}")
            segments, info = model.transcribe(
                self.video_path, word_timestamps=False, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300}, beam_size=5,
            )
            result = []
            for i, seg in enumerate(segments):
                original = str(seg.text or "").strip()
                if not original:
                    continue
                result.append({
                    "index": i + 1,
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "speaker": "Speaker 1",
                    "speaker_raw": "SPEAKER_00",
                    "gender": self.default_gender if self.default_gender in ("Male", "Female") else "Auto",
                    "detected_gender": self.default_gender if self.default_gender in ("Male", "Female") else "Unknown",
                    "gender_confidence": 0.0,
                    "gender_male": 0.0,
                    "gender_female": 0.0,
                    "original": original, "translated": "", "tts_text": "",
                    "voice": "", "audio_path": "",
                })

            # Always extract clean mono audio first (needed for gender even when
            # diarization is unavailable).
            diarization_ok = False
            turns = []
            speaker_map = {}
            diarization_error = ""
            audio_ok = False
            try:
                extract_audio_wav(self.video_path, str(audio_tmp), 16000)
                audio_ok = audio_tmp.exists()
            except Exception as exc:
                diarization_error = f"audio extract: {exc}"

            # Real speaker diarization (pyannote Community-1).
            # Requires HF token + accepting model terms. Without it we keep
            # a single "Speaker 1" label so transcription remains usable.
            if audio_ok:
                try:
                    token = _hf_token(self.settings)
                    if not token:
                        diarization_error = (
                            "No Hugging Face token — speaker diarization skipped. "
                            "Add HF_TOKEN in AI / API Settings and accept "
                            "pyannote/speaker-diarization-community-1 terms. "
                            "Gender detection still runs."
                        )
                        self.progress.emit(diarization_error)
                    else:
                        self.progress.emit("Running Community-1 speaker diarization...")
                        turns = diarize_audio(
                            str(audio_tmp), use_gpu=self.use_gpu, token=token,
                            min_speakers=self.settings.get("min_speakers") or None,
                            max_speakers=self.settings.get("max_speakers") or None,
                        )
                        diarization_ok = bool(turns)
                        for seg in result:
                            raw = assign_speaker(seg["start"], seg["end"], turns)
                            seg["speaker_raw"] = raw
                            seg["speaker"] = _canonical_speaker_name(raw, speaker_map)
                except Exception as exc:
                    diarization_error = (diarization_error + " | " if diarization_error else "") + str(exc)
                    for seg in result:
                        seg["speaker"] = "Speaker 1"
                        seg["speaker_raw"] = "SPEAKER_00"

            if not diarization_ok:
                for seg in result:
                    seg.setdefault("speaker", "Speaker 1")
                    seg.setdefault("speaker_raw", "SPEAKER_00")

            # ALWAYS run ML gender detection when audio is available.
            # Default Voice Gender only overrides the final label when the user
            # explicitly chose Male or Female — Auto uses the real analysis.
            gender_results = {}
            if result and audio_ok:
                speakers = sorted({s.get("speaker_raw") for s in result})
                try:
                    for raw in speakers:
                        label = speaker_map.get(raw, raw)
                        self.progress.emit(f"Analyzing gender for {label}...")
                        gender_results[raw] = detect_gender_for_speaker(
                            str(audio_tmp),
                            turns or [(0, 10**9, raw)],
                            raw,
                            use_gpu=self.use_gpu,
                        )
                    # If every speaker is still Unknown, run a whole-file pass.
                    if all(
                        gender_results.get(r, {}).get("gender", "Unknown") == "Unknown"
                        for r in speakers
                    ):
                        self.progress.emit("Gender unclear — whole-file analysis...")
                        whole = detect_gender_for_speaker(
                            str(audio_tmp),
                            [(0, 10**9, "SPEAKER_00")],
                            "SPEAKER_00",
                            use_gpu=False,
                        )
                        if whole.get("gender") in ("Male", "Female"):
                            for r in speakers:
                                gender_results[r] = whole
                except Exception as exc:
                    diarization_error = (
                        (diarization_error + " | " if diarization_error else "")
                        + f"gender: {exc}"
                    )

            for seg in result:
                raw = seg.get("speaker_raw")
                g = gender_results.get(raw, {})
                detected = g.get("gender", "Unknown")
                conf = float(g.get("confidence", 0.0) or 0.0)
                seg["detected_gender"] = detected
                seg["gender_confidence"] = conf
                seg["gender_male"] = float(g.get("male", 0.0) or 0.0)
                seg["gender_female"] = float(g.get("female", 0.0) or 0.0)

                if self.default_gender in ("Male", "Female"):
                    # User forced a global gender — still keep detected_* for review.
                    seg["gender"] = self.default_gender
                elif detected in ("Male", "Female"):
                    seg["gender"] = detected
                else:
                    seg["gender"] = "Auto"

            hardware = {
                "device": device, "compute": compute,
                "diarization": "Community-1" if diarization_ok else "Fallback",
                "speaker_count": len(set(s.get("speaker") for s in result)) if result else 0,
                "diarization_error": diarization_error,
                "gender_model": "wav2vec2+pitch" if gender_results else "not-run",
                "gender_summary": {
                    raw: {
                        "gender": gender_results[raw].get("gender"),
                        "confidence": round(float(gender_results[raw].get("confidence", 0)), 3),
                    }
                    for raw in gender_results
                },
            }
            self.finished.emit(self.video_name, result, getattr(info, "language", "unknown"), hardware)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                if audio_tmp.exists(): audio_tmp.unlink()
            except Exception:
                pass


class GenderWorker(QThread):
    """Compatibility worker. New transcription performs per-speaker gender analysis."""
    finished = pyqtSignal(str, dict)
    error = pyqtSignal(str)
    def __init__(self, video_name: str, video_path: str, segments: list, settings=None):
        super().__init__(); self.video_name=video_name; self.video_path=video_path; self.segments=segments; self.settings=settings or {}
    def run(self):
        try:
            self.finished.emit(self.video_name, {"global": "Unknown", "message": "Gender is now analyzed per speaker during transcription."})
        except Exception as e:
            self.error.emit(str(e))


class OpenAITranslateWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)

    def __init__(self, video_name: str, segments: list, target_lang: str, 
                 api_key: str, model: str = "gpt-3.5-turbo", prompt: str = None):
        super().__init__()
        self.video_name = video_name
        self.segments = segments
        self.target_lang = target_lang
        self.api_key = api_key
        self.model = model
        self.prompt = prompt or "You are a professional translator. Translate from Chinese to the target language. Preserve cultural context, idioms, and emotional tone. Keep proper names in their original form unless they have common translations."

    def run(self):
        try:
            if requests is None:
                raise RuntimeError("requests is not installed. Run: python -m pip install requests")
            
            if not self.api_key:
                raise RuntimeError("OpenAI API key is required. Please add it in AI / API Settings.")
            
            # Get language name from code
            lang_name = next((name for name, code in LANGUAGES.items() if code == self.target_lang), self.target_lang)
            
            total = max(1, len(self.segments))
            
            # Build list of texts to translate
            texts = []
            for seg in self.segments:
                text = seg.get("original", "").strip()
                if text:
                    texts.append(text)
                else:
                    texts.append("")
            
            # Filter out empty texts
            valid_indices = [i for i, t in enumerate(texts) if t]
            if not valid_indices:
                self.finished.emit(self.video_name, self.segments)
                return
            
            # Translate in batches to reduce API calls
            batch_size = 15  # Adjust based on your needs
            translations = {}
            
            for i in range(0, len(valid_indices), batch_size):
                batch_indices = valid_indices[i:i+batch_size]
                batch_texts = [texts[idx] for idx in batch_indices]
                
                # Create prompt for batch translation
                numbered_texts = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch_texts))
                prompt = f"""{self.prompt}

Translate the following texts from Chinese to {lang_name}. 
Return ONLY the translations, one per line, in the same order.

Texts to translate:
{numbered_texts}"""
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": f"You are a professional translator. Translate from Chinese to {lang_name}. Preserve meaning and natural flow. Never return error messages as translations."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
                
                # Retry logic
                response_text = None
                for attempt in range(3):
                    try:
                        response = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            response_text = response.json()["choices"][0]["message"]["content"]
                            break
                        elif response.status_code == 429:
                            time.sleep(2 ** attempt)
                            continue
                        else:
                            error_data = response.json().get("error", {})
                            error_msg = error_data.get("message", str(response.status_code))
                            if attempt < 2:
                                time.sleep(1)
                                continue
                            raise RuntimeError(f"OpenAI API error: {error_msg}")
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(1)
                            continue
                        raise
                
                if not response_text:
                    raise RuntimeError("No translation received from OpenAI")
                
                # Parse translations
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                # Remove numbering if present
                parsed = []
                for line in lines:
                    # Remove "1. " prefixes
                    cleaned = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned:
                        parsed.append(cleaned)
                
                # Ensure we have the right number
                if len(parsed) < len(batch_texts):
                    # Fill missing with original text
                    parsed.extend([batch_texts[j] for j in range(len(parsed), len(batch_texts))])
                elif len(parsed) > len(batch_texts):
                    parsed = parsed[:len(batch_texts)]
                
                # Store translations
                for j, idx in enumerate(batch_indices):
                    if j < len(parsed):
                        translations[idx] = parsed[j]
                
                self.progress.emit(f"Translating {self.video_name}: {min(i + batch_size, len(valid_indices))}/{len(valid_indices)}")
                time.sleep(0.3)  # Small delay between batches
            
            # Apply translations to segments
            for i, seg in enumerate(self.segments):
                if i in translations and translations[i]:
                    seg["translated"] = translations[i]
                    seg["tts_text"] = translations[i]
                elif not seg.get("translated"):
                    # Fallback to original if translation failed
                    seg["translated"] = seg.get("original", "")
                    seg["tts_text"] = seg.get("original", "")
            
            self.finished.emit(self.video_name, self.segments)
            
        except Exception as e:
            self.error.emit(str(e))


class TranslateWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)

    def __init__(self, video_name: str, segments: list, target_lang: str, 
                 provider: str = "google", openai_key: str = None, 
                 openai_model: str = "gpt-3.5-turbo", openai_prompt: str = None):
        super().__init__()
        self.video_name = video_name
        self.segments = segments
        self.target_lang = target_lang
        self.provider = provider  # "google", "gemini", "openai"
        self.openai_key = openai_key
        self.openai_model = openai_model
        self.openai_prompt = openai_prompt

    def run(self):
        try:
            if self.provider == "openai":
                # Use the OpenAI worker
                if self.openai_key is None:
                    raise RuntimeError("OpenAI API key is required for OpenAI translation")
                
                # Create and run OpenAI worker directly
                self.openai_worker = OpenAITranslateWorker(
                    self.video_name, self.segments, self.target_lang,
                    self.openai_key, self.openai_model, self.openai_prompt
                )
                # Copy signals
                self.openai_worker.progress.connect(self.progress.emit)
                self.openai_worker.finished.connect(self.finished.emit)
                self.openai_worker.error.connect(self.error.emit)
                self.openai_worker.start()
                return
            
            # Existing Google Translate code
            if GoogleTranslator is None:
                raise RuntimeError(
                    "deep-translator is not installed. Run: "
                    "python -m pip install deep-translator"
                )
            translator = GoogleTranslator(source="auto", target=self.target_lang)
            total = max(1, len(self.segments))

            for i, seg in enumerate(self.segments):
                text = seg.get("original", "").strip()
                if not text:
                    continue
                translated = None

                for attempt in range(4):
                    try:
                        candidate = translator.translate(text)
                        if candidate and not is_server_error_text(candidate):
                            translated = candidate
                            break
                    except Exception:
                        pass

                    if attempt < 3:
                        time.sleep(1.5 * (attempt + 1))

                if not translated:
                    translated = text

                seg["translated"] = translated
                seg["tts_text"] = translated
                self.progress.emit(
                    f"Translating {self.video_name}: {i + 1}/{total}"
                )

            self.finished.emit(self.video_name, self.segments)
        except Exception as e:
            self.error.emit(str(e))


def _voxcpm_device(mode="Auto"):
    """Map UI device mode to the official VoxCPM device string."""
    mode = str(mode or "Auto").strip()
    if mode in ("CPU", "VoxCPM2 Local CPU"):
        return "cpu"
    if mode in ("NVIDIA CUDA", "CUDA", "cuda"):
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    # Auto: prefer CUDA when VRAM is enough; otherwise CPU.
    if torch is None or not torch.cuda.is_available():
        return "cpu"
    name = nvidia_gpu_name().lower()
    # VoxCPM2 needs ~8GB VRAM — force CPU on low-VRAM cards.
    if any(x in name for x in ("1050", "1060", "1650", "quadro p", "mx150", "mx250")):
        return "cpu"
    return "cuda"


def load_voxcpm_model(device_mode="Auto", progress_cb=None):
    """Load (and cache) the VoxCPM2 model. First run downloads ~weights from HF.

    progress_cb(msg) is optional and used to keep the UI status alive while
    the large checkpoint downloads / loads.
    """
    global _VOXCPM_MODEL
    if VoxCPM is None:
        raise RuntimeError(
            "VoxCPM is not installed.\n"
            "Run:  python -m pip install voxcpm\n"
            "Requires Python 3.10–3.12 and PyTorch >= 2.5."
        )
    if _VOXCPM_MODEL is not None:
        return _VOXCPM_MODEL

    device = _voxcpm_device(device_mode)
    if progress_cb:
        progress_cb(
            f"Downloading/loading VoxCPM2 on {device}… "
            "first run can take several minutes (model is large)."
        )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    VOXCPM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    last_err = None
    # Try official kwargs first, then progressively simpler fallbacks for older builds.
    attempts = [
        dict(load_denoiser=False, device=device, optimize=(device == "cuda"),
             cache_dir=str(VOXCPM_CACHE_DIR)),
        dict(load_denoiser=False, device=device, optimize=False,
             cache_dir=str(VOXCPM_CACHE_DIR)),
        dict(load_denoiser=False, device=device, optimize=False),
        dict(load_denoiser=False, device="cpu", optimize=False),
        dict(load_denoiser=False),
    ]
    for kwargs in attempts:
        try:
            if progress_cb:
                progress_cb(f"VoxCPM2.from_pretrained(device={kwargs.get('device', 'auto')})…")
            _VOXCPM_MODEL = VoxCPM.from_pretrained("openbmb/VoxCPM2", **kwargs)
            if progress_cb:
                progress_cb(f"VoxCPM2 ready on {kwargs.get('device', 'auto')}.")
            return _VOXCPM_MODEL
        except TypeError as exc:
            # Unexpected keyword — try next signature.
            last_err = exc
            continue
        except Exception as exc:
            last_err = exc
            # OOM / CUDA errors → fall through to CPU attempt.
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg:
                continue
            # Network / HF errors should surface immediately.
            if any(tok in msg for tok in ("huggingface", "connection", "timed out", "401", "403", "404")):
                raise RuntimeError(
                    f"VoxCPM2 model download failed: {exc}\n"
                    "Check internet access to Hugging Face, or set:\n"
                    "  set HF_ENDPOINT=https://hf-mirror.com\n"
                    "then retry."
                ) from exc
            continue

    raise RuntimeError(
        f"Could not load VoxCPM2. Last error: {last_err}\n"
        "Tips:\n"
        " • pip install -U voxcpm torch soundfile\n"
        " • Choose 'VoxCPM2 Local CPU' if GPU VRAM is low\n"
        " • First download needs internet (Hugging Face)"
    )


def _voxcpm_voice_prefix(gender="Female", style=""):
    """Build the official VoxCPM2 voice-design control instruction in parentheses."""
    style = str(style or "").strip()
    if style and style.startswith("(") and style.endswith(")"):
        return style
    if gender == "Male":
        base = "A clear adult male voice, natural conversational tone, steady pace"
    else:
        base = "A clear adult female voice, natural conversational tone, warm and steady"
    if style:
        # Avoid double-wrapping; merge user style into the design tag.
        base = f"{base}, {style}"
    return f"({base})"


def save_voxcpm_tts(text, target: Path, reference_wav="", prompt_text="", style="",
                    device_mode="Auto", speed=1.0, gender="Female", progress_cb=None):
    """Generate realistic speech with VoxCPM2 (local).

    Matches the official API:
      wav = model.generate(text="(voice design)Hello", cfg_value=2.0, ...)
    Optional reference_wav_path for voice cloning.
    """
    if sf is None:
        raise RuntimeError("soundfile is required for VoxCPM2. Run: python -m pip install soundfile")
    if VoxCPM is None:
        raise RuntimeError(
            "VoxCPM is not installed.\n"
            "Install with:  python -m pip install voxcpm\n"
            "Requires Python 3.10–3.12 and PyTorch >= 2.5."
        )
    if str(device_mode).strip() in ("CPU", "VoxCPM2 Local CPU"):
        device_mode = "CPU"

    model = load_voxcpm_model(device_mode, progress_cb=progress_cb)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = str(text or "").strip()
    if not clean:
        raise RuntimeError("Empty VoxCPM text")

    # Official voice-design syntax: (instruction)text
    prefix = _voxcpm_voice_prefix(gender=gender, style=style)
    spoken = f"{prefix}{clean}"

    ref = str(reference_wav or "").strip()
    ptxt = str(prompt_text or "").strip()
    kwargs = {
        "text": spoken,
        "cfg_value": 2.0,
        "inference_timesteps": 10,
    }
    if ref and Path(ref).exists():
        kwargs["reference_wav_path"] = ref
        if ptxt:
            kwargs["prompt_wav_path"] = ref
            kwargs["prompt_text"] = ptxt

    try:
        wav = model.generate(**kwargs)
    except TypeError:
        # Older / minimal signature
        try:
            wav = model.generate(text=spoken, cfg_value=2.0, inference_timesteps=10)
        except Exception:
            wav = model.generate(text=clean)
    except Exception as exc:
        # Retry once without reference if cloning args caused the failure.
        if "reference" in str(exc).lower() or "prompt" in str(exc).lower():
            wav = model.generate(text=spoken, cfg_value=2.0, inference_timesteps=10)
        else:
            raise

    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise RuntimeError("VoxCPM2 returned empty audio array")
    sr = int(getattr(getattr(model, "tts_model", None), "sample_rate", 48000) or 48000)
    if speed and abs(float(speed) - 1.0) > 0.02 and librosa is not None:
        arr = librosa.effects.time_stretch(arr, rate=float(speed))
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1e-6:
        arr = arr * min(0.95, 0.85 / peak)
    sf.write(str(target), arr, sr)
    if not target.exists() or target.stat().st_size < 1500:
        raise RuntimeError("VoxCPM2 wrote an empty/too-small WAV file")
    return str(target)


def calculate_effective_tts_speed(seg: dict, text: str, base_speed: float = 1.0,
                              max_rate_pct: int = 45) -> float:
    """Estimate a natural speed that fits the original subtitle/activity window.

    Uses the target TTS text for the primary estimate and the original/source text
    as a secondary signal. This is especially useful for Chinese/Japanese/Korean
    source subtitles where a short source line may carry a lot of information.
    """
    try:
        base_speed = max(0.70, min(1.60, float(base_speed or 1.0)))
        start = float(seg.get("start", 0) or 0)
        end = float(seg.get("end", start + 1) or start + 1)
        slot = max(0.25, end - start)
        n = max(1, len(str(text or "").strip()))

        # Natural target-language character rates. These are intentionally
        # conservative; final export still performs frame-accurate fitting.
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            cps = 9.5
        elif any("\u3040" <= ch <= "\u30ff" for ch in text):
            cps = 10.0
        elif any("\uac00" <= ch <= "\ud7af" for ch in text):
            cps = 10.5
        elif any("\u1780" <= ch <= "\u17ff" for ch in text):
            cps = 10.5
        elif any("\u0e00" <= ch <= "\u0e7f" for ch in text):
            cps = 11.0
        else:
            cps = 14.5

        natural_sec = n / cps
        target_sec = slot * 0.94
        required = natural_sec / max(0.20, target_sec)

        # Source subtitle is only a density hint; the real target window always wins.
        source = str(seg.get("original") or "").strip()
        if source:
            source_n=max(1,len(source))
            if any("一"<=ch<="鿿" for ch in source): source_cps=9.5
            elif any("ぁ"<=ch<="ヿ" for ch in source): source_cps=10.0
            elif any("가"<=ch<="힯" for ch in source): source_cps=10.5
            else: source_cps=14.5
            source_natural=source_n/source_cps
            if source_natural>slot*0.94:
                required=max(required,source_natural/max(0.20,slot*0.92))
        auto_factor=max(0.82,min(1.0+max_rate_pct/100.0,required))
        result=base_speed*auto_factor
        return max(0.75,min(1.0+max_rate_pct/100.0,result))
    except Exception:
        return max(0.70, min(1.0 + max_rate_pct / 100.0, float(base_speed or 1.0)))


class TTSWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)

    def __init__(self, video_name: str, segments: list, lang_name: str,
                 default_gender: str, speed: float, settings: dict):
        super().__init__()
        self.video_name = video_name
        self.segments = segments
        self.lang_name = lang_name
        self.default_gender = default_gender
        self.speed = speed
        self.settings = settings or {}

    def run(self):
        try:
            provider = self.settings.get("tts_provider", "Edge-TTS")
            total = sum(1 for s in self.segments if s.get("tts_text", "").strip())
            done = 0
            failed = 0
            last_error = ""

            async def make_edge():
                nonlocal done, failed, last_error

                # Edge-TTS is network-bound. The old implementation generated
                # every sentence strictly one after another, which made a
                # 100-200 subtitle video unnecessarily slow. Generate several
                # independent sentences concurrently while keeping a modest
                # limit so Edge is not flooded with requests.
                concurrency = int(self.settings.get("edge_concurrency", 5) or 5)
                concurrency = max(1, min(8, concurrency))
                semaphore = asyncio.Semaphore(concurrency)

                # Reuse the same in-flight task when two rows have identical
                # text/voice/settings. This prevents duplicate Edge requests.
                in_flight = {}

                async def generate_one(seg):
                    text = seg.get("tts_text", "").strip()
                    if not text:
                        return True, ""

                    gender = seg.get("gender", "Auto")
                    if gender == "Auto":
                        gender = seg.get("detected_gender", "Unknown")
                    if gender not in ("Male", "Female"):
                        gender = self.default_gender

                    voice = seg.get("voice") or default_voice(self.lang_name, gender)
                    seg["voice"] = voice

                    # Automatically adapt speech speed to the ORIGINAL subtitle
                    # window. This makes short Chinese/Japanese/Korean lines fast
                    # enough while keeping longer lines natural.
                    max_rate = int(self.settings.get("edge_max_rate_pct", 60) or 60)
                    max_rate = max(20, min(55, max_rate))
                    effective_speed = calculate_effective_tts_speed(
                        seg, text, self.speed, max_rate_pct=max_rate
                    )
                    rate_pct = int(round((effective_speed - 1.0) * 100))
                    rate_pct = max(-20, min(max_rate, rate_pct))
                    rate = f"{rate_pct:+d}%"

                    pitch_hz = int(self.settings.get("edge_pitch_hz", 0) or 0)
                    pitch_hz = max(-12, min(12, pitch_hz))
                    pitch = f"{pitch_hz:+d}Hz"

                    volume_pct = int(self.settings.get("edge_volume_pct", 0) or 0)
                    volume_pct = max(-6, min(6, volume_pct))
                    volume = f"{volume_pct:+d}%"

                    cache = tts_cache_path(text, voice, rate, pitch, volume)
                    cache_key = str(cache)

                    async def do_generate():
                        # Cache reads are cheap and happen before network work.
                        if cache.exists() and cache.stat().st_size >= 1024:
                            return str(cache), ""
                        async with semaphore:
                            if cache.exists() and cache.stat().st_size >= 1024:
                                return str(cache), ""
                            await save_tts_with_retry(
                                text, voice, rate, cache, retries=4,
                                pitch=pitch, volume=volume
                            )
                        return str(cache), ""

                    task = in_flight.get(cache_key)
                    if task is None:
                        task = asyncio.create_task(do_generate())
                        in_flight[cache_key] = task

                    try:
                        audio_path, error_text = await task
                        seg["audio_path"] = audio_path
                        seg["tts_error"] = ""
                        return True, ""
                    except Exception as exc:
                        seg["audio_path"] = ""
                        seg["tts_error"] = str(exc)
                        return False, str(exc)

                tasks = [
                    asyncio.create_task(generate_one(seg))
                    for seg in self.segments
                    if seg.get("tts_text", "").strip()
                ]

                completed = 0
                for task in asyncio.as_completed(tasks):
                    ok, error_text = await task
                    completed += 1
                    if ok:
                        done += 1
                    else:
                        failed += 1
                        last_error = error_text
                    self.progress.emit(
                        int(100 * completed / max(1, total)),
                        f"Edge-TTS: {done}/{total} | {concurrency} parallel | failed {failed}"
                        + (f" | last: {last_error[:100]}" if last_error else "")
                    )

                # Second pass: retry rows that still have no usable audio file.
                # This fixes mid-batch rate limits that left silence after ~1 min.
                retry_segs = [
                    s for s in self.segments
                    if s.get("tts_text", "").strip() and (
                        not s.get("audio_path")
                        or not Path(str(s.get("audio_path"))).exists()
                        or Path(str(s.get("audio_path"))).stat().st_size < 1024
                    )
                ]
                if retry_segs:
                    self.progress.emit(
                        95,
                        f"Edge-TTS retry: {len(retry_segs)} missing segment(s)..."
                    )
                    await asyncio.sleep(1.5)
                    retry_tasks = [asyncio.create_task(generate_one(seg)) for seg in retry_segs]
                    for task in asyncio.as_completed(retry_tasks):
                        ok, error_text = await task
                        if ok:
                            done += 1
                            failed = max(0, failed - 1)
                        else:
                            last_error = error_text
                        self.progress.emit(
                            98,
                            f"Edge-TTS retry done: ok={done} failed={failed}"
                            + (f" | last: {last_error[:100]}" if last_error else "")
                        )

            def make_gemini():
                nonlocal done, failed, last_error
                keys = self.settings.get("gemini_api_keys") or self.settings.get("gemini_api_key", "")
                model = self.settings.get("gemini_tts_model", "gemini-3.1-flash-tts-preview")
                style = self.settings.get(
                    "gemini_tts_style",
                    "Professional human voice-over. Clear studio narration, natural conversational tone, steady pitch, clean pronunciation, subtle emotion, natural pauses. Do not sound robotic, exaggerated, theatrical, breathy, or synthetic."
                )
                interval = float(self.settings.get("gemini_request_interval", 0.65) or 0.65)
                key_state = {}

                for seg_index, seg in enumerate(self.segments):
                    text = seg.get("tts_text", "").strip()
                    if not text:
                        continue
                    voice = seg.get("gemini_voice") or self.settings.get("gemini_default_voice", "Kore")
                    seg["gemini_voice"] = voice
                    target_ms = max(250, int((float(seg.get("end", 0)) - float(seg.get("start", 0))) * 1000))
                    effective_speed = calculate_effective_tts_speed(
                        seg, text, self.speed,
                        max_rate_pct=int(self.settings.get("edge_max_rate_pct", 60) or 60),
                    )
                    cache = gemini_tts_cache_path(text, voice, effective_speed, model, style, target_ms)
                    try:
                        if not cache.exists() or cache.stat().st_size < 1200:
                            result = save_gemini_tts_with_retry(
                                text=text,
                                voice=voice,
                                target=cache,
                                api_keys=keys,
                                model=model,
                                style_prompt=style,
                                speed=effective_speed,
                                target_duration_sec=target_ms / 1000.0,
                                retries_per_key=2,
                                min_request_interval=interval,
                                key_state=key_state,
                            )
                        else:
                            result = "cache"
                        seg["audio_path"] = str(cache)
                        seg["voice"] = voice
                        seg["tts_error"] = ""
                        seg["tts_provider"] = "Google Gemini TTS"
                        done += 1
                        last_error = ""
                    except Exception as exc:
                        seg["audio_path"] = ""
                        seg["tts_error"] = str(exc)
                        last_error = str(exc)
                        failed += 1
                    message = f"Google TTS: {done}/{total} | failed {failed}"
                    if last_error:
                        message += f" | last: {last_error[:140]}"
                    self.progress.emit(
                        int(100 * (done + failed) / max(1, total)),
                        message
                    )

                # The worker intentionally finishes even when some segments
                # failed. Export All can then use the successful audio rows.
                if failed:
                    self.progress.emit(
                        int(100 * (done + failed) / max(1, total)),
                        f"Google TTS finished with {done} success / {failed} failed."
                        + (f" Last error: {last_error[:180]}" if last_error else "")
                    )

            def make_voxcpm(force_cpu=False):
                nonlocal done, failed, last_error
                model_name = self.settings.get("voxcpm_model", "openbmb/VoxCPM2")
                device_mode = "CPU" if force_cpu else self.settings.get("voxcpm_device", "Auto")
                style = self.settings.get("voxcpm_style", "")
                resolved = _voxcpm_device(device_mode)

                def _cb(msg):
                    self.progress.emit(max(1, int(100 * (done + failed) / max(1, total))), str(msg))

                _cb(
                    f"Loading VoxCPM2 ({device_mode} → {resolved})… "
                    "first run downloads the model from Hugging Face (can take several minutes)."
                )
                # Pre-load once so the first segment does not look "stuck".
                try:
                    load_voxcpm_model(device_mode, progress_cb=_cb)
                except Exception as exc:
                    raise RuntimeError(
                        f"VoxCPM2 failed to load: {exc}\n\n"
                        "Checklist:\n"
                        " 1. pip install -U voxcpm torch soundfile\n"
                        " 2. Internet access to Hugging Face (or HF_ENDPOINT mirror)\n"
                        " 3. Try provider 'VoxCPM2 Local CPU' on low-VRAM GPUs\n"
                        " 4. Python 3.10–3.12 required"
                    ) from exc

                for seg in self.segments:
                    text = seg.get("tts_text", "").strip()
                    if not text:
                        continue
                    gender = seg.get("gender", "Auto")
                    if gender == "Auto":
                        gender = seg.get("detected_gender", "Unknown")
                    if gender not in ("Male", "Female"):
                        gender = self.default_gender if self.default_gender in ("Male", "Female") else "Female"
                    ref = seg.get("speaker_reference_wav", "") or self.settings.get("voxcpm_reference_wav", "")
                    prompt_text = seg.get("speaker_reference_text", "") or self.settings.get("voxcpm_prompt_text", "")
                    ref_key = str(ref) if ref else "none"
                    cache_key = hashlib.sha256(
                        f"voxcpm2v2|{model_name}|{resolved}|{gender}|{ref_key}|{style}|{self.speed:.3f}|{text}".encode("utf-8")
                    ).hexdigest()
                    cache = TTS_CACHE_DIR / f"{cache_key}.wav"
                    try:
                        if not cache.exists() or cache.stat().st_size < 1500:
                            _cb(f"VoxCPM2 synthesizing {done + failed + 1}/{total} ({gender})…")
                            save_voxcpm_tts(
                                text, cache, ref, prompt_text, style,
                                device_mode, self.speed, gender=gender, progress_cb=_cb,
                            )
                        seg["audio_path"] = str(cache)
                        seg["voice"] = f"VoxCPM2 {gender}"
                        seg["tts_provider"] = "VoxCPM2"
                        seg["tts_error"] = ""
                        done += 1
                    except Exception as exc:
                        failed += 1
                        last_error = str(exc)
                        seg["audio_path"] = ""
                        seg["tts_error"] = last_error
                    self.progress.emit(
                        int(100 * (done + failed) / max(1, total)),
                        f"VoxCPM2: {done}/{total} ok | failed {failed}"
                        + (f" | {last_error[:120]}" if last_error else "")
                    )
                if failed and done == 0:
                    raise RuntimeError(
                        f"VoxCPM2 failed on all segments.\nLast error: {last_error}\n\n"
                        "Tips:\n"
                        " • Switch to 'VoxCPM2 Local CPU' if GPU VRAM is low\n"
                        " • Ensure pip install voxcpm succeeded\n"
                        " • First run needs Hugging Face download"
                    )

            if provider == "Google Gemini TTS":
                make_gemini()
            elif provider == "VoxCPM2 Local":
                make_voxcpm(force_cpu=False)
            elif provider == "VoxCPM2 Local CPU":
                make_voxcpm(force_cpu=True)
            else:
                asyncio.run(make_edge())
            self.finished.emit(self.video_name, self.segments)
        except Exception as e:
            self.error.emit(str(e))


class _ExportProgressLogger(ProgressBarLogger if ProgressBarLogger is not None else object):
    """Forward MoviePy/FFmpeg frame progress to the Qt export worker.

    MoviePy's previous logger=None made the application appear frozen from
    roughly 75% until rendering finished. This logger keeps the UI alive with
    real frame progress and elapsed time.
    """
    def __init__(self, worker, total_duration: float):
        if ProgressBarLogger is not None:
            super().__init__()
        self.worker = worker
        self.total_duration = max(0.1, float(total_duration or 0.1))
        self.started = time.monotonic()
        self.last_emit = 0.0

    def callback(self, **changes):
        try:
            if ProgressBarLogger is not None:
                super().callback(**changes)
        except Exception:
            pass
        self._emit_progress()

    def bars_callback(self, bar, attr, value, old_value=None):
        self._emit_progress()

    def _emit_progress(self):
        now = time.monotonic()
        if now - self.last_emit < 0.25:
            return
        self.last_emit = now
        try:
            bar = getattr(self, "bars", {}).get("t", {})
            total = bar.get("total") or 0
            current = bar.get("index") or 0
            if total:
                ratio = max(0.0, min(1.0, float(current) / float(total)))
            else:
                ratio = 0.0
            elapsed = now - self.started
            if ratio > 0.01 and elapsed > 1:
                eta = max(0.0, elapsed * (1.0 - ratio) / ratio)
                eta_txt = f" • ETA {int(eta//60):02d}:{int(eta%60):02d}"
            else:
                eta_txt = ""
            pct = 75 + int(ratio * 15)  # 75..90; mux remains 92..100
            self.worker.progress.emit(
                pct,
                f"Rendering video… {ratio*100:5.1f}%"
                f" • elapsed {int(elapsed//60):02d}:{int(elapsed%60):02d}{eta_txt}",
            )
        except Exception:
            pass

    def message(self, s):
        # Keep MoviePy's verbose messages out of the console/UI unless useful.
        return



def _ass_escape_text(value: str) -> str:
    s = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    return s.replace("\n", r"\N")


def _ass_color(hex_color: str, alpha: int = 0) -> str:
    raw = str(hex_color or "#FFFFFF").strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    try:
        raw = f"{int(raw[:6], 16):06X}"
    except Exception:
        raw = "FFFFFF"
    a = max(0, min(255, int(alpha)))
    return f"&H{a:02X}{raw[4:6]}{raw[2:4]}{raw[0:2]}"


def _ass_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", r"\:")


def _ass_time(sec: float, duration: float) -> str:
    sec = max(0.0, min(float(duration), float(sec)))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _write_fast_ass(path: Path, segments: list, settings: dict,
                    width: int, height: int, duration: float) -> bool:
    """Create a lightweight ASS layer for FFmpeg/libass Turbo Export."""
    family = str(settings.get("font_family", "Noto Sans Khmer") or "Noto Sans Khmer")
    base_size = int(settings.get("subtitle_font_size", 48) or 48)
    font_size = max(10, int(base_size * max(height, 360) / 720))
    title_size = max(10, int(settings.get("title_size", 48) or 48))
    sub_color = _ass_color(settings.get("subtitle_color", "#FFFFFF"))
    title_color = _ass_color(settings.get("title_color", "#FFFFFF"))
    bar_op = max(0.0, min(1.0, float(settings.get("subtitle_bar_opacity", 0.65) or 0.0)))
    back_alpha = int(round((1.0 - bar_op) * 255))
    back_color = _ass_color("#000000", back_alpha)
    border_style = 3 if bar_op > 0.02 else 1

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {max(1, int(width))}",
        f"PlayResY: {max(1, int(height))}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        f"Style: DubbySub,{family},{font_size},{sub_color},{sub_color},"
        f"&H00000000,{back_color},0,0,0,0,100,100,0,0,{border_style},"
        f"{3 if font_size >= 40 else 2},0,5,20,20,10,1",
        f"Style: DubbyTitle,{family},{title_size},{title_color},{title_color},"
        f"&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,0,5,20,20,10,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events = []

    if settings.get("add_subtitles", False):
        x_pct = float(settings.get("subtitle_position_x", 50) or 50) / 100.0
        y_pct = float(settings.get("subtitle_position_y", 88) or 88) / 100.0
        x = int(max(0, min(width, x_pct * width)))
        y = int(max(0, min(height, y_pct * height)))
        for seg in segments:
            value = str(seg.get("translated", "") or "").strip()
            if not value:
                continue
            st = max(0.0, float(seg.get("start", 0) or 0))
            en = min(duration, max(st + 0.05, float(seg.get("end", st + 0.05) or st + 0.05)))
            if en <= st:
                continue
            events.append(
                f"Dialogue: 0,{_ass_time(st,duration)},{_ass_time(en,duration)},DubbySub,,0,0,0,,"
                f"{{\\an5\\pos({x},{y})\\q2}}{_ass_escape_text(value)}"
            )

    if settings.get("add_title", False):
        title = str(settings.get("title_text", "") or "").strip()
        if title:
            if "title_position_x" in settings or "title_position_y" in settings:
                x = int(max(0, min(width, float(settings.get("title_position_x", 50)) / 100.0 * width)))
                y = int(max(0, min(height, float(settings.get("title_position_y", 10)) / 100.0 * height)))
            else:
                pos = str(settings.get("title_position", "top") or "top").lower()
                x = width // 2
                y = 20 if pos == "top" else (height // 2 if pos == "center" else max(20, height - 180))
            events.append(
                f"Dialogue: 10,0:00:00.00,{_ass_time(duration,duration)},DubbyTitle,,0,0,0,,"
                f"{{\\an5\\pos({x},{y})}}{_ass_escape_text(title)}"
            )

    try:
        path.write_text("\n".join(header + events) + "\n", encoding="utf-8-sig")
        return path.exists() and path.stat().st_size > 100
    except Exception:
        return False


def _fast_export_video(worker, video_path: str, audio_path: str, final_path: Path,
                       ass_path: Path, logo_path: Optional[Path], settings: dict,
                       duration: float, width: int, height: int) -> None:
    """Turbo FFmpeg export: crop + libass + logo + NVENC in one video pass."""
    ffmpeg = ffmpeg_executable()
    mode = str(settings.get("export_hwaccel", "Auto") or "Auto").lower()
    preset = str(settings.get("preset", "ultrafast") or "ultrafast")
    crf = int(settings.get("crf", 22) or 22)

    try:
        enc_text = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=False, timeout=8
        ).stdout
    except Exception:
        enc_text = ""

    use_nvenc = "h264_nvenc" in enc_text and (
        mode in ("auto", "gpu", "hardware") or "nvidia" in mode
    )
    if "nvidia" in mode and not use_nvenc:
        raise RuntimeError("NVIDIA NVENC is not available in the installed FFmpeg.")

    filters = []
    aspect = str(settings.get("aspect_ratio", "Original") or "Original")
    ratio = width / max(1, height)

    if aspect == "16:9":
        target = 16 / 9
        if ratio > target:
            cw = int(height * target)
            filters.append(f"crop={cw}:{height}:{max(0,(width-cw)//2)}:0")
        elif ratio < target:
            ch = int(width / target)
            filters.append(f"crop={width}:{ch}:0:{max(0,(height-ch)//2)}")
    elif aspect == "9:16":
        target = 9 / 16
        if ratio > target:
            cw = int(height * target)
            filters.append(f"crop={cw}:{height}:{max(0,(width-cw)//2)}:0")
        elif ratio < target:
            ch = int(width / target)
            filters.append(f"crop={width}:{ch}:0:{max(0,(height-ch)//2)}")
    elif aspect == "1:1":
        side = min(width, height)
        filters.append(f"crop={side}:{side}:{max(0,(width-side)//2)}:{max(0,(height-side)//2)}")

    if settings.get("add_subtitles", False) or (
        settings.get("add_title", False) and str(settings.get("title_text", "")).strip()
    ):
        filters.append(f"subtitles='{_ass_filter_path(ass_path)}'")

    # Build a single FFmpeg filter graph. The logo is pre-sized/opacity-adjusted.
    if logo_path and logo_path.exists():
        x_pct = max(0.0, min(1.0, float(settings.get("logo_position_x", 90) or 90) / 100.0))
        y_pct = max(0.0, min(1.0, float(settings.get("logo_position_y", 10) or 10) / 100.0))
        if "logo_position_x" in settings or "logo_position_y" in settings:
            ox = f"W*{x_pct:.6f}-w/2"
            oy = f"H*{y_pct:.6f}-h/2"
        else:
            pos = str(settings.get("logo_position", "top-right") or "top-right").lower()
            margin = 20
            ox = str(margin) if "left" in pos else f"W-w-{margin}"
            oy = str(margin) if "top" in pos else f"H-h-{margin}"
        main_chain = ",".join(filters) if filters else "null"
        filter_complex = (
            f"[0:v]{main_chain}[v0];"
            f"[1:v]format=rgba[lg];"
            f"[v0][lg]overlay=x={ox}:y={oy}:format=auto[v]"
        )
        input_args = ["-i", str(video_path), "-i", str(logo_path), "-i", str(audio_path)]
        audio_index = 2
        video_map = "[v]"
    else:
        main_chain = ",".join(filters) if filters else "null"
        filter_complex = f"[0:v]{main_chain}[v]"
        input_args = ["-i", str(video_path), "-i", str(audio_path)]
        audio_index = 1
        video_map = "[v]"

    if use_nvenc:
        ep = {"ultrafast":"p1","veryfast":"p2","faster":"p3","fast":"p4","medium":"p5","slow":"p6"}.get(preset, "p1")
        vcodec = ["-c:v", "h264_nvenc", "-preset", ep, "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
        label = "NVIDIA NVENC"
    else:
        cpu_preset = "ultrafast" if preset in ("ultrafast", "veryfast", "faster") else preset
        vcodec = [
            "-c:v", "libx264", "-preset", cpu_preset, "-crf", str(crf),
            "-threads", str(max(2, os.cpu_count() or 6)),
            "-x264-params", "keyint=250:min-keyint=25:ref=1:bframes=0",
        ]
        label = "CPU x264"

    args = [
        "-y", "-hide_banner", "-loglevel", "error",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", video_map,
        "-map", f"{audio_index}:a:0",
        *vcodec,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-af", f"aresample=async=1:first_pts=0,apad,atrim=0:{duration:.3f}",
        "-t", f"{duration:.3f}",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        str(final_path),
    ]
    worker.progress.emit(72, f"Turbo Export • {label} • FFmpeg single pass…")
    result = run_ffmpeg(
        args, quiet=True,
        timeout=float(settings.get("export_mux_timeout", 1800) or 1800)
    )
    if result.returncode != 0 or not final_path.exists() or final_path.stat().st_size < 10_000:
        raise RuntimeError(
            "Turbo FFmpeg export failed. Try Standard (MoviePy) Export or check FFmpeg/NVENC."
        )


class ExportWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str, str, str)
    error = pyqtSignal(str)

    def __init__(self, video_name: str, video_path: str,
                 segments: list, settings: dict):
        super().__init__()
        self.video_name = video_name
        self.video_path = video_path
        self.segments = segments
        self.settings = settings

    def run(self):
        temp = TEMP_DIR / safe_filename(Path(self.video_name).stem)
        temp.mkdir(parents=True, exist_ok=True)
        video = new_audio = final_clip = None
        silent_video = temp / "video_silent.mp4"
        dubbed_wav = temp / "dubbed_48k.wav"
        started_at = time.monotonic()

        try:
            self.progress.emit(1, f"Export started: {self.video_name}")
            if VideoFileClip is None or AudioFileClip is None:
                raise RuntimeError("MoviePy is required for export.")
            if AudioSegment is None:
                raise RuntimeError("pydub is required for export.")

            self.progress.emit(3, "Opening source video...")
            video = VideoFileClip(self.video_path)
            # Prefer ffprobe duration for long / VFR files so the master
            # audio clock matches the real file (MoviePy can under-report).
            probe_dur = ffprobe_duration(self.video_path)
            moviepy_dur = float(video.duration or 0)
            if probe_dur > 0.5:
                # Trust probe when it is close to or longer than MoviePy.
                if moviepy_dur <= 0.05 or abs(probe_dur - moviepy_dur) < 2.0 or probe_dur > moviepy_dur:
                    duration = probe_dur
                else:
                    duration = max(probe_dur, moviepy_dur)
            else:
                duration = moviepy_dur
            if duration <= 0.05:
                raise RuntimeError("Could not read video duration.")

            # -------- TTS / speech track --------
            self.progress.emit(10, "Building clean, synchronized dubbing track...")

            # IMPORTANT: use one master audio clock for the whole video.
            # 48 kHz PCM is the final clock used by FFmpeg/AAC. Every TTS
            # segment is positioned by its absolute subtitle timestamp.
            master_sr = 48000
            total_ms = max(1, int(round(duration * 1000.0)))
            final_audio = AudioSegment.silent(
                duration=total_ms,
                frame_rate=master_sr,
            ).set_channels(2)

            speech_ranges = []
            # Include every segment that claims an audio path; verify files below.
            ordered = sorted(
                [s for s in self.segments if str(s.get("audio_path") or "").strip()],
                key=lambda s: float(s.get("start", 0) or 0),
            )
            missing_tts = 0
            placed = 0
            last_end_ms = 0

            for i, seg in enumerate(ordered):
                audio_path = seg.get("audio_path")
                if not audio_path or not Path(str(audio_path)).exists():
                    missing_tts += 1
                    continue

                try:
                    tts = AudioSegment.from_file(str(audio_path))
                except Exception:
                    missing_tts += 1
                    continue

                # Convert every generated voice to exactly the master format.
                try:
                    tts = tts.set_frame_rate(master_sr).set_channels(2).set_sample_width(2)
                except Exception:
                    pass

                start_s = max(0.0, float(seg.get("start", 0) or 0))
                end_s = max(start_s, float(seg.get("end", start_s) or start_s))
                start_ms = int(round(start_s * 1000.0))
                end_ms = int(round(end_s * 1000.0))

                # Never allow one sentence to push the following sentence.
                if i + 1 < len(ordered):
                    try:
                        next_start_ms = int(round(
                            max(0.0, float(ordered[i + 1].get("start", 0) or 0)) * 1000.0
                        ))
                        if next_start_ms > start_ms + 120:
                            end_ms = min(end_ms, next_start_ms - 20)
                    except Exception:
                        pass

                # Slot = original subtitle / speech window on the video timeline.
                # Always lock voice to that window so it stays on the same frames.
                target_ms = max(160, end_ms - start_ms)

                # Clean first (gentle EQ/normalize), THEN force duration to the
                # original window. Cleaning before fit avoids silence-trim
                # undoing a perfect length match.
                tts = improve_audio_segment(tts)
                tts = fit_tts_to_slot(
                    tts,
                    target_ms,
                    max_stretch=float(self.settings.get("tts_max_stretch", 0.28)),
                    allow_trim=True,
                    pad_short=bool(self.settings.get("tts_pad_short", True)),
                )

                # Apply user gain after duration lock.
                gain_db = float(self.settings.get("voice_gain_db", 0.0))
                if abs(gain_db) > 0.01:
                    tts = tts + gain_db

                if start_ms >= total_ms:
                    continue

                # Absolute frame lock: never speak past this segment's end
                # (or the next line's start / video end).
                hard_end = min(total_ms, end_ms)
                available = max(1, hard_end - start_ms)
                if len(tts) > available:
                    fade = min(28, max(8, available // 16))
                    tts = tts[:available]
                    if len(tts) > fade + 5:
                        tts = tts.fade_out(fade)

                final_audio = final_audio.overlay(tts, position=start_ms)
                placed_end = min(hard_end, start_ms + len(tts))
                speech_ranges.append((start_ms, placed_end))
                last_end_ms = max(last_end_ms, placed_end)
                placed += 1

            expected = sum(1 for s in self.segments if str(s.get("tts_text") or "").strip())
            self.progress.emit(
                22,
                f"Voice track: placed {placed}/{max(expected, len(ordered))} segments "
                f"(missing files: {missing_tts}, last voice @ {last_end_ms/1000:.1f}s / {duration:.1f}s)"
            )
            if placed == 0:
                raise RuntimeError(
                    "No TTS audio files were found for export. "
                    "Run Generate Voice first and wait until all segments succeed."
                )
            if expected and placed < max(1, int(expected * 0.5)):
                self.progress.emit(
                    23,
                    f"Warning: only {placed}/{expected} voice files placed — "
                    "re-run Generate Voice for missing rows, then export again."
                )

            # -------- BGM --------
            if self.settings.get("keep_bgm", True) and video.audio is not None:
                self.progress.emit(28, "Extracting and cleaning original BGM...")

                orig = temp / "original_48k.wav"
                bgm_path = temp / "bgm_48k.wav"

                # Extract directly with FFmpeg instead of MoviePy. This avoids
                # an extra resampling/timestamp conversion on long videos.
                extract_result = run_ffmpeg([
                    "-y",
                    "-i", str(self.video_path),
                    "-vn",
                    "-ac", "2",
                    "-ar", str(master_sr),
                    "-c:a", "pcm_s16le",
                    str(orig),
                ])
                if extract_result.returncode != 0 or not orig.exists():
                    # Safe fallback to the already opened MoviePy audio.
                    video.audio.write_audiofile(
                        str(orig),
                        fps=master_sr,
                        nbytes=2,
                        codec="pcm_s16le",
                        ffmpeg_params=["-ac", "2"],
                        logger=None,
                    )

                ok = extract_bgm_only(
                    str(orig),
                    str(bgm_path),
                    self.settings.get("vocal_strength", "strong"),
                )
                if not ok:
                    shutil.copy2(orig, bgm_path)

                bgm = AudioSegment.from_file(str(bgm_path))
                bgm = bgm.set_frame_rate(master_sr).set_channels(2).set_sample_width(2)

                # Exact master length. Pad if necessary; never extend the video.
                if len(bgm) < total_ms:
                    bgm = bgm + AudioSegment.silent(
                        duration=total_ms - len(bgm),
                        frame_rate=master_sr,
                    )
                else:
                    bgm = bgm[:total_ms]

                vol = float(self.settings.get("bgm_volume", 0.25))
                if vol <= 0:
                    bgm = bgm - 60
                else:
                    bgm = bgm + (20 * math.log10(max(0.001, vol)))

                # Duck BGM around speech (vectorized — avoids O(n²) AudioSegment
                # slice/concat on long videos with many subtitle rows).
                duck = float(self.settings.get("bgm_duck_db", -9))
                if speech_ranges and duck < 0:
                    try:
                        samples = np.array(bgm.get_array_of_samples(), dtype=np.float32)
                        if bgm.channels == 2:
                            samples = samples.reshape((-1, 2))
                        # Gain factor from dB
                        gain = float(10 ** (duck / 20.0))
                        sr_ms = float(bgm.frame_rate) / 1000.0
                        for start, end in speech_ranges:
                            s0 = max(0, int((start - 120) * sr_ms))
                            s1 = min(len(samples), int((end + 180) * sr_ms))
                            if s1 > s0:
                                samples[s0:s1] *= gain
                        samples = np.clip(samples, -32768, 32767).astype(np.int16)
                        bgm = bgm._spawn(
                            samples.reshape(-1).tobytes(),
                            overrides={
                                "channels": bgm.channels,
                                "sample_width": 2,
                                "frame_rate": bgm.frame_rate,
                            },
                        )
                    except Exception:
                        # Fallback to the older per-range path
                        for start, end in speech_ranges:
                            s0 = max(0, start - 120)
                            s1 = min(total_ms, end + 180)
                            if s1 > s0:
                                part = bgm[s0:s1] + duck
                                bgm = bgm[:s0] + part + bgm[s1:]

                final_audio = final_audio.overlay(bgm)

            # Force the final master to EXACTLY the video duration.
            if len(final_audio) < total_ms:
                final_audio += AudioSegment.silent(
                    duration=total_ms - len(final_audio),
                    frame_rate=master_sr,
                )
            final_audio = final_audio[:total_ms].set_frame_rate(master_sr).set_channels(2).set_sample_width(2)

            final_audio.export(
                str(dubbed_wav),
                format="wav",
                parameters=[
                    "-ar", str(master_sr),
                    "-ac", "2",
                    "-sample_fmt", "s16",
                ],
            )

            # -------- TURBO EXPORT PATH --------
            # Skip MoviePy CompositeVideoClip completely. FFmpeg/libass handles
            # crop + subtitle/title + logo + encoding in one pass.
            export_mode = str(
                self.settings.get("export_mode", "Turbo (FFmpeg + NVENC)") or ""
            )
            if export_mode.startswith("Turbo"):
                self.progress.emit(50, "Preparing Turbo FFmpeg export…")
                out_dir = resolve_output_dir(self.settings)
                final_name = f"DubbyAI_{safe_filename(Path(self.video_name).stem)}.mp4"
                final_path = out_dir / final_name

                src_w = int(getattr(video, "w", 0) or 0)
                src_h = int(getattr(video, "h", 0) or 0)
                if src_w <= 0 or src_h <= 0:
                    raise RuntimeError("Could not read source video dimensions.")

                ass_path = temp / "dubby_fast.ass"
                if not _write_fast_ass(
                    ass_path, self.segments, self.settings,
                    src_w, src_h, duration
                ):
                    raise RuntimeError("Could not prepare Turbo subtitle/title layer.")

                logo_fast = None
                if self.settings.get("add_logo", False):
                    logo_path = self.settings.get("logo_path")
                    if logo_path and Path(logo_path).exists():
                        pil_logo = Image.open(logo_path).convert("RGBA")
                        logo_w = int(self.settings.get("logo_width", 140) or 140)
                        ratio = logo_w / max(1, pil_logo.width)
                        logo_h = max(1, int(pil_logo.height * ratio))
                        pil_logo = pil_logo.resize(
                            (logo_w, logo_h), Image.Resampling.LANCZOS
                        )
                        opacity = float(self.settings.get("logo_opacity", 0.85) or 0.85)
                        if opacity < 1:
                            alpha = pil_logo.getchannel("A")
                            alpha = alpha.point(
                                lambda p: int(p * max(0.0, min(1.0, opacity)))
                            )
                            pil_logo.putalpha(alpha)
                        logo_fast = temp / "logo_fast.png"
                        pil_logo.save(str(logo_fast))

                self.progress.emit(65, "Rendering subtitles/title with FFmpeg…")
                _fast_export_video(
                    self, self.video_path, str(dubbed_wav), final_path,
                    ass_path, logo_fast, self.settings,
                    duration, src_w, src_h
                )

                # Write SRT without running the MoviePy path.
                srt_path = out_dir / (
                    f"{safe_filename(Path(self.video_name).stem)}_"
                    f"{self.settings.get('target_lang', 'translated')}.srt"
                )
                if srt is not None:
                    subs = []
                    for seg in self.segments:
                        translated = seg.get("translated", "").strip()
                        if not translated:
                            continue
                        subs.append(
                            srt.Subtitle(
                                index=int(seg["index"]),
                                start=srt.timedelta(seconds=float(seg["start"])),
                                end=srt.timedelta(seconds=float(seg["end"])),
                                content=translated
                            )
                        )
                    srt_path.write_text(srt.compose(subs), encoding="utf-8")

                elapsed = time.monotonic() - started_at
                self.progress.emit(
                    100,
                    f"Turbo Export complete: {Path(final_path).name} • "
                    f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
                )
                self.finished.emit(
                    self.video_name, str(final_path), str(srt_path)
                )
                return

            # -------- Visual layers --------
            self.progress.emit(50, "Building video layers...")
            # Do NOT attach the WAV to MoviePy. MoviePy's audio mux/resampling
            # path can introduce small clock differences on long/VFR videos.
            # We render a silent video first, then mux dubbed_48k.wav with
            # FFmpeg using one exact audio clock.
            new_audio = None
            base = video.set_audio(None)
            clips = [base]

            # Aspect ratio crop.
            aspect = self.settings.get("aspect_ratio", "Original")
            if aspect != "Original":
                target_ratio = {"16:9": 16/9, "9:16": 9/16, "1:1": 1.0}.get(aspect)
                if target_ratio and video.h and video.w:
                    current = video.w / video.h
                    if current > target_ratio:
                        new_w = int(video.h * target_ratio)
                        x = int((video.w - new_w) / 2)
                        base = base.crop(x1=x, x2=x + new_w)
                    elif current < target_ratio:
                        new_h = int(video.w / target_ratio)
                        y = int((video.h - new_h) / 2)
                        base = base.crop(y1=y, y2=y + new_h)
                    clips = [base]

            width = int(video.w)
            height = int(video.h)

            # Subtitle bar + text — free position (X%, Y%) anywhere on the frame.
            # Bar uses Premiere-style Gaussian blur plate when opacity > 0.
            if self.settings.get("add_subtitles", False):
                self.progress.emit(52, "Rendering Unicode subtitles...")
                bar_h = int(self.settings.get("subtitle_bar_height", 140))
                bar_op = float(self.settings.get("subtitle_bar_opacity", 0.65))
                blur_radius = int(self.settings.get("subtitle_blur_radius", 18))
                pos_x = float(self.settings.get("subtitle_position_x", 50))
                pos_y = float(self.settings.get("subtitle_position_y", 88))
                # Scale font with video height so Khmer stays readable on 1080p+.
                family = self.settings.get("font_family", "Noto Sans Khmer")
                base_size = int(self.settings.get("subtitle_font_size", 48))
                font_size = max(10, int(base_size * max(height, 360) / 720))
                # Wider stroke for Khmer outlines on busy backgrounds.
                stroke_w = 3 if font_size >= 40 else 2

                text_w = max(120, int(width * 0.92))
                text_h = max(48, bar_h)

                # Pre-render ONE soft bar (Gaussian blur is expensive). Reuse
                # the same plate for every subtitle row — major export speedup.
                shared_bar_arr = None
                shared_bar_w = shared_bar_h = 0
                shared_bx = shared_by = 0
                if bar_op > 0.02:
                    bar_img = gaussian_blur_bar(
                        text_w + 28, text_h + 16,
                        opacity=bar_op, radius=blur_radius,
                    )
                    if bar_img is not None:
                        shared_bar_arr = np.array(bar_img)
                        shared_bar_w, shared_bar_h = bar_img.width, bar_img.height
                        shared_bx = int(max(0, min(
                            width - shared_bar_w,
                            pos_x / 100.0 * width - shared_bar_w / 2,
                        )))
                        shared_by = int(max(0, min(
                            height - shared_bar_h,
                            pos_y / 100.0 * height - shared_bar_h / 2,
                        )))

                sub_count = 0
                for seg in self.segments:
                    text = seg.get("translated", "").strip()
                    if not text:
                        continue
                    img = text_image(
                        text, text_w, text_h,
                        font_size, family,
                        self.settings.get("subtitle_color", "#FFFFFF"),
                        stroke=stroke_w
                    )
                    if img is None:
                        continue
                    dur_seg = max(0.2, float(seg["end"]) - float(seg["start"]))
                    start_s = float(seg["start"])

                    if shared_bar_arr is not None:
                        bar_clip = ImageClip(shared_bar_arr).set_start(
                            start_s
                        ).set_duration(dur_seg).set_position((shared_bx, shared_by))
                        clips.append(bar_clip)
                    elif bar_op > 0.02 and ColorClip is not None:
                        bar = ColorClip(
                            size=(text_w + 20, text_h + 10), color=(0, 0, 0)
                        ).set_opacity(bar_op)
                        bar = bar.set_start(start_s).set_duration(dur_seg)
                        bx = int(max(0, min(width - (text_w + 20),
                                            pos_x / 100.0 * width - (text_w + 20) / 2)))
                        by = int(max(0, min(height - (text_h + 10),
                                            pos_y / 100.0 * height - (text_h + 10) / 2)))
                        bar = bar.set_position((bx, by))
                        clips.append(bar)

                    arr = np.array(img)
                    tx = int(max(0, min(width - img.width,
                                        pos_x / 100.0 * width - img.width / 2)))
                    ty = int(max(0, min(height - img.height,
                                        pos_y / 100.0 * height - img.height / 2)))
                    ic = ImageClip(arr).set_start(start_s).set_duration(dur_seg).set_position((tx, ty))
                    clips.append(ic)
                    sub_count += 1
                self.progress.emit(60, f"Composited {sub_count} subtitle overlays...")

            # Title.
            title = self.settings.get("title_text", "").strip()
            if self.settings.get("add_title", False) and title:
                img = text_image(
                    title, width, 160,
                    int(self.settings.get("title_size", 48)),
                    self.settings.get("font_family", "Noto Sans Khmer"),
                    self.settings.get("title_color", "#FFFFFF"),
                    stroke=3
                )
                if img is not None:
                    tc = ImageClip(np.array(img)).set_duration(duration)
                    if "title_position_x" in self.settings or "title_position_y" in self.settings:
                        x_pct = float(self.settings.get("title_position_x", 50)) / 100.0
                        y_pct = float(self.settings.get("title_position_y", 10)) / 100.0
                        tx = int(max(0, min(width - img.width, x_pct * width - img.width / 2)))
                        ty = int(max(0, min(height - img.height, y_pct * height - img.height / 2)))
                        tc = tc.set_position((tx, ty))
                    else:
                        p = self.settings.get("title_position", "top")
                        if p == "center":
                            tc = tc.set_position("center")
                        elif p == "bottom":
                            tc = tc.set_position(("center", height - 180))
                        else:
                            tc = tc.set_position(("center", 20))
                    clips.append(tc)

            # Logo.
            logo_path = self.settings.get("logo_path")
            if self.settings.get("add_logo", False) and logo_path and Path(logo_path).exists():
                pil_logo = Image.open(logo_path).convert("RGBA")
                logo_w = int(self.settings.get("logo_width", 140))
                ratio = logo_w / max(1, pil_logo.width)
                logo_h = max(1, int(pil_logo.height * ratio))
                pil_logo = pil_logo.resize(
                    (logo_w, logo_h), Image.Resampling.LANCZOS
                )
                opacity = float(self.settings.get("logo_opacity", 0.85))
                if opacity < 1:
                    alpha = pil_logo.getchannel("A")
                    alpha = alpha.point(lambda p: int(p * opacity))
                    pil_logo.putalpha(alpha)

                logo_file = temp / "logo.png"
                pil_logo.save(str(logo_file))
                lc = ImageClip(str(logo_file)).set_duration(duration)

                if "logo_position_x" in self.settings or "logo_position_y" in self.settings:
                    x_pct = float(self.settings.get("logo_position_x", 90)) / 100.0
                    y_pct = float(self.settings.get("logo_position_y", 10)) / 100.0
                    x = int(max(0, min(width - logo_w, x_pct * width - logo_w / 2)))
                    y = int(max(0, min(height - logo_h, y_pct * height - logo_h / 2)))
                    xy = (x, y)
                else:
                    margin = 20
                    pos = self.settings.get("logo_position", "top-right")
                    if pos == "top-left":
                        xy = (margin, margin)
                    elif pos == "bottom-left":
                        xy = (margin, height - logo_h - margin)
                    elif pos == "bottom-right":
                        xy = (width - logo_w - margin, height - logo_h - margin)
                    else:
                        xy = (width - logo_w - margin, margin)
                lc = lc.set_position(xy)
                clips.append(lc)

            self.progress.emit(70, "Compositing...")
            final_clip = CompositeVideoClip(clips, size=(width, height))

            final_name = f"DubbyAI_{safe_filename(Path(self.video_name).stem)}.mp4"
            out_dir = resolve_output_dir(self.settings)
            final_path = out_dir / final_name

            # Hardware-aware video encoding. Auto-detect the GPU encoder supported by FFmpeg.
            preset=self.settings.get("preset","veryfast"); crf=int(self.settings.get("crf",20))
            threads=max(2,int(self.settings.get("threads",max(4,(os.cpu_count() or 8)))))
            hw_mode=str(self.settings.get("export_hwaccel","Auto") or "Auto").lower()
            def has_encoder(name):
                try:
                    r=subprocess.run([ffmpeg_executable(),"-hide_banner","-encoders"],capture_output=True,text=True,timeout=8,check=False)
                    return name in (r.stdout+r.stderr)
                except Exception:return False
            if hw_mode in ("auto","gpu","hardware"): candidates=[("NVIDIA","h264_nvenc"),("AMD","h264_amf"),("Intel","h264_qsv")]
            elif "nvidia" in hw_mode: candidates=[("NVIDIA","h264_nvenc")]
            elif "amd" in hw_mode or "amf" in hw_mode: candidates=[("AMD","h264_amf")]
            elif "intel" in hw_mode or "qsv" in hw_mode: candidates=[("Intel","h264_qsv")]
            else: candidates=[]
            hw_vendor=encoder=None
            for vendor,enc in candidates:
                if has_encoder(enc): hw_vendor,encoder=vendor,enc; break
            if encoder=="h264_nvenc":
                ep={"ultrafast":"p1","veryfast":"p2","faster":"p3","fast":"p4","medium":"p5","slow":"p6"}.get(preset,"p2")
                ffmpeg_params=["-preset",ep,"-rc","vbr","-cq",str(crf),"-b:v","0","-pix_fmt","yuv420p","-movflags","+faststart"]; export_preset=ep; label=f"{hw_vendor} GPU encoder"
            elif encoder=="h264_amf":
                q={"ultrafast":"speed","veryfast":"speed","faster":"speed","fast":"balanced","medium":"balanced","slow":"quality"}.get(preset,"speed")
                ffmpeg_params=["-quality",q,"-rc","vbr_peak","-qp_i",str(max(16,crf)),"-qp_p",str(max(18,crf+2)),"-pix_fmt","yuv420p","-movflags","+faststart"]; export_preset=preset; label=f"{hw_vendor} GPU encoder"
            elif encoder=="h264_qsv":
                qp={"ultrafast":"veryfast","veryfast":"veryfast","faster":"faster","fast":"fast","medium":"medium","slow":"slow"}.get(preset,"veryfast")
                ffmpeg_params=["-global_quality",str(crf+2),"-look_ahead","0","-pix_fmt","nv12","-movflags","+faststart"]; export_preset=qp; label=f"{hw_vendor} GPU encoder"
            else:
                encoder="libx264"; ffmpeg_params=["-crf",str(crf),"-preset",preset,"-threads",str(threads),"-x264-params","keyint=250:min-keyint=25:ref=1:bframes=0","-movflags","+faststart"]; export_preset=preset; label="CPU encoder"
            self.progress.emit(75,f"Rendering with {label}...")

            # Render the visual stream with a fixed frame-rate clock and NO audio.
            src_fps = float(getattr(video, "fps", None) or 25.0)
            if src_fps < 1 or src_fps > 120:
                src_fps = 25.0
            # Preserve source FPS unless the user explicitly sets export_max_fps.
            max_fps_setting=float(self.settings.get("export_max_fps",0) or 0)
            encode_fps=src_fps if max_fps_setting<=0 else min(src_fps,max_fps_setting)
            if encode_fps<1: encode_fps=25.0

            # IMPORTANT: logger=None made long exports look frozen. Feed actual
            # frame progress back to the Qt progress bar instead.
            if ProgressBarLogger is not None:
                export_logger = _ExportProgressLogger(self, duration)
            else:
                export_logger = None

            final_clip.write_videofile(
                str(silent_video),
                fps=encode_fps,
                codec=encoder,
                audio=False,
                preset=export_preset,
                threads=threads,
                ffmpeg_params=ffmpeg_params,
                logger=export_logger,
                verbose=False,
            )

            if not silent_video.exists() or silent_video.stat().st_size < 10_000:
                raise RuntimeError(
                    "Video rendering finished without creating a valid temporary MP4."
                )
            self.progress.emit(91, "Video render finished. Preparing final audio/video mux…")

            # FINAL A/V MUX -------------------------------------------------
            # This is the important long-video sync fix:
            # - video and voice are separate streams
            # - WAV is already exactly the source-video duration
            # - FFmpeg maps both from timestamp zero
            # - AAC is encoded once, at 48 kHz
            # - video is copied, so no second video encode can change timing
            self.progress.emit(92, "Muxing clean voice + video with FFmpeg...")

            # Do NOT use -shortest alone: if the silent MP4 is slightly shorter
            # than the master WAV (MoviePy frame rounding), voice after that
            # point was being dropped on long videos. Force a shared duration
            # and pad/trim audio to match the real source length.
            mux_dur = f"{duration:.3f}"
            mux_result = run_ffmpeg([
                "-y",
                "-fflags", "+genpts",
                "-i", str(silent_video),
                "-i", str(dubbed_wav),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",
                "-af", f"aresample=async=1:first_pts=0,apad,atrim=0:{duration:.3f}",
                "-t", mux_dur,
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(final_path),
            ], quiet=True, timeout=float(self.settings.get("export_mux_timeout", 1800) or 1800))

            if mux_result.returncode != 0 or not final_path.exists() or final_path.stat().st_size < 10_000:
                raise RuntimeError(
                    "FFmpeg could not mux the final video and clean voice track. "
                    "Check that FFmpeg is installed and available."
                )

            # SRT.
            srt_path = out_dir / (
                f"{safe_filename(Path(self.video_name).stem)}_"
                f"{self.settings.get('target_lang', 'translated')}.srt"
            )
            if srt is not None:
                subs = []
                for seg in self.segments:
                    translated = seg.get("translated", "").strip()
                    if not translated:
                        continue
                    subs.append(
                        srt.Subtitle(
                            index=int(seg["index"]),
                            start=srt.timedelta(seconds=float(seg["start"])),
                            end=srt.timedelta(seconds=float(seg["end"])),
                            content=translated
                        )
                    )
                srt_path.write_text(srt.compose(subs), encoding="utf-8")

            elapsed = time.monotonic() - started_at
            self.progress.emit(
                100,
                f"Export complete: {Path(final_path).name} • "
                f"{int(elapsed//60):02d}:{int(elapsed%60):02d}",
            )
            self.finished.emit(
                self.video_name, str(final_path), str(srt_path)
            )

        except Exception as e:
            # Never leave a half-written final MP4 that looks valid to the next run.
            for partial in (silent_video,):
                try:
                    if partial.exists():
                        partial.unlink()
                except Exception:
                    pass
            elapsed = time.monotonic() - started_at
            self.error.emit(
                f"{self.video_name}: {e} "
                f"(elapsed {int(elapsed//60):02d}:{int(elapsed%60):02d})"
            )
        finally:
            for obj in (video, new_audio, final_clip):
                try:
                    if obj is not None:
                        obj.close()
                except Exception:
                    pass


# ------------------------- API settings dialog -------------------------

class SecretKeysWidget(QWidget):
    """Multi-line API key editor that stays masked by default (show/hide)."""

    def __init__(self, keys_text: str = "", placeholder: str = "", parent=None):
        super().__init__(parent)
        self._real_text = str(keys_text or "")
        self._visible = False
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.lbl = QLabel("••••  Keys hidden")
        self.lbl.setStyleSheet("color: #9aa4b2; font-size: 12px;")
        row.addWidget(self.lbl, 1)

        self.btn_toggle = QPushButton("Show")
        self.btn_toggle.setFixedWidth(72)
        self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.clicked.connect(self._toggle)
        row.addWidget(self.btn_toggle)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedWidth(64)
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.clicked.connect(self._clear)
        row.addWidget(self.btn_clear)
        layout.addLayout(row)

        self.edit = QTextEdit()
        self.edit.setMaximumHeight(96)
        self.edit.setPlaceholderText(placeholder or "Paste one or more API keys, one per line")
        self.edit.setStyleSheet(
            "QTextEdit { background: #1a1f2b; color: #e8ecf4; border: 1px solid #3a4558; "
            "border-radius: 8px; padding: 8px; font-family: Consolas, 'Courier New', monospace; }"
        )
        self.edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.edit)
        self._refresh_display()

    def _mask_text(self, text: str) -> str:
        lines = []
        for line in (text or "").splitlines():
            s = line.strip()
            if not s:
                continue
            if len(s) <= 8:
                lines.append("•" * max(6, len(s)))
            else:
                lines.append(s[:4] + "•" * min(24, len(s) - 8) + s[-4:])
        return "\n".join(lines)

    def _refresh_display(self):
        self._updating = True
        try:
            if self._visible:
                self.edit.setPlainText(self._real_text)
                self.edit.setReadOnly(False)
                self.btn_toggle.setText("Hide")
                n = len(split_api_keys(self._real_text))
                self.lbl.setText(f"🔓  {n} key(s) visible — keep private")
            else:
                self.edit.setPlainText(self._mask_text(self._real_text))
                self.edit.setReadOnly(True)
                self.btn_toggle.setText("Show")
                n = len(split_api_keys(self._real_text))
                self.lbl.setText(f"🔒  {n} key(s) saved & hidden" if n else "🔒  No keys yet")
        finally:
            self._updating = False

    def _toggle(self):
        self._visible = not self._visible
        self._refresh_display()

    def _clear(self):
        self._real_text = ""
        self._visible = True
        self._refresh_display()

    def _on_text_changed(self):
        if self._updating or not self._visible:
            return
        self._real_text = self.edit.toPlainText()
        n = len(split_api_keys(self._real_text))
        self.lbl.setText(f"🔓  {n} key(s) visible — keep private")

    def keys_text(self) -> str:
        if self._visible:
            self._real_text = self.edit.toPlainText()
        return self._real_text


class ApiSettingsDialog(QDialog):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("API & AI Settings")
        self.resize(780, 760)
        settings = settings or {}
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = QLabel("API keys are stored locally in user_settings.json next to the app.\n"
                        "They stay available after you close and reopen Dubby tomorrow.")
        header.setWordWrap(True)
        header.setStyleSheet("color: #9aa4b2; padding: 4px 2px;")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        form = QFormLayout(body)
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # --- Gemini keys (secret) ---
        keys = settings.get("gemini_api_keys") or settings.get("gemini_api_key", "")
        if isinstance(keys, list):
            keys_text = "\n".join(str(k) for k in keys if str(k).strip())
        else:
            keys_text = str(keys or "")
        self.gemini = SecretKeysWidget(
            keys_text,
            placeholder="Paste Gemini API key(s), one per line — never share these publicly"
        )
        form.addRow("Gemini API Keys:", self.gemini)

        self.gemini_transcribe_model = QComboBox()
        self.gemini_transcribe_model.addItems([
            "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
            "gemini-2.5-flash", "gemini-2.5-flash-lite",
        ])
        self.gemini_transcribe_model.setCurrentText(
            settings.get("gemini_transcribe_model", "gemini-3.7-flash")
        )
        form.addRow("Gemini Transcription Model:", self.gemini_transcribe_model)

        self.gemini_activity_model = QComboBox()
        self.gemini_activity_model.addItems([
            "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
            "gemini-2.5-flash", "gemini-2.5-flash-lite",
        ])
        self.gemini_activity_model.setCurrentText(
            settings.get("gemini_activity_model", settings.get("gemini_transcribe_model", "gemini-3.7-flash"))
        )
        form.addRow("Gemini Activity Model:", self.gemini_activity_model)

        # --- OpenAI (already password) ---
        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key.setText(settings.get("openai_api_key", ""))
        self.openai_key.setPlaceholderText("sk-... (hidden)")
        form.addRow("OpenAI API Key:", self.openai_key)

        self.openai_model = QComboBox()
        self.openai_model.addItems([
            "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview",
            "gpt-3.5-turbo-1106", "gpt-4o", "gpt-4o-mini",
        ])
        self.openai_model.setCurrentText(settings.get("openai_model", "gpt-3.5-turbo"))
        form.addRow("OpenAI Model:", self.openai_model)

        self.translation = QComboBox()
        self.translation.addItems(["Google Translate", "OpenAI ChatGPT", "Gemini (API)"])
        self.translation.setCurrentText(settings.get("translation_provider", "Google Translate"))
        form.addRow("Translation Provider:", self.translation)

        self.translation_prompt = QTextEdit()
        self.translation_prompt.setPlainText(settings.get(
            "translation_prompt",
            "You are a professional translator. Translate from Chinese to the target language. "
            "Preserve cultural context, idioms, and emotional tone. "
            "Keep proper names in their original form unless they have common translations. "
            "Maintain the natural flow of the original dialogue.",
        ))
        self.translation_prompt.setMaximumHeight(80)
        form.addRow("Translation Prompt:", self.translation_prompt)

        self.tts_provider = QComboBox()
        self.tts_provider.addItems(["Edge-TTS", "Google Gemini TTS", "VoxCPM2 Local", "VoxCPM2 Local CPU"])
        self.tts_provider.setCurrentText(settings.get("tts_provider", "Edge-TTS"))
        form.addRow("Voice Provider:", self.tts_provider)

        self.voxcpm_device = QComboBox()
        self.voxcpm_device.addItems(["Auto", "CPU", "NVIDIA CUDA"])
        self.voxcpm_device.setCurrentText(settings.get("voxcpm_device", "Auto"))
        form.addRow("VoxCPM2 Device:", self.voxcpm_device)

        self.hf_token = QLineEdit(settings.get("huggingface_token", ""))
        self.hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token.setPlaceholderText("hf_... (hidden)")
        form.addRow("Hugging Face Token:", self.hf_token)

        self.min_speakers = QSpinBox()
        self.min_speakers.setRange(0, 20)
        self.min_speakers.setValue(int(settings.get("min_speakers", 0) or 0))
        form.addRow("Min Speakers (0=auto):", self.min_speakers)

        self.max_speakers = QSpinBox()
        self.max_speakers.setRange(0, 20)
        self.max_speakers.setValue(int(settings.get("max_speakers", 0) or 0))
        form.addRow("Max Speakers (0=auto):", self.max_speakers)

        self.voxcpm_style = QTextEdit(settings.get(
            "voxcpm_style",
            "Natural conversational voice, clear pronunciation, realistic emotion.",
        ))
        self.voxcpm_style.setMaximumHeight(70)
        form.addRow("VoxCPM2 Voice Style:", self.voxcpm_style)

        self.tts_model = QComboBox()
        self.tts_model.addItems([
            "gemini-3.1-flash-tts-preview",
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts",
        ])
        self.tts_model.setCurrentText(settings.get("gemini_tts_model", "gemini-3.1-flash-tts-preview"))
        form.addRow("Gemini TTS Model:", self.tts_model)

        self.voice = QComboBox()
        self.voice.addItems(GEMINI_TTS_VOICES)
        self.voice.setCurrentText(settings.get("gemini_default_voice", "Kore"))
        form.addRow("Gemini Default Voice:", self.voice)

        self.style = QTextEdit(settings.get(
            "gemini_tts_style",
            "Professional human voice-over. Clear studio narration, natural conversational tone, "
            "steady pitch, clean pronunciation, subtle emotion, natural pauses. "
            "Do not sound robotic, exaggerated, theatrical, breathy, or synthetic.",
        ))
        self.style.setMaximumHeight(90)
        form.addRow("Gemini Voice Direction:", self.style)

        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.0, 10.0)
        self.interval.setSingleStep(0.05)
        self.interval.setDecimals(2)
        self.interval.setValue(float(settings.get("gemini_request_interval", 0.65)))
        self.interval.setSuffix(" sec/request")
        form.addRow("Gemini Request Delay:", self.interval)

        self.gpu = QComboBox()
        self.gpu.addItems(["Auto", "CPU", "NVIDIA CUDA"])
        self.gpu.setCurrentText(settings.get("gpu_mode", "Auto"))
        form.addRow("Hardware:", self.gpu)

        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        info = QLabel(
            "Tips: Gemini keys are masked until you click Show. "
            "Settings are auto-saved to user_settings.json when you press Save.\n"
            "If Gemini transcription closes the app, try gemini-3.6-flash / gemini-2.5-flash-lite "
            "or switch to Local Whisper + Analyze."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #8b95a8; font-size: 12px;")
        layout.addWidget(info)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self):
        keys = split_api_keys(self.gemini.keys_text())
        return {
            "gemini_api_key": keys[0] if keys else "",
            "gemini_api_keys": keys,
            "gemini_transcribe_model": self.gemini_transcribe_model.currentText(),
            "gemini_activity_model": self.gemini_activity_model.currentText(),
            "openai_api_key": self.openai_key.text().strip(),
            "openai_model": self.openai_model.currentText(),
            "translation_prompt": self.translation_prompt.toPlainText().strip(),
            "tts_provider": self.tts_provider.currentText(),
            "voxcpm_device": self.voxcpm_device.currentText(),
            "huggingface_token": self.hf_token.text().strip(),
            "min_speakers": self.min_speakers.value(),
            "max_speakers": self.max_speakers.value(),
            "voxcpm_style": self.voxcpm_style.toPlainText().strip(),
            "gemini_tts_model": self.tts_model.currentText(),
            "gemini_default_voice": self.voice.currentText(),
            "gemini_tts_style": self.style.toPlainText().strip(),
            "gemini_request_interval": self.interval.value(),
            "translation_provider": self.translation.currentText(),
            "gpu_mode": self.gpu.currentText(),
        }


# ------------------------- Main editor -------------------------



class TimelineEditorWidget(QWidget):
    """Interactive visual timeline for transcript/voice editing.

    The widget is intentionally dependency-light: it paints video, subtitle,
    voice and BGM tracks with QPainter, supports click-to-seek, segment
    selection and playhead dragging, and exposes signals to the main editor.
    """
    segmentSelected = pyqtSignal(int)
    seekRequested = pyqtSignal(float)
    segmentEdited = pyqtSignal(int)
    playVoiceRequested = pyqtSignal(int)  # play generated TTS for this row

    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments = []
        self.duration = 1.0
        self.playhead = 0.0
        self.selected = -1
        self.zoom = 1.0
        self.dragging_playhead = False
        self.drag_mode = None
        self.drag_row = -1
        self.drag_anchor_time = 0.0
        self.drag_original_start = 0.0
        self.drag_original_end = 0.0
        self.setMinimumHeight(250)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.ArrowCursor)  # always start with normal mouse
        self.setToolTip(
            "Click subtitle block to select/trim.  "
            "Double-click VOICE block (or press V) to preview generated TTS."
        )

    def set_data(self, segments, duration, playhead=0.0):
        self.segments = segments or []
        self.duration = max(float(duration or 0.0), 0.1)
        self.playhead = max(0.0, min(float(playhead or 0.0), self.duration))
        self.update()

    def set_playhead(self, seconds):
        self.playhead = max(0.0, min(float(seconds or 0.0), self.duration))
        self.update()

    def set_selected(self, row):
        self.selected = int(row) if row is not None else -1
        self.update()

    def set_zoom(self, value):
        self.zoom = max(0.5, min(float(value), 3.0))
        self.update()

    def _left(self):
        return 72

    def _track_width(self):
        return max(100.0, (self.width() - self._left() - 18) * self.zoom)

    def _x_for_time(self, t):
        return self._left() + (float(t) / self.duration) * self._track_width()

    def _time_for_x(self, x):
        return max(0.0, min(self.duration, (x - self._left()) / self._track_width() * self.duration))

    def _fmt(self, sec):
        sec=max(0,int(sec))
        return f"{sec//60:02d}:{sec%60:02d}"

    def paintEvent(self, event):
        p=QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0b0f19"))

        left=self._left()
        top=34
        track_h=42
        gap=6
        # Premiere-style track labels with accent colors
        labels=[
            ("VIDEO",     "#475569"),
            ("SUBTITLES", "#3b82f6"),
            ("VOICE",     "#8b5cf6"),
            ("BGM",       "#10b981"),
        ]

        # Time ruler
        p.setPen(QColor("#64748b"))
        interval=5 if self.duration <= 120 else 10 if self.duration <= 300 else 30
        t=0
        while t <= self.duration + 0.01:
            x=self._x_for_time(t)
            p.drawLine(int(x), 16, int(x), 28)
            p.drawText(int(x)+3, 14, self._fmt(t))
            t += interval

        # Track backgrounds (distinct layers)
        for i,(label,color) in enumerate(labels):
            y=top+i*(track_h+gap)
            # track header label
            p.setPen(QColor(color))
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            p.drawText(6, y+26, label)
            # track lane
            p.setPen(QColor("#1e293b"))
            p.setBrush(QColor("#111827"))
            p.drawRoundedRect(int(left), y, int(self._track_width()), track_h, 4, 4)
            # subtle left accent bar
            p.setBrush(QColor(color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(int(left), y, 3, track_h, 2, 2)

        # Video strip: frame-like blocks
        n=max(1,min(24,int(self.duration/2)+1))
        for i in range(n):
            a=self.duration*i/n; b=self.duration*(i+1)/n
            x1=self._x_for_time(a); x2=self._x_for_time(b)
            p.setBrush(QColor("#334155" if i%2 else "#475569"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(int(x1)+1,top+2,max(2,int(x2-x1)-2),track_h-4)
            # faux thumbnail frame line
            p.setPen(QColor("#64748b"))
            p.drawRect(int(x1)+3,top+6,max(1,int(x2-x1)-6),track_h-12)

        # Subtitle and voice blocks from actual segments
        for row,seg in enumerate(self.segments):
            try:
                a=float(seg.get('start',0)); b=float(seg.get('end',a+0.1))
            except Exception:
                continue
            a=max(0,min(self.duration,a)); b=max(a+0.03,min(self.duration,b))
            x1=self._x_for_time(a); x2=self._x_for_time(b)
            sy=top+(track_h+gap)          # SUBTITLES track
            vy=top+2*(track_h+gap)        # VOICE track
            selected=(row==self.selected)
            p.setBrush(QColor("#3b82f6" if selected else "#2563eb"))
            p.setPen(QColor("#93c5fd") if selected else QColor("#1e40af"))
            block_w = max(5, int(x2-x1))
            p.drawRoundedRect(int(x1),sy,block_w,track_h,4,4)
            # Explicit left/right timing handles
            if selected:
                p.setBrush(QColor("#dbeafe"))
                p.drawRect(int(x1), sy + 7, 4, track_h - 14)
                p.drawRect(int(x2) - 4, sy + 7, 4, track_h - 14)
            has_audio = bool(seg.get("audio_path") and Path(str(seg.get("audio_path"))).exists())
            if has_audio:
                p.setBrush(QColor("#a78bfa" if selected else "#7c3aed"))
                p.setPen(QColor("#ede9fe") if selected else QColor("#5b21b6"))
            else:
                # Empty voice slot — dimmed so the user knows TTS is missing.
                p.setBrush(QColor("#4c1d95" if selected else "#2e1065"))
                p.setPen(QColor("#6b7280"))
            p.drawRoundedRect(int(x1), vy, max(5, int(x2 - x1)), track_h, 4, 4)
            # Small waveform-style ticks when audio is present
            if has_audio and (x2 - x1) > 28:
                p.setPen(QColor("#ddd6fe"))
                mid_y = vy + track_h // 2
                step = max(4, int((x2 - x1) / 12))
                xi = int(x1) + 6
                while xi < int(x2) - 4:
                    hgt = 4 + ((xi * 7) % 14)
                    p.drawLine(xi, mid_y - hgt // 2, xi, mid_y + hgt // 2)
                    xi += step

            txt=str(seg.get('translated') or seg.get('original') or '')
            txt=' '.join(txt.split())
            if len(txt)>22: txt=txt[:22]+'…'
            p.setPen(QColor("#f8fafc"))
            p.setFont(QFont("Segoe UI", 9))
            if x2-x1>35:
                p.drawText(int(x1)+6,sy+25,txt)
                speaker=str(seg.get('speaker','Speaker 1'))
                label = ("▶ " if has_audio else "○ ") + speaker
                p.setPen(QColor("#ede9fe") if has_audio else QColor("#9ca3af"))
                p.drawText(int(x1)+6, vy+25, label)

        # Playhead (Premiere-style red needle)
        x=self._x_for_time(self.playhead)
        p.setPen(QPen(QColor("#f43f5e"), 2))
        p.drawLine(int(x), 12, int(x), self.height()-6)
        p.setBrush(QColor("#fb7185"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon([QPoint(int(x)-7,12), QPoint(int(x)+7,12), QPoint(int(x),24)])
        p.end()

    def _row_at(self, x, y):
        # Hit-test only on the SUBTITLES track (same geometry as paintEvent)
        track_h, gap, top = 42, 6, 34
        sy = top + (track_h + gap)
        for row, seg in enumerate(self.segments):
            try:
                a=float(seg.get("start",0)); b=float(seg.get("end",a+0.1))
            except Exception:
                continue
            x1=self._x_for_time(a); x2=self._x_for_time(b)
            if sy <= y <= sy + track_h and x1 <= x <= x2:
                return row, x1, x2
        return -1, 0, 0

    def mousePressEvent(self,event):
        if event.button()!=Qt.MouseButton.LeftButton:
            return
        x=float(event.position().x()); y=float(event.position().y())
        row,x1,x2=self._row_at(x,y)
        if row >= 0:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.selected=row
            self.segmentSelected.emit(row)
            seg=self.segments[row]
            self.drag_row=row
            self.drag_anchor_time=self._time_for_x(x)
            self.drag_original_start=float(seg.get("start",0))
            self.drag_original_end=float(seg.get("end",self.drag_original_start+0.1))
            edge=max(7.0, min(14.0, (x2-x1)*0.18))
            if abs(x-x1) <= edge:
                self.drag_mode="start"
            elif abs(x-x2) <= edge:
                self.drag_mode="end"
            else:
                self.drag_mode="move"
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            self.update()
            return

        # Click/drag on empty timeline = move playhead.
        t=self._time_for_x(x)
        self.playhead=t
        self.seekRequested.emit(t)
        self.dragging_playhead=True
        self.drag_mode="playhead"
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.update()

    def mouseMoveEvent(self,event):
        x=float(event.position().x()); y=float(event.position().y())
        if self.drag_mode in ("start","end","move") and 0 <= self.drag_row < len(self.segments):
            seg=self.segments[self.drag_row]
            now=self._time_for_x(x)
            delta=now-self.drag_anchor_time
            min_gap=0.05
            if self.drag_mode=="start":
                new_start=max(0.0, min(self.drag_original_end-min_gap, now))
                seg["start"]=new_start
                self.playhead=new_start
            elif self.drag_mode=="end":
                new_end=min(self.duration, max(self.drag_original_start+min_gap, now))
                seg["end"]=new_end
                self.playhead=new_end
            else:
                length=self.drag_original_end-self.drag_original_start
                new_start=max(0.0, min(self.duration-length, self.drag_original_start+delta))
                seg["start"]=new_start
                seg["end"]=new_start+length
                self.playhead=new_start
            self.seekRequested.emit(self.playhead)
            self.update()
            return
        if self.drag_mode=="playhead" or self.dragging_playhead:
            self.playhead=self._time_for_x(x)
            self.seekRequested.emit(self.playhead)
            self.update()
            return
        row,x1,x2=self._row_at(x,y)
        if row>=0:
            edge=max(7.0, min(14.0, (x2-x1)*0.18))
            if abs(x-x1)<=edge or abs(x-x2)<=edge:
                # Only show resize cursor on segment edges (trim handles)
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                # Normal arrow everywhere else — no OpenHand
                self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton:
            edited=self.drag_row if self.drag_mode in ("start","end","move") else -1
            self.dragging_playhead=False
            self.drag_mode=None
            self.drag_row=-1
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if edited>=0:
                self.segmentEdited.emit(edited)
            self.update()

    def mouseDoubleClickEvent(self, event):
        """Double-click a VOICE block (or any segment) to preview generated TTS."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = float(event.position().x())
        y = float(event.position().y())
        row, _, _ = self._row_at(x, y)
        # Also accept double-click on the VOICE track band even if slightly off.
        if row < 0:
            track_h, gap, top = 42, 6, 34
            vy = top + 2 * (track_h + gap)
            if vy <= y <= vy + track_h:
                # Find nearest segment under this X.
                t = self._time_for_x(x)
                best, best_d = -1, 1e9
                for i, seg in enumerate(self.segments):
                    try:
                        a = float(seg.get("start", 0)); b = float(seg.get("end", a))
                    except Exception:
                        continue
                    if a <= t <= b:
                        best = i
                        break
                    d = min(abs(t - a), abs(t - b))
                    if d < best_d:
                        best_d, best = d, i
                row = best
        if row >= 0:
            self.selected = row
            self.segmentSelected.emit(row)
            self.playVoiceRequested.emit(row)
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_V and self.selected >= 0:
            self.playVoiceRequested.emit(self.selected)
            event.accept()
            return
        super().keyPressEvent(event)


class DubbyAIStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1600, 950)
        self.setMinimumSize(1200, 720)

                # ===== LICENSE CHECK (must pass) =====
        self.license_info = None
        if not self.check_license():
            QMessageBox.critical(
                self,
                "License Required",
                f"A valid license is required to run {APP_NAME}.\nThe application will now exit."
            )
            sys.exit(1)
        # =====================================

        self.videos: Dict[str, str] = {}
        self.segments: Dict[str, list] = {}
        self.activities: Dict[str, list] = {}
        self.current_video: Optional[str] = None
        self.logo_path: Optional[str] = None
        self.workers = []
        self.batch_transcribe_queue = []
        self.batch_export_queue = []
        self.batch_transcribe_active = False
        self.batch_activity_queue = []
        self.batch_activity_active = False
        self.activity_full_auto_active = False
        self.activity_full_auto_started_at = 0.0
        self.batch_export_active = False
        self.batch_export_skipped = []
        self.batch_export_failures = []
        self.batch_export_total = 0
        self.batch_export_completed = 0
        self.batch_export_started_at = 0.0
        self.full_process_active = False
        self.full_process_started_at = 0.0
        self.timeline_maximized = False
        self._preview_playback_path = ""
        self._preview_proxy_retrying = False

        self.settings = {
            "gemini_api_key": "",
            "gemini_api_keys": [],
            "openai_api_key": "",
            "openai_model": "gpt-3.5-turbo",
            "translation_prompt": "You are a professional translator. Translate from Chinese to the target language. Preserve cultural context, idioms, and emotional tone. Keep proper names in their original form unless they have common translations. Maintain the natural flow of the original dialogue.",
            "tts_provider": "Edge-TTS",
            "voxcpm_device": "Auto",
            "voxcpm_model": "openbmb/VoxCPM2",
            "voxcpm_style": "Natural conversational voice, clear pronunciation, realistic emotion, natural pauses.",
            "voxcpm_reference_wav": "",
            "voxcpm_prompt_text": "",
            "huggingface_token": "",
            "min_speakers": 0,
            "max_speakers": 0,
            "gemini_tts_model": "gemini-3.1-flash-tts-preview",
            "gemini_default_voice": "Kore",
            "gemini_tts_style": "Professional human voice-over. Clear studio narration, natural conversational tone, steady pitch, clean pronunciation, subtle emotion, natural pauses. Do not sound robotic, exaggerated, theatrical, breathy, or synthetic.",
            "gemini_request_interval": 0.65,
            "gemini_transcribe_model": "gemini-3.7-flash",
            "gemini_activity_model": "gemini-3.7-flash",
            "output_folder": str(OUTPUT_DIR),
            "transcribe_engine": "Local Whisper + Analyze",
            "translation_provider": "Google Translate",
            "gpu_mode": "Auto",
            "target_lang": "km",
            "target_lang_name": "Khmer",
            "gender_default": "Auto",
            "voice_speed": 1.0,
            "edge_pitch_hz": 0,
            "edge_volume_pct": 0,
            "edge_concurrency": 5,
            "edge_max_rate_pct": 60,
            "tts_max_stretch": 0.38,
            "tts_pad_short": True,
            "voice_gain_db": 0.0,
            "keep_bgm": True,
            "vocal_strength": "strong",
            "bgm_volume": 0.25,
            "bgm_duck_db": -7,
            "add_subtitles": False,
            "subtitle_font_size": 56,
            "editor_font_family": "Khmer UI",
            "editor_font_size": 24,
            "subtitle_color": "#FFFFFF",
            "subtitle_bar_height": 140,
            "subtitle_bar_opacity": 0.65,
            "subtitle_blur_radius": 18,
            "subtitle_position_x": 50,
            "subtitle_position_y": 88,
            "font_family": "Noto Sans Khmer",
            "add_title": False,
            "title_text": "",
            "title_size": 48,
            "title_color": "#FFFFFF",
            "title_position": "top",
            "title_position_x": 50,
            "title_position_y": 10,
            "add_logo": False,
            "logo_path": "",
            "logo_width": 140,
            "logo_opacity": 0.85,
            "logo_position": "top-right",
            "logo_position_x": 90,
            "logo_position_y": 10,
            "aspect_ratio": "Original",
            "crf": 20,
            "preset": "ultrafast",
            "threads": 6,
            "export_hwaccel": "Auto",
        }

        # Restore API keys / AI preferences saved on disk (survives next-day reopen).
        try:
            saved = load_user_settings()
            if saved:
                self.settings.update(saved)
                keys = split_api_keys(self.settings.get("gemini_api_keys") or self.settings.get("gemini_api_key") or "")
                self.settings["gemini_api_keys"] = keys
                self.settings["gemini_api_key"] = keys[0] if keys else ""
        except Exception as exc:
            print(f"[Dubby] Could not load user_settings.json: {exc}")
               

        self.history = []
        self.history_index = -1
        self._recording_history = True

        self._build_ui()
        self._apply_theme()
        self._connect_preview()
        self.statusBar().showMessage("Ready")

        # Update preview instantly whenever title/logo settings change.
        self.chk_title.toggled.connect(self._refresh_preview_overlays)
        self.txt_title.textChanged.connect(self._refresh_preview_overlays)
        self.sp_title_size.valueChanged.connect(self._refresh_preview_overlays)
        self.sp_title_x.valueChanged.connect(self._refresh_preview_overlays)
        self.sp_title_y.valueChanged.connect(self._refresh_preview_overlays)
        self.chk_logo.toggled.connect(self._refresh_preview_overlays)
        self.sp_logo_w.valueChanged.connect(self._refresh_preview_overlays)
        self.sld_logo_op.valueChanged.connect(self._refresh_preview_overlays)
        self.sp_logo_x.valueChanged.connect(self._refresh_preview_overlays)
        self.sp_logo_y.valueChanged.connect(self._refresh_preview_overlays)
        self.cmb_font.currentTextChanged.connect(self._refresh_preview_overlays)
        self.btn_title_color.clicked.connect(self._refresh_preview_overlays)
        self.chk_subtitles.toggled.connect(self._refresh_preview_overlays)
        self.sp_sub_size.valueChanged.connect(self._refresh_preview_overlays)
        self.sp_bar_h.valueChanged.connect(self._refresh_preview_overlays)
        self.sld_bar_op.valueChanged.connect(self._refresh_preview_overlays)

    # ---------------- UI ----------------

    def _apply_theme(self):
        # Modern dark UI inspired by CapCut / professional video editors
        self.setStyleSheet("""
            /* === Global === */
            QMainWindow, QWidget {
                background: #0b0f19;
                color: #e2e8f0;
                font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
                font-size: 13px;
            }
            QToolTip {
                background: #1e293b;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 10px;
            }

            /* === Group boxes (cards) === */
            QGroupBox {
                background: #111827;
                border: 1px solid #1e293b;
                border-radius: 12px;
                margin-top: 14px;
                padding: 14px 12px 12px 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                color: #94a3b8;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.4px;
            }

            /* === Primary buttons === */
            QPushButton {
                background: #3b82f6;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 9px 16px;
                font-weight: 600;
                min-height: 18px;
            }
            QPushButton:hover {
                background: #60a5fa;
            }
            QPushButton:pressed {
                background: #2563eb;
            }
            QPushButton:disabled {
                background: #1e293b;
                color: #64748b;
            }

            /* Secondary / outline style via objectName or special cases */
            QPushButton#secondaryBtn, QToolButton {
                background: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px 14px;
            }
            QPushButton#secondaryBtn:hover, QToolButton:hover {
                background: #334155;
                border-color: #475569;
            }
            QToolButton {
                min-width: 28px;
                min-height: 26px;
            }

            /* Accent / success buttons */
            QPushButton#accentBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8b5cf6, stop:1 #6366f1);
            }
            QPushButton#accentBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #a78bfa, stop:1 #818cf8);
            }
            QPushButton#dangerBtn {
                background: #ef4444;
            }
            QPushButton#dangerBtn:hover {
                background: #f87171;
            }
            QPushButton#successBtn {
                background: #10b981;
            }
            QPushButton#successBtn:hover {
                background: #34d399;
            }

            /* === Inputs === */
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 7px 10px;
                color: #f1f5f9;
                selection-background-color: #3b82f6;
            }
            QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #475569;
            }
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
                border: 1px solid #3b82f6;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #94a3b8;
                width: 0; height: 0;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background: #1e293b;
                border: 1px solid #334155;
                selection-background-color: #3b82f6;
                color: #f1f5f9;
                outline: none;
            }

            /* === Tables === */
            QTableWidget, QListWidget {
                background: #0f172a;
                alternate-background-color: #111827;
                border: 1px solid #1e293b;
                border-radius: 10px;
                gridline-color: #1e293b;
                outline: none;
            }
            QTableWidget::item {
                padding: 6px;
                border: none;
            }
            QTableWidget::item:selected {
                background: #1e3a5f;
                color: #ffffff;
            }
            QHeaderView::section {
                background: #111827;
                color: #94a3b8;
                padding: 8px 6px;
                border: none;
                border-bottom: 1px solid #1e293b;
                font-weight: 600;
                font-size: 12px;
            }

            /* === Sliders === */
            QSlider::groove:horizontal {
                height: 6px;
                background: #1e293b;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #3b82f6;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
                background: #ffffff;
                border: 2px solid #3b82f6;
            }
            QSlider::handle:horizontal:hover {
                background: #60a5fa;
                border-color: #60a5fa;
            }

            /* === Progress === */
            QProgressBar {
                border: none;
                border-radius: 6px;
                text-align: center;
                background: #1e293b;
                color: #e2e8f0;
                height: 18px;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #8b5cf6);
                border-radius: 6px;
            }

            /* === Tabs === */
            QTabWidget::pane {
                border: 1px solid #1e293b;
                border-radius: 10px;
                background: #111827;
                top: -1px;
            }
            QTabBar::tab {
                background: transparent;
                color: #94a3b8;
                padding: 10px 18px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #111827;
                color: #f1f5f9;
                border-bottom: 2px solid #3b82f6;
            }
            QTabBar::tab:hover:!selected {
                color: #e2e8f0;
                background: #1e293b;
            }

            /* === Scrollbars === */
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #475569;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar:horizontal {
                background: transparent;
                height: 10px;
            }
            QScrollBar::handle:horizontal {
                background: #334155;
                border-radius: 5px;
                min-width: 30px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0;
            }

            /* === Labels & status === */
            QLabel {
                color: #cbd5e1;
            }
            QStatusBar {
                background: #0b0f19;
                color: #94a3b8;
                border-top: 1px solid #1e293b;
            }
            QCheckBox {
                spacing: 8px;
                color: #e2e8f0;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #475569;
                background: #0f172a;
            }
            QCheckBox::indicator:checked {
                background: #3b82f6;
                border-color: #3b82f6;
            }

            /* === Splitter handles === */
            QSplitter::handle {
                background: #1e293b;
            }
            QSplitter::handle:horizontal {
                width: 6px;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSplitter::handle:horizontal:hover {
                background: #3b82f6;
            }
            QSplitter::handle:vertical {
                height: 6px;
                margin: 0 2px;
                border-radius: 3px;
            }
            QSplitter::handle:vertical:hover {
                background: #3b82f6;
            }
        """)


    def _build_file_menu(self):
        """Premiere-style File menu: Import, Open, Save, Clear."""
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        menubar.setStyleSheet("""
            QMenuBar {
                background: #0f172a;
                color: #e2e8f0;
                padding: 2px 6px;
                border-bottom: 1px solid #1e293b;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background: #1e293b;
                color: #f8fafc;
            }
            QMenu {
                background: #111827;
                color: #e2e8f0;
                border: 1px solid #334155;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 28px 8px 16px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #2563eb;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background: #1e293b;
                margin: 4px 8px;
            }
        """)

        file_menu = menubar.addMenu("&File")
                # ========== LICENSE MENU ==========
        license_menu = menubar.addMenu("&License")

        act_activate = QAction("Activate License…", self)
        act_activate.triggered.connect(self._show_activate_license)
        license_menu.addAction(act_activate)

        act_deactivate = QAction("Deactivate License", self)
        act_deactivate.triggered.connect(self._deactivate_license)
        license_menu.addAction(act_deactivate)

        license_menu.addSeparator()

        act_status = QAction("License Status", self)
        act_status.triggered.connect(self._show_license_status)
        license_menu.addAction(act_status)
        # ==================================
        act_import = QAction("📁  Import Videos…", self)
        act_import.setShortcut(QKeySequence("Ctrl+I"))
        act_import.setStatusTip("Import video files into the project")
        act_import.triggered.connect(self._upload_videos)
        file_menu.addAction(act_import)

        act_open = QAction("📂  Open Project…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.setStatusTip("Open a saved Dubby project")
        act_open.triggered.connect(self._open_project)
        file_menu.addAction(act_open)

        act_save = QAction("💾  Save Project…", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.setStatusTip("Save the current project")
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_clear = QAction("🗑  Clear Project", self)
        act_clear.setShortcut(QKeySequence("Ctrl+Shift+N"))
        act_clear.setStatusTip("Clear all videos, transcripts, and settings from the workspace")
        act_clear.triggered.connect(self._clear_all)
        file_menu.addAction(act_clear)

        file_menu.addSeparator()

        act_exit = QAction("Exit", self)
        # Use Ctrl+Q only — avoid StandardKey.Quit quirks on some Windows setups.
        act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        act_exit.setStatusTip("Quit Dubby AI Studio")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # ---------- Help / Updates ----------
        help_menu = menubar.addMenu("&Help")

        act_update = QAction("🔄  Check for Updates…", self)
        act_update.setStatusTip("Check whether the manager published a newer version")
        act_update.triggered.connect(lambda: self._check_for_updates(silent=False))
        help_menu.addAction(act_update)

        help_menu.addSeparator()

        act_about = QAction(f"About {APP_NAME}", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _show_about(self):
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> v{APP_VERSION}<br><br>"
            "AI video dubbing / translation studio.<br><br>"
            "Use <b>Help → Check for Updates</b> when the manager releases new features."
        )

    def _check_for_updates(self, silent: bool = False):
        """Contact the manager's update.json feed and offer download if newer."""
        if getattr(self, "_update_worker", None) is not None and self._update_worker.isRunning():
            if not silent:
                self._status("Update check already running…")
            return
        if not silent:
            self._status("Checking for updates…")
        worker = UpdateCheckWorker(UPDATE_INFO_URL, parent=self)
        self._update_worker = worker

        def _on_ok(info: dict):
            remote = str(info.get("version") or "")
            if is_newer_version(remote, APP_VERSION):
                self._status(f"Update available: v{remote}")
                dlg = UpdateAvailableDialog(info, self)
                dlg.exec()
            else:
                if not silent:
                    QMessageBox.information(
                        self, "Up to Date",
                        f"You already have the latest version.\n\n"
                        f"Installed: v{APP_VERSION}\nServer: v{remote or APP_VERSION}"
                    )
                self._status(f"Up to date (v{APP_VERSION})")

        def _on_err(msg: str):
            if silent:
                # Quiet startup failure — do not interrupt the user.
                self._status(f"Update check skipped: {str(msg)[:120]}")
            else:
                QMessageBox.warning(
                    self, "Update Check Failed",
                    f"Could not check for updates:\n\n{msg}"
                )
                self._status("Update check failed")

        worker.finished_ok.connect(_on_ok)
        worker.finished_error.connect(_on_err)
        worker.start()

    def _build_ui(self):
        # Premiere-style top File menu (Import / Open / Save / Clear)
        self._build_file_menu()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top modern header bar
        header = QFrame()
        header.setObjectName("appHeader")
        header.setStyleSheet("""
            QFrame#appHeader {
                background: #111827;
                border-bottom: 1px solid #1e293b;
                border-radius: 0px;
                padding: 4px 0;
            }
        """)
        toolbar = QHBoxLayout(header)
        toolbar.setContentsMargins(12, 8, 12, 8)
        toolbar.setSpacing(8)

        # App title
        title_lbl = QLabel(f"  {APP_NAME}")
        title_lbl.setStyleSheet("font-size: 15px; font-weight: 700; color: #f1f5f9; letter-spacing: 0.3px;")
        toolbar.addWidget(title_lbl)

        toolbar.addSpacing(12)

        self.btn_upload = QPushButton("📁 Import")
        self.btn_upload.setObjectName("secondaryBtn")
        self.btn_upload.clicked.connect(self._upload_videos)
        toolbar.addWidget(self.btn_upload)

        self.btn_open = QPushButton("📂 Open")
        self.btn_open.setObjectName("secondaryBtn")
        self.btn_open.clicked.connect(self._open_project)
        toolbar.addWidget(self.btn_open)

        self.btn_save = QPushButton("💾 Save")
        self.btn_save.setObjectName("secondaryBtn")
        self.btn_save.clicked.connect(self._save_project)
        toolbar.addWidget(self.btn_save)

        self.btn_clear_top = QPushButton("🗑 Clear")
        self.btn_clear_top.setObjectName("secondaryBtn")
        self.btn_clear_top.setToolTip("Clear the whole project (videos, transcripts, settings)")
        self.btn_clear_top.clicked.connect(self._clear_all)
        toolbar.addWidget(self.btn_clear_top)

        self.btn_undo = QPushButton("↶")
        self.btn_undo.setObjectName("secondaryBtn")
        self.btn_undo.setToolTip("Undo")
        self.btn_undo.setFixedWidth(36)
        self.btn_undo.clicked.connect(self._undo)
        toolbar.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("↷")
        self.btn_redo.setObjectName("secondaryBtn")
        self.btn_redo.setToolTip("Redo")
        self.btn_redo.setFixedWidth(36)
        self.btn_redo.clicked.connect(self._redo)
        toolbar.addWidget(self.btn_redo)

        # Top-level task: Transcript Review (same bar as Import / Open / Save / AI·API)
        self.btn_transcript_task = QPushButton("📋 Transcript")
        self.btn_transcript_task.setObjectName("secondaryBtn")
        self.btn_transcript_task.setToolTip(
            "Open Transcript Review task — check Translate / TTS text, genders, and timing."
        )
        self.btn_transcript_task.clicked.connect(self._focus_transcript_review)
        toolbar.addWidget(self.btn_transcript_task)

        toolbar.addStretch()

        self.lbl_project = QLabel("No project")
        self.lbl_project.setStyleSheet("color: #64748b; font-size: 12px;")
        toolbar.addWidget(self.lbl_project)

        toolbar.addSpacing(8)

        self.btn_api = QPushButton("🔑 AI / API")
        self.btn_api.setObjectName("secondaryBtn")
        self.btn_api.clicked.connect(self._api_settings)
        toolbar.addWidget(self.btn_api)

        self.btn_export_top = QPushButton("Export")
        self.btn_export_top.setObjectName("successBtn")
        self.btn_export_top.clicked.connect(self._start_export)
        toolbar.addWidget(self.btn_export_top)

        self.btn_export_all = QPushButton("Export All")
        self.btn_export_all.setObjectName("accentBtn")
        self.btn_export_all.clicked.connect(self._start_export_all)
        toolbar.addWidget(self.btn_export_all)

        root.addWidget(header)

        # Main 3-panel splitter.
        # The two highlighted vertical borders in the UI are real draggable
        # splitter handles. Make them wide/easy to grab instead of 1-2 pixels.
        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.setHandleWidth(10)
        main_split.setChildrenCollapsible(False)
        main_split.setOpaqueResize(True)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setStretchFactor(2, 0)
        main_split.setStyleSheet("")  # uses global theme
        # Content area with padding
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 6)
        content_layout.setSpacing(0)
        content_layout.addWidget(main_split, stretch=1)
        root.addWidget(content, stretch=1)
        main_split.setToolTip(
            "Drag the vertical dividers to resize Project, Timeline/Preview, and Properties."
        )

        # Left panel.
        left = QWidget()
        left_layout = QVBoxLayout(left)

        project_box = QGroupBox("📁  Project")
        pl = QVBoxLayout(project_box)
        self.cmb_video = QComboBox()
        self.cmb_video.currentTextChanged.connect(self._on_video_changed)
        pl.addWidget(QLabel("Current Video"))
        pl.addWidget(self.cmb_video)
        self.lbl_video_info = QLabel("No video selected")
        self.lbl_video_info.setWordWrap(True)
        self.lbl_video_info.setStyleSheet("color: #64748b; font-size: 12px;")
        pl.addWidget(self.lbl_video_info)
        left_layout.addWidget(project_box)

        ai_box = QGroupBox("✨  AI Processing")
        al = QVBoxLayout(ai_box)

        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.cmb_model.setCurrentText("base")
        al.addWidget(QLabel("Whisper Model"))
        al.addWidget(self.cmb_model)

        self.cmb_transcribe_engine = QComboBox()
        self.cmb_transcribe_engine.addItems(["Local Whisper + Analyze", "Google Gemini + Analyze"])
        self.cmb_transcribe_engine.setCurrentText(self.settings.get("transcribe_engine", "Local Whisper + Analyze"))
        self.cmb_transcribe_engine.setToolTip("Google Gemini uses cloud audio transcription + speaker diarization + voice-based gender estimation. Local Whisper keeps the existing offline pipeline.")
        al.addWidget(QLabel("Transcribe Engine"))
        al.addWidget(self.cmb_transcribe_engine)

        self.cmb_language = QComboBox()
        self.cmb_language.addItems(list(LANGUAGES.keys()))
        self.cmb_language.setCurrentText("Khmer")
        self.cmb_language.currentTextChanged.connect(self._target_language_changed)
        al.addWidget(QLabel("Target Language"))
        al.addWidget(self.cmb_language)

        self.cmb_gender = QComboBox()
        self.cmb_gender.addItems(["Auto", "Male", "Female"])
        self.cmb_gender.setCurrentText("Auto")
        self.cmb_gender.setToolTip(
            "Auto = detect real Male/Female per speaker from the video audio.\n"
            "Manual Male/Female overrides detection for every segment."
        )
        al.addWidget(QLabel("Default Voice Gender"))
        al.addWidget(self.cmb_gender)

        self.cmb_gpu = QComboBox()
        self.cmb_gpu.addItems(["Auto", "CPU", "NVIDIA CUDA"])
        self.cmb_gpu.setToolTip(
            "Auto = use NVIDIA CUDA only when a safe probe succeeds.\n"
            "NVIDIA CUDA = force GPU when available; falls back to CPU if unsafe.\n"
            "CPU = always stable (recommended if CUDA previously crashed the app)."
        )
        al.addWidget(QLabel("Hardware"))
        al.addWidget(self.cmb_gpu)

        self.btn_transcribe = QPushButton("①  Transcribe + Analyze")
        self.btn_transcribe.clicked.connect(self._start_transcribe)
        al.addWidget(self.btn_transcribe)

        self.btn_transcribe_all = QPushButton("Transcribe All")
        self.btn_transcribe_all.setObjectName("secondaryBtn")
        self.btn_transcribe_all.clicked.connect(self._start_transcribe_all)
        al.addWidget(self.btn_transcribe_all)

        self.btn_activity = QPushButton("🎬  Analyze Activities")
        self.btn_activity.setToolTip(
            "Analyze the actual video frames + audio with Gemini and create a timestamped activity timeline.\n"
            "Useful for videos with little/no dialogue, such as camping, cooking, travel, and survival videos."
        )
        self.btn_activity.clicked.connect(self._start_activity_analysis)
        al.addWidget(self.btn_activity)

        self.btn_activity_all = QPushButton("Analyze Activities All")
        self.btn_activity_all.setObjectName("secondaryBtn")
        self.btn_activity_all.clicked.connect(self._start_activity_all)
        al.addWidget(self.btn_activity_all)

        self.btn_activity_full_auto = QPushButton("🚀  Activity FULL AUTO")
        self.btn_activity_full_auto.setObjectName("accentBtn")
        self.btn_activity_full_auto.setToolTip(
            "Analyze Activities → create timestamped narration → Generate Voice → Export All.\n"
            "No separate Transcribe or Translate step is required."
        )
        self.btn_activity_full_auto.clicked.connect(self._start_activity_full_auto)
        al.addWidget(self.btn_activity_full_auto)

        self.btn_translate = QPushButton("②  Translate All")
        self.btn_translate.clicked.connect(self._start_translate_all)
        al.addWidget(self.btn_translate)

        self.btn_full_process = QPushButton("🚀  FULL AUTO Pipeline")
        self.btn_full_process.setObjectName("accentBtn")
        self.btn_full_process.setToolTip("Run the complete pipeline for all imported videos:\nTranscribe → Translate → Voice → Export")
        self.btn_full_process.clicked.connect(self._start_full_process)
        al.addWidget(self.btn_full_process)

        self.btn_tts = QPushButton("③  Generate Voice All")
        self.btn_tts.clicked.connect(self._start_tts_all)
        al.addWidget(self.btn_tts)

        left_layout.addWidget(ai_box)

        audio_box = QGroupBox("🔊  Audio / BGM")
        aud = QVBoxLayout(audio_box)
        self.chk_bgm = QCheckBox("Keep original BGM")
        self.chk_bgm.setChecked(True)
        aud.addWidget(self.chk_bgm)

        self.cmb_strength = QComboBox()
        self.cmb_strength.addItems(["light", "medium", "strong", "isolate"])
        self.cmb_strength.setCurrentText("strong")
        self.cmb_strength.setToolTip(
            "light / medium / strong = progressive vocal reduction\n"
            "isolate = maximum side-channel BGM isolation (dialogue removed)"
        )
        aud.addWidget(QLabel("Vocal removal / Isolate BGM"))
        aud.addWidget(self.cmb_strength)

        aud.addWidget(QLabel("BGM Volume"))
        self.sld_bgm = QSlider(Qt.Orientation.Horizontal)
        self.sld_bgm.setRange(0, 100)
        self.sld_bgm.setValue(25)
        self.lbl_bgm = QLabel("25%")
        self.sld_bgm.valueChanged.connect(
            lambda v: self.lbl_bgm.setText(f"{v}%")
        )
        row = QHBoxLayout()
        row.addWidget(self.sld_bgm)
        row.addWidget(self.lbl_bgm)
        aud.addLayout(row)

        aud.addWidget(QLabel("BGM Ducking"))
        self.sld_duck = QSlider(Qt.Orientation.Horizontal)
        self.sld_duck.setRange(0, 15)
        self.sld_duck.setValue(7)
        aud.addWidget(self.sld_duck)
        left_layout.addWidget(audio_box)

        left_scroll = QScrollArea()
        self.left_panel_scroll = left_scroll
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(220)
        main_split.addWidget(left_scroll)

        # Center panel.
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(4)

        # ---------------------------------------------------------------
        # Premiere-style Program Monitor (Preview) — top of center stack.
        # Grouped with Timeline below so the pair behaves like Premiere's
        # Program + Timeline workspace.
        # ---------------------------------------------------------------
        preview_group = QGroupBox("🎬  Program Monitor  ·  Preview")
        self.preview_group = preview_group
        preview_group.setMinimumHeight(180)
        pg = QVBoxLayout(preview_group)
        pg.setContentsMargins(8, 10, 8, 8)
        pg.setSpacing(6)
        preview_group.setMinimumHeight(160)

        preview_frame = QFrame()
        self.preview_frame = preview_frame
        preview_frame.setStyleSheet("""
            QFrame {
                background: #000000;
                border: 1px solid #1e293b;
                border-radius: 10px;
            }
        """)
        preview_stack = QGridLayout(preview_frame)
        preview_stack.setContentsMargins(1, 1, 1, 1)
        preview_stack.setSpacing(0)

        # Use a QLabel fed by QVideoSink instead of QVideoWidget.
        # QVideoWidget uses a native video surface on Windows, which can paint
        # over sibling widgets and makes live title/logo/subtitle overlays
        # unreliable. Rendering the decoded frame ourselves guarantees that
        # editor overlays remain visible above the video.
        self.video_widget = QLabel()
        self.video_widget.setMinimumSize(300, 105)
        self.video_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.video_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_widget.setStyleSheet(
            "QLabel { background: #000000; border: none; }"
        )
        self.video_widget.setScaledContents(False)
        preview_stack.addWidget(self.video_widget, 0, 0)

        self.preview_overlay = QWidget()
        self.preview_overlay.setStyleSheet("background: transparent;")
        self.preview_overlay.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        self.preview_overlay.setCursor(Qt.CursorShape.ArrowCursor)  # simple normal mouse
        self.preview_overlay.installEventFilter(self)
        preview_stack.addWidget(self.preview_overlay, 0, 0)

        self.lbl_preview_title = QLabel(self.preview_overlay)
        self.lbl_preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_title.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.lbl_preview_title.hide()

        self.lbl_preview_logo = QLabel(self.preview_overlay)
        self.lbl_preview_logo.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.lbl_preview_logo.hide()

        self.lbl_preview_subtitle = QLabel(self.preview_overlay)
        self.lbl_preview_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_subtitle.setWordWrap(True)
        # Subtitle is draggable — do NOT make it transparent for mouse events.
        self.lbl_preview_subtitle.setCursor(Qt.CursorShape.ArrowCursor)
        self.lbl_preview_subtitle.hide()

        # Keep the overlay above QVideoWidget on Windows/native video surfaces.
        self.preview_overlay.raise_()
        self.lbl_preview_title.raise_()
        self.lbl_preview_logo.raise_()
        self.lbl_preview_subtitle.raise_()

        pg.addWidget(preview_frame, stretch=1)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        self.controls_widget = QWidget()
        self.controls_widget.setLayout(controls)
        self.controls_widget.setMinimumHeight(38)
        self.controls_widget.setMaximumHeight(44)
        self.controls_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setMinimumSize(96, 34)
        self.btn_play.setObjectName("successBtn")
        self.btn_play.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self.btn_play)

        self.btn_back5 = QPushButton("◀ 5s")
        self.btn_back5.setObjectName("secondaryBtn")
        self.btn_back5.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_back5.clicked.connect(lambda: self._nudge_playhead(-5))
        controls.addWidget(self.btn_back5)

        self.btn_forward5 = QPushButton("5s ▶")
        self.btn_forward5.setObjectName("secondaryBtn")
        self.btn_forward5.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_forward5.clicked.connect(lambda: self._nudge_playhead(5))
        controls.addWidget(self.btn_forward5)

        self.sld_position = QSlider(Qt.Orientation.Horizontal)
        self.sld_position.setRange(0, 1000)
        self.sld_position.setTracking(False)
        self.sld_position.sliderPressed.connect(self._position_slider_pressed)
        self.sld_position.sliderMoved.connect(self._position_slider_moved)
        self.sld_position.sliderReleased.connect(self._position_slider_released)
        controls.addWidget(self.sld_position, stretch=1)

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setMinimumWidth(105)
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.lbl_time)
        pg.addWidget(self.controls_widget, stretch=0)

        self.lbl_preview = QLabel(
            "Live editor preview: subtitles, title and logo are rendered in the export."
        )
        self.lbl_preview.setWordWrap(True)
        pg.addWidget(self.lbl_preview)

        # ---------------------------------------------------------------
        # Timeline group (BOTTOM of the same vertical splitter).
        # ---------------------------------------------------------------
        # ============================================================
        # Premiere-style layered workspace:
        #   Layer 1 → Transcript Review (Translate / TTS) — top task panel
        #   Layer 2 → Multi-track Timeline (VIDEO / SUB / VOICE / BGM)
        # Both live inside a vertical splitter so the user can
        # resize them independently, just like Premiere panels.
        # ============================================================
        timeline_box = QWidget()          # outer container (was QGroupBox)
        self.timeline_box = timeline_box
        tl = QVBoxLayout(timeline_box)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)

        # --- Layer 1: Multi-track Timeline panel (Premiere-style) ---
        timeline_layer = QGroupBox("⏱  Timeline  ·  Multi-track  (VIDEO / SUB / VOICE)")
        self.timeline_layer = timeline_layer
        tll = QVBoxLayout(timeline_layer)
        tll.setContentsMargins(8, 10, 8, 8)
        tll.setSpacing(6)

        # Visual editing tools
        visual_group = QGroupBox("✂  Edit Segment")
        vg = QHBoxLayout(visual_group)
        vg.setContentsMargins(6, 4, 6, 4)
        vg.setSpacing(6)
        self.btn_split_segment = QPushButton("✂ Split")
        self.btn_split_segment.setObjectName("secondaryBtn")
        self.btn_split_segment.setToolTip("Split selected segment at playhead")
        self.btn_split_segment.clicked.connect(self._split_selected_segment)
        vg.addWidget(self.btn_split_segment)
        self.btn_trim_start = QPushButton("◀ Trim Start")
        self.btn_trim_start.setObjectName("secondaryBtn")
        self.btn_trim_start.clicked.connect(self._trim_selected_start)
        vg.addWidget(self.btn_trim_start)
        self.btn_trim_end = QPushButton("Trim End ▶")
        self.btn_trim_end.setObjectName("secondaryBtn")
        self.btn_trim_end.clicked.connect(self._trim_selected_end)
        vg.addWidget(self.btn_trim_end)
        self.btn_delete_segment = QPushButton("🗑 Delete")
        self.btn_delete_segment.setObjectName("dangerBtn")
        self.btn_delete_segment.clicked.connect(self._delete_selected_segment)
        vg.addWidget(self.btn_delete_segment)
        self.btn_play_voice = QPushButton("▶ Voice")
        self.btn_play_voice.setObjectName("successBtn")
        self.btn_play_voice.setToolTip(
            "Play generated TTS for the selected segment.\n"
            "Also: double-click the purple VOICE block, or press V."
        )
        self.btn_play_voice.clicked.connect(self._play_selected_voice)
        vg.addWidget(self.btn_play_voice)

        # Compact edit navigation, matching the small left/right arrow
        # control requested by the user.  The arrows move the actual video
        # playhead by one second; the middle slider can be dragged directly.
        nav_frame = QFrame()
        nav_frame.setObjectName("editNudgeBar")
        nav_frame.setFixedHeight(30)
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(2)

        self.btn_edit_back = QToolButton()
        self.btn_edit_back.setText("◀")
        self.btn_edit_back.setToolTip("Move playhead backward 1 second")
        self.btn_edit_back.setFixedSize(28, 26)
        self.btn_edit_back.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_edit_back.clicked.connect(lambda: self._nudge_playhead(-1))
        nav_layout.addWidget(self.btn_edit_back)

        self.edit_nudge_slider = QSlider(Qt.Orientation.Horizontal)
        self.edit_nudge_slider.setRange(0, 1000)
        self.edit_nudge_slider.setValue(0)
        self.edit_nudge_slider.setFixedHeight(24)
        self.edit_nudge_slider.setToolTip("Drag to move the video playhead")
        self.edit_nudge_slider.sliderPressed.connect(
            self._edit_nudge_slider_pressed
        )
        self.edit_nudge_slider.sliderMoved.connect(
            self._edit_nudge_slider_moved
        )
        self.edit_nudge_slider.sliderReleased.connect(
            self._edit_nudge_slider_released
        )
        nav_layout.addWidget(self.edit_nudge_slider, stretch=1)

        self.btn_edit_forward = QToolButton()
        self.btn_edit_forward.setText("▶")
        self.btn_edit_forward.setToolTip("Move playhead forward 1 second")
        self.btn_edit_forward.setFixedSize(28, 26)
        self.btn_edit_forward.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_edit_forward.clicked.connect(lambda: self._nudge_playhead(1))
        nav_layout.addWidget(self.btn_edit_forward)

        nav_frame.setStyleSheet("""
            QFrame#editNudgeBar {
                background: transparent;
            }
            QToolButton {
                min-width: 28px;
                max-width: 28px;
                min-height: 26px;
                max-height: 26px;
                padding: 0px;
                font-size: 13px;
                font-weight: bold;
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #e2e8f0;
            }
            QToolButton:hover {
                background: #334155;
                border-color: #475569;
            }
            QToolButton:pressed {
                background: #3b82f6;
                border-color: #3b82f6;
            }
        """)
        vg.addWidget(nav_frame, stretch=1)

        vg.addWidget(QLabel("Zoom"))
        self.sld_timeline_zoom = QSlider(Qt.Orientation.Horizontal)
        self.sld_timeline_zoom.setRange(50, 300)
        self.sld_timeline_zoom.setValue(100)
        self.sld_timeline_zoom.setMaximumWidth(150)
        vg.addWidget(self.sld_timeline_zoom)

        self.btn_timeline_max = QPushButton("⛶ Maximize")
        self.btn_timeline_max.setObjectName("secondaryBtn")
        self.btn_timeline_max.clicked.connect(self._toggle_timeline_maximize)
        self.btn_timeline_max.setToolTip(
            "Hide the side panels while keeping Video Preview above the Timeline."
        )
        vg.addWidget(self.btn_timeline_max)
        vg.addStretch()
        tll.addWidget(visual_group)

        # Multi-track canvas (VIDEO / SUBTITLES / VOICE / BGM)
        self.timeline_canvas = TimelineEditorWidget()
        self.timeline_canvas.segmentSelected.connect(self._timeline_segment_selected)
        self.timeline_canvas.seekRequested.connect(self._timeline_seek_requested)
        self.timeline_canvas.segmentEdited.connect(self._timeline_segment_edited)
        self.timeline_canvas.playVoiceRequested.connect(self._play_segment_voice)
        self.sld_timeline_zoom.valueChanged.connect(
            lambda v: self.timeline_canvas.set_zoom(v / 100.0)
        )
        tll.addWidget(self.timeline_canvas, stretch=1)

        # --- Layer 2: Transcript review panel (separate task layer) ---
        # Dedicated workspace for checking translation + TTS text accuracy.
        transcript_layer = QGroupBox("📋  Transcript  ·  Review Layer  (check Translate / TTS)")
        self.transcript_layer = transcript_layer
        trl = QVBoxLayout(transcript_layer)
        trl.setContentsMargins(8, 10, 8, 8)
        trl.setSpacing(6)

        # Gender + apply tools
        gender_bar = QHBoxLayout()
        gender_bar.setSpacing(8)
        gender_bar.addWidget(QLabel("Fill Gender:"))
        self.cmb_fill_gender = QComboBox()
        self.cmb_fill_gender.addItems(["Default", "Male", "Female", "Auto"])
        self.cmb_fill_gender.setCurrentText("Default")
        self.cmb_fill_gender.setMaximumWidth(110)
        gender_bar.addWidget(self.cmb_fill_gender)
        self.btn_fill_gender = QPushButton("Apply to Selected / All")
        self.btn_fill_gender.setObjectName("secondaryBtn")
        self.btn_fill_gender.clicked.connect(self._fill_gender_rows)
        gender_bar.addWidget(self.btn_fill_gender)
        gender_bar.addStretch()
        trl.addLayout(gender_bar)

        # Clipboard workflow
        clipboard_tools = QHBoxLayout()
        clipboard_tools.setSpacing(6)
        self.btn_copy_original = QPushButton("Copy Original")
        self.btn_copy_original.setObjectName("secondaryBtn")
        self.btn_copy_original.setToolTip(
            "Copy Original lines (one per row) → paste into ChatGPT to translate."
        )
        self.btn_copy_original.clicked.connect(self._copy_original_column)
        clipboard_tools.addWidget(self.btn_copy_original)
        self.btn_copy_all_rows = QPushButton("Copy All Rows")
        self.btn_copy_all_rows.setObjectName("secondaryBtn")
        self.btn_copy_all_rows.clicked.connect(self._copy_all_rows)
        clipboard_tools.addWidget(self.btn_copy_all_rows)
        self.btn_copy_subtitles = QPushButton("Copy Subtitle")
        self.btn_copy_subtitles.setObjectName("secondaryBtn")
        self.btn_copy_subtitles.clicked.connect(self._copy_column)
        clipboard_tools.addWidget(self.btn_copy_subtitles)
        self.btn_copy_tts = QPushButton("Copy TTS")
        self.btn_copy_tts.setObjectName("secondaryBtn")
        self.btn_copy_tts.clicked.connect(self._copy_tts_column)
        clipboard_tools.addWidget(self.btn_copy_tts)
        self.btn_paste_subtitles = QPushButton("Paste to Subtitle")
        self.btn_paste_subtitles.setToolTip(
            "Paste ChatGPT translation (one line per subtitle). Also fills empty TTS."
        )
        self.btn_paste_subtitles.clicked.connect(self._paste_column_to_segments)
        clipboard_tools.addWidget(self.btn_paste_subtitles)
        self.btn_paste_both = QPushButton("Paste Subtitle+TTS")
        self.btn_paste_both.setObjectName("successBtn")
        self.btn_paste_both.setToolTip(
            "Paste ChatGPT lines into BOTH Subtitle and TTS (recommended)."
        )
        self.btn_paste_both.clicked.connect(self._paste_subtitle_and_tts)
        clipboard_tools.addWidget(self.btn_paste_both)
        self.btn_paste_tts = QPushButton("Paste to TTS")
        self.btn_paste_tts.clicked.connect(self._paste_tts_column)
        clipboard_tools.addWidget(self.btn_paste_tts)
        trl.addLayout(clipboard_tools)

        # Editor font (Khmer) — readable review of Subtitle / TTS
        font_bar = QHBoxLayout()
        font_bar.setSpacing(8)
        font_bar.addWidget(QLabel("Editor Font:"))
        self.cmb_editor_font = QComboBox()
        self.cmb_editor_font.addItems([
            "Khmer UI", "Khmer Nida", "Noto Sans Khmer",
            "Leelawadee", "System Default"
        ])
        self.cmb_editor_font.setCurrentText(
            self.settings.get("editor_font_family", "Khmer UI")
        )
        self.cmb_editor_font.setToolTip("Font for the transcript table and Subtitle/TTS editor.")
        font_bar.addWidget(self.cmb_editor_font)
        font_bar.addWidget(QLabel("Size:"))
        self.sp_editor_font_size = QSpinBox()
        self.sp_editor_font_size.setRange(14, 48)
        self.sp_editor_font_size.setValue(int(self.settings.get("editor_font_size", 24)))
        self.sp_editor_font_size.setToolTip(
            "Khmer text needs a larger size. Try 18–24 for comfortable reading."
        )
        font_bar.addWidget(self.sp_editor_font_size)
        self.btn_apply_editor_font = QPushButton("Apply Font")
        self.btn_apply_editor_font.setObjectName("secondaryBtn")
        self.btn_apply_editor_font.clicked.connect(self._apply_editor_font)
        font_bar.addWidget(self.btn_apply_editor_font)
        font_bar.addStretch()
        trl.addLayout(font_bar)
        self.cmb_editor_font.currentTextChanged.connect(lambda *_: self._apply_editor_font())
        self.sp_editor_font_size.valueChanged.connect(lambda *_: self._apply_editor_font())

        # Transcript table
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "#", "Start", "End", "Speaker", "Gender",
            "Original", "Subtitle", "TTS"
        ])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        # Larger row height + font so Khmer Subtitle / TTS columns are readable.
        self.table.verticalHeader().setDefaultSectionSize(52)
        self.table.setFont(QFont(self.settings.get("editor_font_family", "Khmer UI"), int(self.settings.get("editor_font_size", 24))))
        self.table.cellChanged.connect(self._table_changed)
        self.table.itemSelectionChanged.connect(self._selected_row_changed)
        trl.addWidget(self.table, stretch=1)

        # Subtitle / TTS inline editor
        subtitle_editor_box = QGroupBox("Subtitle / TTS Editor")
        seb = QVBoxLayout(subtitle_editor_box)
        self.txt_subtitle_edit = QTextEdit()
        self.txt_subtitle_edit.setPlaceholderText(
            "Select a transcript row, then paste or rewrite the subtitle here..."
        )
        self.txt_subtitle_edit.setMaximumHeight(180)
        self.txt_subtitle_edit.setFont(QFont(self.settings.get("editor_font_family", "Khmer UI"), max(18, int(self.settings.get("editor_font_size", 24)) + 2)))
        seb.addWidget(self.txt_subtitle_edit)
        editor_buttons = QHBoxLayout()
        self.btn_apply_subtitle = QPushButton("Apply Subtitle to Selected")
        self.btn_apply_subtitle.clicked.connect(self._apply_subtitle_editor)
        editor_buttons.addWidget(self.btn_apply_subtitle)
        self.btn_apply_tts = QPushButton("Apply as TTS")
        self.btn_apply_tts.clicked.connect(self._apply_tts_editor)
        editor_buttons.addWidget(self.btn_apply_tts)
        editor_buttons.addStretch()
        seb.addLayout(editor_buttons)
        trl.addWidget(subtitle_editor_box)

        # Nested splitter: Timeline layer ↔ Transcript layer
        # (Premiere-style independent panel heights)
        # Under Program Monitor: Timeline ONLY.
        # Transcript Review is a top-bar task (📋 Transcript) in its own window.
        tl.addWidget(timeline_layer, stretch=1)

        # Keep a reference used by maximize/restore
        self.timeline_actions_group = timeline_layer
        self.timeline_transcript_splitter = None  # no longer embedded under preview

        # Host Transcript Review in a dedicated task window (opened from header).
        self.transcript_dialog = QDialog(self)
        self.transcript_dialog.setWindowTitle("📋  Transcript Review  ·  Translate / TTS")
        self.transcript_dialog.resize(1100, 720)
        self.transcript_dialog.setMinimumSize(720, 480)
        # Apply saved editor font after widgets exist
        QTimer.singleShot(0, self._apply_editor_font)
        _td_layout = QVBoxLayout(self.transcript_dialog)
        _td_layout.setContentsMargins(10, 10, 10, 10)
        _td_layout.addWidget(transcript_layer)

        # ---------------------------------------------------------------
        # Preview ↔ Timeline only (vertical splitter)
        # ---------------------------------------------------------------
        self.center_splitter = QSplitter(Qt.Orientation.Vertical)
        self.center_splitter.setHandleWidth(10)
        self.center_splitter.setChildrenCollapsible(False)
        self.center_splitter.setOpaqueResize(True)
        self.center_splitter.addWidget(preview_group)
        self.center_splitter.addWidget(timeline_box)

        self.center_splitter.widget(0).setMinimumHeight(145)
        self.center_splitter.widget(1).setMinimumHeight(200)
        self.center_splitter.setStretchFactor(0, 1)
        self.center_splitter.setStretchFactor(1, 3)
        self.center_splitter.setSizes([420, 520])
        self.center_splitter.setStyleSheet("""
            QSplitter::handle:vertical {
                background: #1e293b;
                height: 8px;
                margin: 0 1px;
                border-radius: 4px;
            }
            QSplitter::handle:vertical:hover {
                background: #3b82f6;
            }
            QSplitter::handle:vertical:pressed {
                background: #2563eb;
            }
        """)
        self.center_splitter.setToolTip(
            "Drag this divider up/down to resize Video Preview and Timeline workspace."
        )
        # Apply SizeVer cursor only on the actual handle (not the whole area)
        QTimer.singleShot(
            0,
            lambda: (
                self.center_splitter.handle(1).setCursor(Qt.CursorShape.SizeVerCursor)
                if self.center_splitter.handle(1) is not None else None
            ),
        )
        cl.addWidget(self.center_splitter, stretch=1)

        main_split.addWidget(center)

        # Right panel.
        right = QWidget()
        self.right_panel = right
        right.setMinimumWidth(260)
        rl = QVBoxLayout(right)

        tabs = QTabWidget()
        rl.addWidget(tabs)

        # Text tab.
        text_tab = QWidget()
        tx = QVBoxLayout(text_tab)

        self.chk_subtitles = QCheckBox("Show translated subtitles")
        self.chk_subtitles.setChecked(False)
        tx.addWidget(self.chk_subtitles)

        self.cmb_font = QComboBox()
        self.cmb_font.addItems([
            "Khmer Nida", "Noto Sans Khmer", "Khmer UI",
            "Leelawadee", "System Default"
        ])
        self.cmb_font.setCurrentText(self.settings.get("font_family", "Noto Sans Khmer"))
        tx.addWidget(QLabel("Export / Preview Font (Khmer)"))
        tx.addWidget(self.cmb_font)
        tx.addWidget(QLabel(
            "Also set Editor Font size in 📋 Transcript for the review table."
        ))

        self.sp_sub_size = QSpinBox()
        self.sp_sub_size.setRange(10, 160)
        self.sp_sub_size.setValue(int(self.settings.get("subtitle_font_size", 56)))
        self.sp_sub_size.setToolTip(
            "Burned-in subtitle size on export/preview. Khmer is clearer around 52–64."
        )
        tx.addWidget(QLabel("Subtitle Size on Video (10–160)"))
        tx.addWidget(self.sp_sub_size)

        self.btn_sub_color = QPushButton("Subtitle Color")
        self.sub_color = "#FFFFFF"
        self.btn_sub_color.clicked.connect(self._pick_sub_color)
        tx.addWidget(self.btn_sub_color)

        self.sp_bar_h = QSpinBox()
        self.sp_bar_h.setRange(40, 400)
        self.sp_bar_h.setValue(140)
        tx.addWidget(QLabel("Subtitle Bar Height"))
        tx.addWidget(self.sp_bar_h)

        self.sld_bar_op = QSlider(Qt.Orientation.Horizontal)
        self.sld_bar_op.setRange(0, 100)
        self.sld_bar_op.setValue(65)
        tx.addWidget(QLabel("Subtitle Bar Opacity"))
        tx.addWidget(self.sld_bar_op)

        self.sp_blur_radius = QSpinBox()
        self.sp_blur_radius.setRange(0, 60)
        self.sp_blur_radius.setValue(18)
        self.sp_blur_radius.setToolTip(
            "Premiere-style Gaussian Blur radius for the subtitle background plate.\n"
            "0 = hard rectangle, 12–24 = soft natural plate, higher = wider soft glow."
        )
        tx.addWidget(QLabel("Subtitle Bar Gaussian Blur (radius)"))
        tx.addWidget(self.sp_blur_radius)

        # Free position — place subtitle anywhere on the video frame
        sub_pos_row = QHBoxLayout()
        self.sp_sub_x = QSpinBox()
        self.sp_sub_x.setRange(0, 100)
        self.sp_sub_x.setValue(int(self.settings.get("subtitle_position_x", 50)))
        self.sp_sub_x.setSuffix("%")
        self.sp_sub_y = QSpinBox()
        self.sp_sub_y.setRange(0, 100)
        self.sp_sub_y.setValue(int(self.settings.get("subtitle_position_y", 88)))
        self.sp_sub_y.setSuffix("%")
        sub_pos_row.addWidget(QLabel("X"))
        sub_pos_row.addWidget(self.sp_sub_x)
        sub_pos_row.addWidget(QLabel("Y"))
        sub_pos_row.addWidget(self.sp_sub_y)
        tx.addWidget(QLabel("Subtitle Position (drag on preview or set %)"))
        tx.addLayout(sub_pos_row)
        self.sp_sub_x.valueChanged.connect(self._refresh_preview_overlays)
        self.sp_sub_y.valueChanged.connect(self._refresh_preview_overlays)

        title_box = QGroupBox("Title")
        tt = QVBoxLayout(title_box)
        self.chk_title = QCheckBox("Enable title")
        tt.addWidget(self.chk_title)
        self.txt_title = QLineEdit()
        self.txt_title.setPlaceholderText("Enter title...")
        tt.addWidget(self.txt_title)
        self.sp_title_size = QSpinBox()
        self.sp_title_size.setRange(18, 120)
        self.sp_title_size.setValue(48)
        tt.addWidget(QLabel("Size"))
        tt.addWidget(self.sp_title_size)
        title_pos_row = QHBoxLayout()
        self.sp_title_x = QSpinBox()
        self.sp_title_x.setRange(0, 100)
        self.sp_title_x.setValue(int(self.settings.get("title_position_x", 50)))
        self.sp_title_x.setSuffix("%")
        self.sp_title_y = QSpinBox()
        self.sp_title_y.setRange(0, 100)
        self.sp_title_y.setValue(int(self.settings.get("title_position_y", 10)))
        self.sp_title_y.setSuffix("%")
        title_pos_row.addWidget(QLabel("X"))
        title_pos_row.addWidget(self.sp_title_x)
        title_pos_row.addWidget(QLabel("Y"))
        title_pos_row.addWidget(self.sp_title_y)
        tt.addWidget(QLabel("Title Position (anywhere in video)"))
        tt.addLayout(title_pos_row)

        self.cmb_title_pos = QComboBox()
        self.cmb_title_pos.addItems(["top", "center", "bottom"])
        self.cmb_title_pos.setCurrentText(self.settings.get("title_position", "top"))
        tt.addWidget(QLabel("Preset Position"))
        tt.addWidget(self.cmb_title_pos)
        self.btn_title_color = QPushButton("Title Color")
        self.title_color = "#FFFFFF"
        self.btn_title_color.clicked.connect(self._pick_title_color)
        tt.addWidget(self.btn_title_color)
        tx.addWidget(title_box)

        tabs.addTab(text_tab, "  📝  Text  ")

        # Logo tab.
        logo_tab = QWidget()
        lg = QVBoxLayout(logo_tab)
        self.chk_logo = QCheckBox("Enable logo")
        lg.addWidget(self.chk_logo)
        self.btn_logo = QPushButton("Select PNG/JPG")
        self.btn_logo.clicked.connect(self._select_logo)
        lg.addWidget(self.btn_logo)
        self.lbl_logo = QLabel("No logo selected")
        lg.addWidget(self.lbl_logo)
        self.sp_logo_w = QSpinBox()
        self.sp_logo_w.setRange(40, 800)
        self.sp_logo_w.setValue(140)
        lg.addWidget(QLabel("Logo Width"))
        lg.addWidget(self.sp_logo_w)
        self.sld_logo_op = QSlider(Qt.Orientation.Horizontal)
        self.sld_logo_op.setRange(0, 100)
        self.sld_logo_op.setValue(85)
        lg.addWidget(QLabel("Opacity"))
        lg.addWidget(self.sld_logo_op)
        logo_pos_row = QHBoxLayout()
        self.sp_logo_x = QSpinBox()
        self.sp_logo_x.setRange(0, 100)
        self.sp_logo_x.setValue(int(self.settings.get("logo_position_x", 90)))
        self.sp_logo_x.setSuffix("%")
        self.sp_logo_y = QSpinBox()
        self.sp_logo_y.setRange(0, 100)
        self.sp_logo_y.setValue(int(self.settings.get("logo_position_y", 10)))
        self.sp_logo_y.setSuffix("%")
        logo_pos_row.addWidget(QLabel("X"))
        logo_pos_row.addWidget(self.sp_logo_x)
        logo_pos_row.addWidget(QLabel("Y"))
        logo_pos_row.addWidget(self.sp_logo_y)
        lg.addWidget(QLabel("Logo Position (anywhere in video)"))
        lg.addLayout(logo_pos_row)

        self.cmb_logo_pos = QComboBox()
        self.cmb_logo_pos.addItems([
            "top-left", "top-right", "bottom-left", "bottom-right"
        ])
        self.cmb_logo_pos.setCurrentText(self.settings.get("logo_position", "top-right"))
        lg.addWidget(QLabel("Preset Position"))
        lg.addWidget(self.cmb_logo_pos)

        self.btn_logo_center = QPushButton("Center Logo")
        self.btn_logo_center.clicked.connect(
            lambda: self._set_overlay_position("logo", 50, 50)
        )
        lg.addWidget(self.btn_logo_center)
        lg.addStretch()
        tabs.addTab(logo_tab, "  🖼  Logo  ")

        # Voice tab.
        voice_tab = QWidget()
        vc = QVBoxLayout(voice_tab)
        vc.addWidget(QLabel("TTS Provider"))
        self.cmb_tts_provider = QComboBox()
        self.cmb_tts_provider.addItems(["Edge-TTS", "Google Gemini TTS", "VoxCPM2 Local", "VoxCPM2 Local CPU"])
        self.cmb_tts_provider.setCurrentText(self.settings.get("tts_provider", "Edge-TTS"))
        if hasattr(self, "txt_voxcpm_ref"):
            self.txt_voxcpm_ref.setText(self.settings.get("voxcpm_reference_wav", ""))
        if hasattr(self, "cmb_export_hwaccel"):
            self.cmb_export_hwaccel.setCurrentText(self.settings.get("export_hwaccel", "Auto"))
        vc.addWidget(self.cmb_tts_provider)

        vc.addWidget(QLabel("Voice Speed"))
        self.sld_speed = QSlider(Qt.Orientation.Horizontal)
        self.sld_speed.setRange(80, 120)
        self.sld_speed.setValue(100)
        self.lbl_speed = QLabel("1.00x")
        self.sld_speed.valueChanged.connect(
            lambda v: self.lbl_speed.setText(f"{v/100:.2f}x")
        )
        vc.addWidget(self.sld_speed)
        vc.addWidget(self.lbl_speed)

        vc.addWidget(QLabel("Edge-TTS Pitch (Hz) — keep near 0 for the clearest natural voice"))
        self.sld_edge_pitch = QSlider(Qt.Orientation.Horizontal)
        self.sld_edge_pitch.setRange(-12, 12)
        self.sld_edge_pitch.setValue(int(self.settings.get("edge_pitch_hz", 0)))
        self.lbl_edge_pitch = QLabel(f"{self.sld_edge_pitch.value():+d} Hz")
        self.sld_edge_pitch.valueChanged.connect(
            lambda v: self.lbl_edge_pitch.setText(f"{v:+d} Hz")
        )
        vc.addWidget(self.sld_edge_pitch)
        vc.addWidget(self.lbl_edge_pitch)

        vc.addWidget(QLabel("Edge-TTS Volume Trim (%) — 0 is recommended"))
        self.sld_edge_volume = QSlider(Qt.Orientation.Horizontal)
        self.sld_edge_volume.setRange(-6, 6)
        self.sld_edge_volume.setValue(int(self.settings.get("edge_volume_pct", 0)))
        self.lbl_edge_volume = QLabel(f"{self.sld_edge_volume.value():+d}%")
        self.sld_edge_volume.valueChanged.connect(
            lambda v: self.lbl_edge_volume.setText(f"{v:+d}%")
        )
        vc.addWidget(self.sld_edge_volume)
        vc.addWidget(self.lbl_edge_volume)

        vc.addWidget(QLabel("Edge-TTS Parallel Requests (faster generation)"))
        self.sp_edge_concurrency = QSpinBox()
        self.sp_edge_concurrency.setRange(1, 8)
        self.sp_edge_concurrency.setValue(int(self.settings.get("edge_concurrency", 5) or 5))
        self.sp_edge_concurrency.setToolTip(
            "Generate several Edge-TTS sentences at the same time. 4-6 is recommended."
        )
        vc.addWidget(self.sp_edge_concurrency)
        vc.addWidget(QLabel(
            "Recommended: 5. Higher is faster, but too many parallel requests may trigger service throttling."
        ))

        vc.addWidget(QLabel(
            "Edge-TTS natural profile: moderate rate, 0 Hz pitch, no heavy audio processing. "
            "This keeps consonants and human-like dynamics clear."
        ))

        vc.addWidget(QLabel("VoxCPM2 Reference Voice (optional)"))
        ref_row = QHBoxLayout()
        self.txt_voxcpm_ref = QLineEdit(self.settings.get("voxcpm_reference_wav", ""))
        self.btn_voxcpm_ref = QPushButton("Browse")
        self.btn_voxcpm_ref.clicked.connect(self._select_voxcpm_reference)
        ref_row.addWidget(self.txt_voxcpm_ref); ref_row.addWidget(self.btn_voxcpm_ref)
        vc.addLayout(ref_row)

        vc.addWidget(QLabel("Selected Speaker Voice"))
        self.cmb_speaker_voice = QComboBox()
        vc.addWidget(self.cmb_speaker_voice)

        vc.addWidget(QLabel("Google Gemini Voice (used when Google Gemini TTS is selected)"))
        self.cmb_gemini_voice = QComboBox()
        self.cmb_gemini_voice.addItems(GEMINI_TTS_VOICES)
        vc.addWidget(self.cmb_gemini_voice)
        self.btn_apply_gemini_voice = QPushButton("Apply Gemini Voice to Selected Rows")
        self.btn_apply_gemini_voice.clicked.connect(self._apply_gemini_voice_to_rows)
        vc.addWidget(self.btn_apply_gemini_voice)

        vc.addWidget(QLabel("Recommended default: Edge-TTS at 1.00x, 0 Hz pitch, 0% volume trim for a clean, natural voice."))

        self.btn_apply_voice = QPushButton("Apply Voice to Selected Rows")
        self.btn_apply_voice.clicked.connect(self._apply_voice_to_rows)
        vc.addWidget(self.btn_apply_voice)

        vc.addWidget(QLabel(
            "v3: Community-1 diarization + ML gender classifier analyze each speaker separately. "
            "Gender confidence is shown in the transcript; uncertain cases remain Auto for review. "
            "VoxCPM2 can use per-speaker reference audio for voice cloning."
        ))
        vc.addStretch()
        tabs.addTab(voice_tab, "  🎙  Voice  ")

        # Activity analysis tab.
        activity_tab = QWidget()
        ac = QVBoxLayout(activity_tab)
        ac.setSpacing(8)
        ac.addWidget(QLabel("Timestamped Visual Activities"))
        self.activity_table = QTableWidget(0, 5)
        self.activity_table.setHorizontalHeaderLabels(["#", "Start", "End", "Activity", "Narration"])
        self.activity_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.activity_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.activity_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.activity_table.verticalHeader().setDefaultSectionSize(42)
        self.activity_table.cellDoubleClicked.connect(self._activity_row_activated)
        ac.addWidget(self.activity_table, stretch=1)
        self.lbl_activity_info = QLabel("No activity analysis yet.")
        self.lbl_activity_info.setWordWrap(True)
        self.lbl_activity_info.setStyleSheet("color:#94a3b8; font-size:12px;")
        ac.addWidget(self.lbl_activity_info)
        self.txt_activity_narration = QTextEdit()
        self.txt_activity_narration.setPlaceholderText("Combined activity narration will appear here…")
        self.txt_activity_narration.setMaximumHeight(150)
        ac.addWidget(self.txt_activity_narration)
        self.btn_copy_activity = QPushButton("Copy Activity Narration")
        self.btn_copy_activity.setObjectName("secondaryBtn")
        self.btn_copy_activity.clicked.connect(self._copy_activity_narration)
        ac.addWidget(self.btn_copy_activity)

        self.btn_activity_to_timeline = QPushButton("➕ Send Activities to Subtitle / TTS")
        self.btn_activity_to_timeline.setObjectName("accentBtn")
        self.btn_activity_to_timeline.setToolTip(
            "Convert the analyzed activities into timestamped Subtitle/TTS rows. "
            "Existing transcript rows are preserved unless you choose to replace them."
        )
        self.btn_activity_to_timeline.clicked.connect(self._activity_to_timeline)
        ac.addWidget(self.btn_activity_to_timeline)

        self.btn_activity_to_timeline_replace = QPushButton("♻ Replace Transcript with Activities")
        self.btn_activity_to_timeline_replace.setObjectName("secondaryBtn")
        self.btn_activity_to_timeline_replace.setToolTip(
            "Replace the current transcript rows with the analyzed activity timeline."
        )
        self.btn_activity_to_timeline_replace.clicked.connect(
            lambda: self._activity_to_timeline(replace=True)
        )
        ac.addWidget(self.btn_activity_to_timeline_replace)
        tabs.addTab(activity_tab, "  🎬  Activity  ")

        # Export tab.
        export_tab = QWidget()
        ex = QVBoxLayout(export_tab)

        ex.addWidget(QLabel("Output Folder"))
        out_row = QHBoxLayout()
        self.txt_output_folder = QLineEdit()
        self.txt_output_folder.setPlaceholderText(str(OUTPUT_DIR))
        self.txt_output_folder.setText(str(self.settings.get("output_folder") or OUTPUT_DIR))
        self.txt_output_folder.setToolTip(
            "Exported MP4 and SRT files are saved here.\n"
            "Leave as the default app output folder or choose any folder."
        )
        out_row.addWidget(self.txt_output_folder, stretch=1)
        self.btn_browse_output = QPushButton("Browse…")
        self.btn_browse_output.setObjectName("secondaryBtn")
        self.btn_browse_output.setFixedWidth(90)
        self.btn_browse_output.clicked.connect(self._browse_output_folder)
        out_row.addWidget(self.btn_browse_output)
        self.btn_open_output = QPushButton("Open")
        self.btn_open_output.setObjectName("secondaryBtn")
        self.btn_open_output.setFixedWidth(70)
        self.btn_open_output.setToolTip("Open the current output folder in the file manager")
        self.btn_open_output.clicked.connect(self._open_output_folder)
        out_row.addWidget(self.btn_open_output)
        ex.addLayout(out_row)
        hint = QLabel("Default: app /output  ·  Choose any folder for Export / Export All")
        hint.setStyleSheet("color:#64748b; font-size:11px;")
        ex.addWidget(hint)

        ex.addWidget(QLabel("Aspect Ratio"))
        self.cmb_aspect = QComboBox()
        self.cmb_aspect.addItems(["Original", "16:9", "9:16", "1:1"])
        ex.addWidget(self.cmb_aspect)

        ex.addWidget(QLabel("Encoding preset (faster = lower quality, use 'veryfast' for speed)"))
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(["ultrafast", "veryfast", "faster", "fast", "medium", "slow"])
        self.cmb_preset.setCurrentText("ultrafast")
        ex.addWidget(self.cmb_preset)

        ex.addWidget(QLabel("Export Engine"))
        self.cmb_export_mode = QComboBox()
        self.cmb_export_mode.addItems([
            "Turbo (FFmpeg + NVENC)",
            "Standard (MoviePy)"
        ])
        self.cmb_export_mode.setCurrentText(
            self.settings.get("export_mode", "Turbo (FFmpeg + NVENC)")
        )
        self.cmb_export_mode.setToolTip(
            "Turbo = much faster FFmpeg/libass rendering and one video encode pass. "
            "Recommended for long videos.\\n"
            "Standard = original MoviePy compositor for maximum compatibility."
        )
        ex.addWidget(self.cmb_export_mode)

        ex.addWidget(QLabel("Video Encoder / GPU"))
        self.cmb_export_hwaccel = QComboBox()
        self.cmb_export_hwaccel.addItems(["Auto", "NVIDIA NVENC (legacy low-VRAM GPU)", "CPU x264"])
        self.cmb_export_hwaccel.setCurrentText(self.settings.get("export_hwaccel", "Auto"))
        ex.addWidget(self.cmb_export_hwaccel)
        ex.addWidget(QLabel("Auto uses NVIDIA h264_nvenc when available; otherwise CPU x264."))

        ex.addWidget(QLabel("CRF / Quality (lower = higher quality, 20-23 recommended)"))
        self.sp_crf = QSpinBox()
        self.sp_crf.setRange(16, 30)
        self.sp_crf.setValue(22)
        ex.addWidget(self.sp_crf)

        self.btn_export = QPushButton("🚀 Generate + Export")
        self.btn_export.clicked.connect(self._start_export)
        ex.addWidget(self.btn_export)

        self.chk_shutdown = QCheckBox("Shutdown PC when finished")
        self.chk_shutdown.setChecked(bool(self.settings.get("shutdown_when_finished", False)))
        self.chk_shutdown.setToolTip(
            "After FULL AUTO, Activity FULL AUTO, or Export All completes,\n"
            "schedule a PC shutdown in 2 minutes.\n"
            "Windows: run  shutdown /a  in Command Prompt to cancel.\n"
            "Linux: run  shutdown -c  to cancel."
        )
        ex.addWidget(self.chk_shutdown)

        self.btn_clear = QPushButton("Clear Project")
        self.btn_clear.clicked.connect(self._clear_all)
        ex.addWidget(self.btn_clear)
        ex.addStretch()
        tabs.addTab(export_tab, "  ⚙  Export  ")

        main_split.addWidget(right)

        # Initial proportions. The user can freely drag either splitter
        # handle afterwards; the center panel receives the extra space.
        main_split.setSizes([320, 980, 340])

        # Bottom progress. Status messages are shown only in the real Qt status bar
        # to avoid displaying the same message twice.
        self.progress = QProgressBar()
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        # Decode video frames into a Qt video sink so the preview can be
        # composed with editable title/logo/subtitle layers.
        self.video_sink = QVideoSink(self)
        self.video_sink.videoFrameChanged.connect(self._video_frame_changed)
        self.player.setVideoSink(self.video_sink)

        self._last_video_pixmap = QPixmap()
        self._preview_source_size = QSize(0, 0)
        self._last_play_click = 0.0
        self._position_slider_dragging = False
        self._position_slider_target = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_player_ui)
        self.timer.start(250)

        # Reliable editor shortcuts. WindowShortcut makes them work even when
        # focus is on the timeline/table, while the normal keyPressEvent below
        # still protects text-entry widgets from stealing Space/Arrow keys.
        self.shortcut_play_pause = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.shortcut_play_pause.setContext(Qt.ShortcutContext.WindowShortcut)
        self.shortcut_play_pause.activated.connect(self._shortcut_play_pause)

        self.shortcut_back_5 = QShortcut(
            QKeySequence(Qt.Key.Key_Left), self
        )
        self.shortcut_back_5.setContext(Qt.ShortcutContext.WindowShortcut)
        self.shortcut_back_5.activated.connect(lambda: self._nudge_playhead(-5))

        self.shortcut_forward_5 = QShortcut(
            QKeySequence(Qt.Key.Key_Right), self
        )
        self.shortcut_forward_5.setContext(Qt.ShortcutContext.WindowShortcut)
        self.shortcut_forward_5.activated.connect(lambda: self._nudge_playhead(5))

        # Optional silent update check a few seconds after the UI is ready.
        if AUTO_CHECK_UPDATES_ON_START:
            QTimer.singleShot(3500, lambda: self._check_for_updates(silent=True))

    def _video_frame_changed(self, frame):
        """Display the decoded QVideoFrame in the editor preview.

        High-res masters (1080p / 2K / 4K) are downscaled to the preview
        widget size before QPixmap conversion so the UI stays responsive and
        Windows Media Foundation does not drop frames under memory pressure.
        """
        try:
            image = frame.toImage()
            if image.isNull():
                return
            # Remember native master size for overlay layout, then downscale
            # the raster we actually paint.
            self._preview_source_size = image.size()
            target = self.video_widget.size() if hasattr(self, "video_widget") else QSize(960, 540)
            max_w = max(320, target.width())
            max_h = max(180, target.height())
            # Cap paint size even if the widget is huge (multi-monitor).
            max_w = min(max_w, 1600)
            max_h = min(max_h, 900)
            if image.width() > max_w or image.height() > max_h:
                image = image.scaled(
                    max_w, max_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            pix = QPixmap.fromImage(image)
            if pix.isNull():
                return
            self._last_video_pixmap = pix
            self._update_video_frame_pixmap()
        except Exception as exc:
            # Preview failure should never stop playback/export.
            self._status(f"Preview frame warning: {exc}")
                # ==================== LICENSE METHODS ====================

    def check_license(self) -> bool:
        cached = load_local_license()
        if cached and cached.get("serial"):
            result = validate_serial(cached["serial"])
            if result.get("valid"):
                self.license_info = result
                self._update_license_status()
                return True

        # Show activation dialog
        dlg = LicenseDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result:
            self.license_info = dlg.result
            self._update_license_status()
            return True
        return False

    def _update_license_status(self):
        """Show license status permanently on the right side of the status bar."""
        info = getattr(self, "license_info", None)

        if not info or not info.get("valid"):
            text = "No valid license"
            color = "#e74c3c"
        elif info.get("type") == "lifetime":
            text = "License: Lifetime ✓"
            color = "#27ae60"
        else:
            days = info.get("days_left")
            if days is not None:
                text = f"License: {days} day(s) left"
            else:
                text = "License: Time-limited ✓"
            color = "#27ae60"

        if not hasattr(self, "_license_status_label"):
            self._license_status_label = QLabel(text)
            self._license_status_label.setStyleSheet(
                f"padding: 0 12px; color: {color}; font-weight: 600;"
            )
            self.statusBar().addPermanentWidget(self._license_status_label)
        else:
            self._license_status_label.setText(text)
            self._license_status_label.setStyleSheet(
                f"padding: 0 12px; color: {color}; font-weight: 600;"
            )

    def _show_activate_license(self):
        dlg = LicenseDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result:
            self.license_info = dlg.result
            self._update_license_status()
            QMessageBox.information(self, "License", "License activated successfully.")

    def _deactivate_license(self):
        reply = QMessageBox.question(
            self,
            "Deactivate License",
            "This will remove the license from this computer.\n"
            "You can activate it again later on this or another machine.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            clear_local_license()
            self.license_info = None
            self._update_license_status()
            QMessageBox.information(self, "Deactivated", "License has been removed.")

    def _show_license_status(self):
        info = getattr(self, "license_info", None)
        if not info:
            msg = "No license information available."
        else:
            lines = [
                f"Status : {'Valid' if info.get('valid') else 'Invalid'}",
                f"Type   : {info.get('type', 'unknown').title()}",
            ]
            if info.get("days_left") is not None:
                lines.append(f"Days left : {info.get('days_left')}")
            if info.get("expires_at"):
                lines.append(f"Expires  : {info.get('expires_at')[:10]}")
            lines.append(f"Message : {info.get('message', '')}")
            msg = "\n".join(lines)

        QMessageBox.information(self, "License Status", msg)

    def _require_license(self) -> bool:
        """Call this at the start of Export / TTS / Full Auto etc."""
        if not getattr(self, "license_info", None) or not self.license_info.get("valid"):
            QMessageBox.warning(
                self,
                "License Required",
                "A valid license is required for this feature.\n\n"
                "Go to  License → Activate License…"
            )
            return False
        return True
    def _update_video_frame_pixmap(self):
        if not hasattr(self, "video_widget"):
            return
        pix = getattr(self, "_last_video_pixmap", QPixmap())
        if pix.isNull():
            return
        target = self.video_widget.size()
        if target.width() < 2 or target.height() < 2:
            return
        scaled = pix.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_widget.setPixmap(scaled)
        self._refresh_preview_overlays()

    def _video_content_rect(self):
        """Return the actual displayed-video rectangle inside the preview."""
        w = max(1, self.preview_overlay.width())
        h = max(1, self.preview_overlay.height())
        src = getattr(self, "_preview_source_size", QSize(0, 0))
        if src.width() <= 0 or src.height() <= 0:
            return 0, 0, w, h
        scale = min(w / src.width(), h / src.height())
        dw = max(1, int(src.width() * scale))
        dh = max(1, int(src.height() * scale))
        return (w - dw) // 2, (h - dh) // 2, dw, dh

    def _shortcut_play_pause(self):
        focus = self.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit)):
            return
        self._toggle_play()

    def _connect_preview(self):
        self.player.positionChanged.connect(self._player_position_changed)
        self.player.durationChanged.connect(self._player_duration_changed)
        self.player.playbackStateChanged.connect(self._player_state_changed)
        self.player.mediaStatusChanged.connect(self._player_media_status_changed)
        self.player.errorOccurred.connect(self._player_error)

    # ---------------- Preview ----------------

    def _player_state_changed(self, state):
        # UI follows the real QMediaPlayer state; never guess the state
        # immediately after calling play()/pause().
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("⏸ Pause")
        else:
            self.btn_play.setText("▶ Play")

        self.btn_play.setEnabled(
            self.player.source().isValid()
            and self.player.mediaStatus() != QMediaPlayer.MediaStatus.InvalidMedia
        )

    def _player_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            self.btn_play.setEnabled(False)
            self.btn_play.setText("▶ Play")
            self._try_preview_proxy_fallback("InvalidMedia")
        elif status == QMediaPlayer.MediaStatus.LoadedMedia:
            self.btn_play.setEnabled(True)
            if hasattr(self, "video_widget") and self.video_widget.text() == "Loading preview…":
                self.video_widget.setText("")
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.btn_play.setEnabled(self.player.source().isValid())
            self.btn_play.setText("▶ Play")
        elif status in (
            QMediaPlayer.MediaStatus.LoadingMedia,
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.BufferingMedia,
        ):
            self.btn_play.setEnabled(self.player.source().isValid())

    def _player_error(self, error, error_string):
        if error != QMediaPlayer.Error.NoError:
            self.btn_play.setEnabled(False)
            self.btn_play.setText("▶ Play")
            msg = error_string or getattr(error, "name", str(error))
            self._status(f"Preview error: {msg}")
            # If the original master failed, try building / switching to a proxy.
            self._try_preview_proxy_fallback(str(msg))

    def _toggle_play(self):
        if not self.player.source().isValid():
            self._status("Import/select a video first.")
            return

        # Ignore accidental double-clicks while Qt is still changing state.
        now = time.monotonic()
        if now - getattr(self, "_last_play_click", 0.0) < 0.12:
            return
        self._last_play_click = now

        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            return

        if self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.setPosition(0)

        self.player.play()

    def _edit_nudge_slider_pressed(self):
        self._edit_nudge_dragging = True

    def _edit_nudge_slider_moved(self, value):
        if self.player.duration() > 0:
            target = int(self.player.duration() * value / 1000)
            self.player.setPosition(target)

    def _edit_nudge_slider_released(self):
        self._edit_nudge_dragging = False

    def _nudge_playhead(self, seconds):
        if self.player.duration() <= 0:
            return
        pos = max(
            0,
            min(
                self.player.duration(),
                self.player.position() + int(seconds * 1000),
            ),
        )
        self.player.setPosition(pos)
        if hasattr(self, "timeline_canvas"):
            self.timeline_canvas.set_playhead(pos / 1000.0)
        self._refresh_preview_overlays()
        self._status(
            f"Playhead: {self._fmt_ms(pos)} / {self._fmt_ms(self.player.duration())}"
        )

    def _position_slider_pressed(self):
        self._position_slider_dragging = True

    def _position_slider_moved(self, value):
        # Preview the requested position while dragging, but don't let
        # positionChanged fight the user's mouse movement.
        if self.player.duration() > 0:
            self._position_slider_target = int(
                self.player.duration() * value / 1000
            )
            self.lbl_time.setText(
                f"{self._fmt_ms(self._position_slider_target)} / "
                f"{self._fmt_ms(self.player.duration())}"
            )

    def _position_slider_released(self):
        if self.player.duration() > 0:
            value = self.sld_position.value()
            self.player.setPosition(
                int(self.player.duration() * value / 1000)
            )
        self._position_slider_dragging = False

    def _seek(self, value):
        # Backward-compatible helper for code that still calls _seek().
        if self.player.duration() > 0:
            self.player.setPosition(int(self.player.duration() * value / 1000))

    def _player_position_changed(self, pos):
        dur = self.player.duration()
        if dur > 0:
            if not getattr(self, "_position_slider_dragging", False):
                self.sld_position.blockSignals(True)
                self.sld_position.setValue(int(pos * 1000 / dur))
                self.sld_position.blockSignals(False)
                self.lbl_time.setText(
                    f"{self._fmt_ms(pos)} / {self._fmt_ms(dur)}"
                )
            else:
                target = getattr(self, "_position_slider_target", pos)
                self.lbl_time.setText(
                    f"{self._fmt_ms(target)} / {self._fmt_ms(dur)}"
                )
            if hasattr(self, "timeline_canvas"):
                self.timeline_canvas.set_playhead(pos / 1000.0)
            if hasattr(self, "edit_nudge_slider") and not getattr(
                self, "_edit_nudge_dragging", False
            ):
                self.edit_nudge_slider.blockSignals(True)
                self.edit_nudge_slider.setValue(int(pos * 1000 / dur))
                self.edit_nudge_slider.blockSignals(False)
            self._refresh_preview_overlays()

    def _player_duration_changed(self, dur):
        self.lbl_time.setText(f"00:00 / {self._fmt_ms(dur)}")
        self._refresh_preview_overlays()

    def _update_player_ui(self):
        pass

    @staticmethod
    def _fmt_ms(ms: int) -> str:
        sec = max(0, int(ms / 1000))
        return f"{sec // 60:02d}:{sec % 60:02d}"

    # ---------------- Live preview overlays ----------------

    def _overlay_xy(self, kind, width, height, item_w, item_h):
        # Position overlays inside the actual video image, not the black
        # letterbox area around a vertical/horizontal video.
        vx, vy, vw, vh = self._video_content_rect()
        if kind == "title":
            xp = self.sp_title_x.value() / 100.0
            yp = self.sp_title_y.value() / 100.0
        else:
            xp = self.sp_logo_x.value() / 100.0
            yp = self.sp_logo_y.value() / 100.0
        x = int(vx + xp * vw - item_w / 2)
        y = int(vy + yp * vh - item_h / 2)
        x = max(vx, min(max(vx, vx + vw - item_w), x))
        y = max(vy, min(max(vy, vy + vh - item_h), y))
        return x, y

    def _refresh_preview_overlays(self, *args):
        if not hasattr(self, "preview_overlay"):
            return

        w = max(1, self.preview_overlay.width())
        h = max(1, self.preview_overlay.height())

        # Keep the transparent editor layer above the decoded video frame.
        self.preview_overlay.raise_()
        self.lbl_preview_title.raise_()
        self.lbl_preview_logo.raise_()
        self.lbl_preview_subtitle.raise_()

        # ---------------- Title ----------------
        title = self.txt_title.text().strip()
        if self.chk_title.isChecked() and title:
            self.lbl_preview_title.setText(title)
            size = max(8, int(self.sp_title_size.value() * max(1, h) / 720))
            self.lbl_preview_title.setStyleSheet(
                f"color:{self.title_color}; background:rgba(0,0,0,0);"
                f"font-family:'{self.cmb_font.currentText()}';"
                f"font-size:{size}px; font-weight:600;"
            )
            self.lbl_preview_title.adjustSize()
            tw = min(max(self.lbl_preview_title.width() + 20, 80), w)
            th = max(self.lbl_preview_title.height() + 10, 30)
            self.lbl_preview_title.setFixedSize(tw, th)
            x, y = self._overlay_xy("title", w, h, tw, th)
            self.lbl_preview_title.move(x, y)
            self.lbl_preview_title.show()
        else:
            self.lbl_preview_title.hide()

        # ---------------- Logo ----------------
        logo_path = self.logo_path
        if self.chk_logo.isChecked() and logo_path and Path(logo_path).exists():
            pix = QPixmap(logo_path)
            if not pix.isNull():
                _, _, vw, _ = self._video_content_rect()
                logo_w = max(20, int(self.sp_logo_w.value() * max(1, vw) / 1080))
                logo_w = min(logo_w, max(20, vw - 8))
                logo_h = max(1, int(pix.height() * logo_w / max(1, pix.width())))
                scaled = pix.scaled(
                    logo_w, logo_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.lbl_preview_logo.setPixmap(scaled)
                self.lbl_preview_logo.setFixedSize(scaled.size())
                self.lbl_preview_logo.setWindowOpacity(
                    max(0.0, min(1.0, self.sld_logo_op.value() / 100.0))
                )
                x, y = self._overlay_xy(
                    "logo", w, h, scaled.width(), scaled.height()
                )
                self.lbl_preview_logo.move(x, y)
                self.lbl_preview_logo.show()
            else:
                self.lbl_preview_logo.hide()
        else:
            self.lbl_preview_logo.hide()

        # ---------------- Live subtitle ----------------
        subtitle_text = ""
        if self.chk_subtitles.isChecked() and self.current_video:
            now = self.player.position() / 1000.0
            for seg in self.segments.get(self.current_video, []):
                try:
                    a = float(seg.get("start", 0))
                    b = float(seg.get("end", a + 0.1))
                except Exception:
                    continue
                if a <= now <= b:
                    subtitle_text = str(
                        seg.get("translated") or seg.get("original") or ""
                    ).strip()
                    break

        if subtitle_text:
            vx, vy, vw, vh = self._video_content_rect()
            font_size = max(10, int(self.sp_sub_size.value() * max(1, vh) / 720))
            bar_h = max(40, int(self.sp_bar_h.value() * max(1, vh) / 720))
            self.lbl_preview_subtitle.setText(subtitle_text)
            self.lbl_preview_subtitle.setStyleSheet(
                f"color:{self.sub_color};"
                f"background:rgba(0,0,0,{max(0,min(1,self.sld_bar_op.value()/100.0))});"
                f"font-family:'{self.cmb_font.currentText()}';"
                f"font-size:{font_size}px; font-weight:600;"
                "padding:8px 16px; border-radius:6px;"
            )
            self.lbl_preview_subtitle.setFixedWidth(max(100, int(vw * 0.92)))
            self.lbl_preview_subtitle.setFixedHeight(
                min(max(40, bar_h), max(40, vh))
            )
            # Free X/Y position (% of the actual video image area).
            px = float(self.sp_sub_x.value() if hasattr(self, "sp_sub_x")
                       else self.settings.get("subtitle_position_x", 50))
            py = float(self.sp_sub_y.value() if hasattr(self, "sp_sub_y")
                       else self.settings.get("subtitle_position_y", 88))
            sw = self.lbl_preview_subtitle.width()
            sh = self.lbl_preview_subtitle.height()
            x = int(vx + px / 100.0 * vw - sw / 2)
            y = int(vy + py / 100.0 * vh - sh / 2)
            x = max(vx, min(vx + vw - sw, x))
            y = max(vy, min(vy + vh - sh, y))
            self.lbl_preview_subtitle.move(x, y)
            self.lbl_preview_subtitle.show()
        else:
            self.lbl_preview_subtitle.hide()

    def _set_overlay_position(self, kind, x, y):
        if kind == "logo":
            self.sp_logo_x.setValue(int(x))
            self.sp_logo_y.setValue(int(y))
        elif kind == "subtitle":
            if hasattr(self, "sp_sub_x"):
                self.sp_sub_x.setValue(int(x))
            if hasattr(self, "sp_sub_y"):
                self.sp_sub_y.setValue(int(y))
            self.settings["subtitle_position_x"] = int(x)
            self.settings["subtitle_position_y"] = int(y)
        else:
            self.sp_title_x.setValue(int(x))
            self.sp_title_y.setValue(int(y))
        self._refresh_preview_overlays()

    def eventFilter(self, obj, event):
        if obj is getattr(self, "preview_overlay", None):
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                pos = event.position().toPoint()
                # Priority: logo → title → subtitle (all free-move on preview)
                for kind, label in (
                    ("logo", self.lbl_preview_logo),
                    ("title", self.lbl_preview_title),
                    ("subtitle", self.lbl_preview_subtitle),
                ):
                    if label.isVisible() and label.geometry().contains(pos):
                        self._drag_overlay_kind = kind
                        self._drag_overlay_offset = pos - label.pos()
                        self.preview_overlay.setCursor(Qt.CursorShape.ClosedHandCursor)
                        return True

                # Clicking empty preview seeks the video.
                if self.player.duration() > 0:
                    vx, vy, vw, vh = self._video_content_rect()
                    if vx <= pos.x() <= vx + vw:
                        ratio = (pos.x() - vx) / max(1, vw)
                        self.player.setPosition(
                            int(max(0, min(1, ratio)) * self.player.duration())
                        )
                        return True

            elif (
                event.type() == QEvent.Type.MouseMove
                and getattr(self, "_drag_overlay_kind", None)
            ):
                pos = event.position().toPoint()
                kind = self._drag_overlay_kind
                if kind == "logo":
                    label = self.lbl_preview_logo
                elif kind == "subtitle":
                    label = self.lbl_preview_subtitle
                else:
                    label = self.lbl_preview_title
                vx, vy, vw, vh = self._video_content_rect()
                nx = max(vx, min(vx + vw - label.width(),
                                 pos.x() - self._drag_overlay_offset.x()))
                ny = max(vy, min(vy + vh - label.height(),
                                 pos.y() - self._drag_overlay_offset.y()))
                cx = nx + label.width() / 2
                cy = ny + label.height() / 2
                xp = round((cx - vx) / max(1, vw) * 100)
                yp = round((cy - vy) / max(1, vh) * 100)
                self._set_overlay_position(
                    kind,
                    max(0, min(100, xp)),
                    max(0, min(100, yp)),
                )
                return True

            elif (
                event.type() == QEvent.Type.MouseButtonRelease
                and getattr(self, "_drag_overlay_kind", None)
            ):
                self._drag_overlay_kind = None
                self.preview_overlay.setCursor(Qt.CursorShape.ArrowCursor)
                return True

        return super().eventFilter(obj, event)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_video_frame_pixmap)
        QTimer.singleShot(0, self._refresh_preview_overlays)

    def keyPressEvent(self,event):
        # Space = play/pause. Left/Right = 5-second seek.
        focus=self.focusWidget()
        text_focus=isinstance(focus,(QLineEdit,QTextEdit))
        if event.key()==Qt.Key.Key_Space and not text_focus:
            self._toggle_play()
            event.accept(); return
        if event.key()==Qt.Key.Key_Left and not text_focus:
            self._nudge_playhead(-5); event.accept(); return
        if event.key()==Qt.Key.Key_Right and not text_focus:
            self._nudge_playhead(5); event.accept(); return
        super().keyPressEvent(event)

    # ---------------- Project/video ----------------

    def _upload_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Import Videos", "",
            "Videos (*.mp4 *.mkv *.avi *.mov *.webm *.m4v)"
        )
        if not files:
            return

        if len(files) + len(self.videos) > MAX_VIDEOS:
            QMessageBox.warning(
                self, "Limit", f"Maximum {MAX_VIDEOS} videos."
            )
            return

        for src in files:
            src_path = Path(src)
            # Do not unnecessarily duplicate large files. Keep the source path.
            self.videos[src_path.name] = str(src_path)

        self.cmb_video.clear()
        self.cmb_video.addItems(list(self.videos.keys()))
        if self.videos:
            self.cmb_video.setCurrentIndex(0)

        self.lbl_project.setText(f"{len(self.videos)} video(s)")
        self._snapshot()

    def _on_video_changed(self, name: str):
        if not name or name not in self.videos:
            return
        self.current_video = name
        path = self.videos[name]
        self.player.stop()
        self._position_slider_dragging = False
        self._last_video_pixmap = QPixmap()
        if hasattr(self, "video_widget"):
            self.video_widget.clear()
            self.video_widget.setText("Loading preview…")

        duration, w, h = get_video_info(path)
        self.lbl_video_info.setText(
            f"{Path(path).name}\n"
            f"{w} × {h}\n"
            f"Duration: {duration:.1f}s"
        )

        # Prefer a lightweight H.264 proxy for 1080p+/2K/4K so Qt preview works.
        # Export still uses the original master path stored in self.videos.
        preview_path = path
        try:
            if w >= 1800 or h >= 1000:
                self._status(f"Preparing editor preview for {w}×{h}…")
                preview_path = ensure_preview_proxy(
                    path, max_width=1280, progress_cb=self._status
                )
                if preview_path != path:
                    self._status(f"Preview proxy ready for {Path(path).name}")
        except Exception as exc:
            self._status(f"Preview proxy skipped: {exc}")
            preview_path = path

        abs_path = str(Path(preview_path).resolve())
        self._preview_playback_path = abs_path
        self.player.setSource(QUrl.fromLocalFile(abs_path))
        self.btn_play.setText("▶ Play")
        # Enable Play once media is loaded; mediaStatusChanged will refine this.
        self.btn_play.setEnabled(True)

        self._load_table()
        self._load_activity_table()
        self._update_voice_list()


    def _try_preview_proxy_fallback(self, reason: str = ""):
        """When Qt cannot decode the master, rebuild an H.264 proxy and reload it."""
        if not self.current_video or self.current_video not in self.videos:
            return
        if getattr(self, "_preview_proxy_retrying", False):
            return
        master = self.videos[self.current_video]
        current = getattr(self, "_preview_playback_path", "") or ""
        # Already on a proxy that still failed — do not loop forever.
        try:
            if current and Path(current).resolve() != Path(master).resolve() and "proxy_" in Path(current).name:
                self._status(f"Preview still failed after proxy ({reason}). Codec may be unsupported.")
                return
        except Exception:
            pass
        self._preview_proxy_retrying = True
        try:
            self._status(f"Preview decode failed ({reason}). Building H.264 proxy…")
            proxy = ensure_preview_proxy(master, max_width=1280, progress_cb=self._status)
            abs_path = str(Path(proxy).resolve())
            if abs_path == str(Path(master).resolve()):
                self._status("Could not build a preview proxy for this file.")
                return
            self._preview_playback_path = abs_path
            self.player.stop()
            self.player.setSource(QUrl.fromLocalFile(abs_path))
            self.btn_play.setEnabled(True)
            self._status(f"Preview proxy loaded for {Path(master).name}")
        except Exception as exc:
            self._status(f"Preview proxy fallback failed: {exc}")
        finally:
            self._preview_proxy_retrying = False

    def _target_language_changed(self, name):
        self.settings["target_lang_name"] = name
        self.settings["target_lang"] = LANGUAGES.get(name, "km")
        self._update_voice_list()

    # ---------------- Transcript table ----------------

    def _load_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)

        if not self.current_video:
            self.table.blockSignals(False)
            return

        for seg in self.segments.get(self.current_video, []):
            row = self.table.rowCount()
            self.table.insertRow(row)

            values = [
                str(seg.get("index", row + 1)),
                f"{float(seg.get('start', 0)):.2f}",
                f"{float(seg.get('end', 0)):.2f}",
                seg.get("speaker", "Speaker 1"),
                seg.get("gender", "Auto"),
                seg.get("original", ""),
                seg.get("translated", ""),
                seg.get("tts_text", ""),
            ]
            family, size, body_font = self._editor_khmer_font()
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                if col in (5, 6, 7):  # Original / Subtitle / TTS
                    item.setFont(body_font)
                    item.setToolTip(val)  # full text on hover when cell is narrow
                else:
                    item.setFont(QFont(family, max(12, size - 2)))
                self.table.setItem(row, col, item)
            self.table.setRowHeight(row, max(44, int(size * 2.8)))

            # Speaker and gender use combo boxes for easy editing.
            speaker = QComboBox()
            speaker.addItems(["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"])
            speaker.setCurrentText(seg.get("speaker", "Speaker 1"))
            speaker.currentTextChanged.connect(
                lambda value, r=row: self._set_table_field(r, "speaker", value)
            )
            self.table.setCellWidget(row, 3, speaker)

            gender = QComboBox()
            gender.addItems(["Auto", "Male", "Female"])
            gender.setCurrentText(seg.get("gender", "Auto"))
            det = seg.get("detected_gender", "Unknown")
            conf = float(seg.get("gender_confidence", 0.0) or 0.0)
            male_p = float(seg.get("gender_male", 0.0) or 0.0)
            female_p = float(seg.get("gender_female", 0.0) or 0.0)
            gender.setToolTip(
                f"Detected: {det}  conf={conf:.0%}\n"
                f"Male={male_p:.0%}  Female={female_p:.0%}\n"
                "Override manually if the AI is wrong."
            )
            gender.currentTextChanged.connect(
                lambda value, r=row: self._set_table_field(r, "gender", value)
            )
            self.table.setCellWidget(row, 4, gender)

        self.table.blockSignals(False)
        self._apply_editor_font()
        self.lbl_preview.setText(
            f"{self.table.rowCount()} transcript segments. "
            "Use the visual timeline to select, seek, split, trim, or delete segments."
        )
        self._refresh_timeline()

    def _refresh_timeline(self):
        if not hasattr(self, "timeline_canvas"):
            return
        duration=0.1
        if self.current_video and self.current_video in self.videos:
            try: duration=max(duration,get_video_info(self.videos[self.current_video])[0])
            except Exception: pass
        segs=self.segments.get(self.current_video,[]) if self.current_video else []
        self.timeline_canvas.set_data(segs,duration,self.player.position()/1000.0)

    def _timeline_segment_selected(self, row):
        if row < 0 or row >= self.table.rowCount(): return
        self.table.selectRow(row)
        item=self.table.item(row,1)
        if item:
            try:
                sec=float(item.text())
                self.player.setPosition(int(sec*1000))
            except Exception: pass
        self._selected_row_changed()

    def _timeline_segment_edited(self, row):
        if not self.current_video:
            return
        segs=self.segments.get(self.current_video,[])
        if row<0 or row>=len(segs):
            return
        self._snapshot()
        seg=segs[row]
        # Keep timing sane and sync the playhead/video preview.
        try:
            a=max(0.0,float(seg.get("start",0)))
            b=max(a+0.05,float(seg.get("end",a+0.05)))
            seg["start"]=a; seg["end"]=b
            self.player.setPosition(int(a*1000))
        except Exception:
            pass
        self._load_table()
        self.table.selectRow(row)
        self.timeline_canvas.set_selected(row)
        self._refresh_preview_overlays()
        self._status(f"Adjusted subtitle segment {row+1}.")

    def _timeline_seek_requested(self, seconds):
        self.player.setPosition(int(max(0,float(seconds))*1000))

    def _timeline_selected_index(self):
        if not self.current_video: return -1
        rows=self.table.selectionModel().selectedRows()
        if rows: return rows[0].row()
        return getattr(self.timeline_canvas,'selected',-1)

    def _split_selected_segment(self):
        idx=self._timeline_selected_index()
        segs=self.segments.get(self.current_video,[]) if self.current_video else []
        if idx<0 or idx>=len(segs):
            QMessageBox.information(self,"Split","Select a subtitle/voice segment first.")
            return
        seg=segs[idx]
        start=float(seg.get('start',0)); end=float(seg.get('end',start))
        cut=float(self.timeline_canvas.playhead)
        if cut <= start+0.05 or cut >= end-0.05:
            QMessageBox.information(self,"Split","Move the playhead inside the selected segment.")
            return
        self._snapshot()
        a=dict(seg); b=dict(seg)
        a['end']=cut
        b['start']=cut
        a['index']=idx+1; b['index']=idx+2
        segs[idx:idx+1]=[a,b]
        for n,x in enumerate(segs,1): x['index']=n
        self._load_table(); self.table.selectRow(idx+1)
        self._status("Segment split at playhead.")

    def _trim_selected_start(self):
        idx=self._timeline_selected_index(); segs=self.segments.get(self.current_video,[]) if self.current_video else []
        if idx<0 or idx>=len(segs): return
        cut=float(self.timeline_canvas.playhead); end=float(segs[idx].get('end',0))
        if cut>=end-0.05: return
        self._snapshot(); segs[idx]['start']=cut
        self._load_table(); self.table.selectRow(idx); self._status("Segment start trimmed.")

    def _trim_selected_end(self):
        idx=self._timeline_selected_index(); segs=self.segments.get(self.current_video,[]) if self.current_video else []
        if idx<0 or idx>=len(segs): return
        start=float(segs[idx].get('start',0)); cut=float(self.timeline_canvas.playhead)
        if cut<=start+0.05: return
        self._snapshot(); segs[idx]['end']=cut
        self._load_table(); self.table.selectRow(idx); self._status("Segment end trimmed.")

    def _delete_selected_segment(self):
        idx=self._timeline_selected_index(); segs=self.segments.get(self.current_video,[]) if self.current_video else []
        if idx<0 or idx>=len(segs): return
        self._snapshot(); del segs[idx]
        for n,x in enumerate(segs,1): x['index']=n
        self._load_table(); self._status("Segment deleted.")

    def _play_selected_voice(self):
        idx = self._timeline_selected_index()
        if idx < 0:
            QMessageBox.information(self, "Play Voice", "Select a segment first.")
            return
        self._play_segment_voice(idx)

    def _play_segment_voice(self, row):
        """Preview the generated TTS audio for one transcript row."""
        if not self.current_video:
            return
        segs = self.segments.get(self.current_video, [])
        if row < 0 or row >= len(segs):
            return
        path = str(segs[row].get("audio_path") or "").strip()
        if not path or not Path(path).exists():
            self._status(
                f"Segment {row + 1}: no generated voice yet. Run Generate Voice first."
            )
            return
        try:
            # Reuse a dedicated TTS preview player so the main video is not interrupted.
            if not hasattr(self, "_tts_player") or self._tts_player is None:
                self._tts_player = QMediaPlayer()
                self._tts_audio = QAudioOutput()
                self._tts_player.setAudioOutput(self._tts_audio)
                self._tts_audio.setVolume(1.0)
            self._tts_player.stop()
            self._tts_player.setSource(QUrl.fromLocalFile(path))
            self._tts_player.play()
            # Seek main playhead to the segment start for visual sync.
            try:
                self.player.setPosition(int(float(segs[row].get("start", 0)) * 1000))
            except Exception:
                pass
            self._status(
                f"Playing voice for segment {row + 1}: {Path(path).name}"
            )
        except Exception as exc:
            self._status(f"Could not play voice: {exc}")

    def _set_table_field(self, row, key, value):
        if not self.current_video:
            return
        segs = self.segments.get(self.current_video, [])
        if row < len(segs):
            self._snapshot()
            segs[row][key] = value
            if key == "gender":
                segs[row]["voice"] = ""

    def _table_changed(self, row, col):
        if not self.current_video:
            return
        segs = self.segments.get(self.current_video, [])
        if row >= len(segs):
            return

        self._snapshot()
        seg = segs[row]
        item = self.table.item(row, col)
        if item is None:
            return

        try:
            if col == 1:
                seg["start"] = float(item.text())
            elif col == 2:
                seg["end"] = float(item.text())
            elif col == 5:
                seg["original"] = item.text()
            elif col == 6:
                seg["translated"] = item.text()
            elif col == 7:
                seg["tts_text"] = item.text()
        except Exception:
            pass

    def _update_voice_list(self):
        self.cmb_speaker_voice.clear()
        lang = language_family(self.cmb_language.currentText())
        voices = VOICES.get(lang, {})
        for gender, vals in voices.items():
            for voice in vals:
                self.cmb_speaker_voice.addItem(
                    f"{gender} — {voice}", voice
                )

    def _select_voxcpm_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select speaker reference audio", "", "Audio (*.wav *.mp3 *.m4a *.flac)")
        if path:
            self.txt_voxcpm_ref.setText(path)
            self.settings["voxcpm_reference_wav"] = path

    def _apply_gemini_voice_to_rows(self):
        if not self.current_video:
            return
        voice = self.cmb_gemini_voice.currentText().strip()
        rows = sorted(set(i.row() for i in self.table.selectedItems()))
        segs = self.segments.get(self.current_video, [])
        self._snapshot()
        for r in rows:
            if r < len(segs):
                segs[r]["gemini_voice"] = voice
        self._status(f"Gemini voice applied to {len(rows)} row(s).")

    def _apply_voice_to_rows(self):
        if not self.current_video:
            return
        voice = self.cmb_speaker_voice.currentData()
        if not voice:
            return
        rows = sorted(set(i.row() for i in self.table.selectedItems()))
        segs = self.segments.get(self.current_video, [])
        self._snapshot()
        for r in rows:
            if r < len(segs):
                segs[r]["voice"] = voice
        self._status(f"Voice applied to {len(rows)} row(s).")

    # ---------------- AI actions ----------------

    def _hardware_gpu(self):
        """Safe GPU decision for Whisper / diarization / gender.

        Never returns True when CUDA probe fails — this prevents the whole
        process from exiting when the user selects Auto or NVIDIA CUDA on
        unsupported or broken GPU setups.
        """
        mode = self.cmb_gpu.currentText().strip()
        if mode == "CPU":
            return False
        if mode in ("NVIDIA CUDA", "CUDA"):
            ok = cuda_safe_for_whisper()
            if not ok:
                self._status(
                    "NVIDIA CUDA not usable (driver/VRAM/toolkit). Using CPU instead."
                )
            return ok
        # Auto: use CUDA only when the probe succeeds.
        return cuda_safe_for_whisper()

    def _load_activity_table(self):
        if not hasattr(self, "activity_table"):
            return
        self.activity_table.blockSignals(True)
        self.activity_table.setRowCount(0)
        rows = self.activities.get(self.current_video, []) if self.current_video else []
        narration_lines = []
        for i, row in enumerate(rows, 1):
            r = self.activity_table.rowCount()
            self.activity_table.insertRow(r)
            values = [
                str(i),
                f"{float(row.get('start', 0)):.2f}",
                f"{float(row.get('end', 0)):.2f}",
                str(row.get("activity", "")),
                str(row.get("narration", "")),
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(str(row.get("description", "")) if c in (3, 4) else value)
                self.activity_table.setItem(r, c, item)
            self.activity_table.setRowHeight(r, 44)
            if row.get("narration"):
                narration_lines.append(str(row["narration"]).strip())
        self.activity_table.blockSignals(False)
        combined = " ".join(x for x in narration_lines if x)
        self.txt_activity_narration.setPlainText(combined)
        if rows:
            self.lbl_activity_info.setText(
                f"{len(rows)} activities • Double-click to seek • "
                f"Narration: {self.settings.get('target_lang_name', 'Khmer')} • "
                "Click ‘Send Activities to Subtitle / TTS’ for the next step."
            )
        else:
            self.lbl_activity_info.setText("No activity analysis yet. Click 🎬 Analyze Activities.")

    def _activity_row_activated(self, row, column=0):
        if not self.current_video:
            return
        rows = self.activities.get(self.current_video, [])
        if 0 <= row < len(rows):
            try:
                self.player.setPosition(int(float(rows[row].get("start", 0)) * 1000))
                self._status(f"Seeked to activity {row + 1}: {rows[row].get('activity', '')}")
            except Exception as exc:
                self._status(f"Could not seek to activity: {exc}")

    def _copy_activity_narration(self):
        text = self.txt_activity_narration.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self._status("Activity narration copied to clipboard.")

    def _activity_to_timeline(self, replace=False, show_message=True):
        """Convert Gemini activity analysis into normal Dubby transcript/TTS rows.

        Activity analysis already returns reliable start/end timestamps and a
        narration field.  We reuse the existing transcript schema so the rows
        can immediately go through the normal edit -> TTS -> export pipeline.
        """
        if not self.current_video:
            if show_message:
                QMessageBox.warning(self, "Activity Timeline", "Import and select a video first.")
            return

        activities = list(self.activities.get(self.current_video, []) or [])
        if not activities:
            if show_message:
                QMessageBox.warning(
                    self, "Activity Timeline",
                    "Analyze Activities first. No activity results are available for this video."
                )
            return

        existing = list(self.segments.get(self.current_video, []) or [])
        if existing and not replace:
            reply = QMessageBox.question(
                self, "Add Activities to Timeline",
                f"This video already has {len(existing)} transcript row(s).\n\n"
                f"Add {len(activities)} activity row(s) after the existing rows?\n\n"
                "Choose No to replace the transcript instead.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.No:
                replace = True

        self._snapshot()

        # Build stable timestamped rows. Activity narration is already produced
        # in the selected target language by the Activity AI. It becomes both
        # subtitle and TTS text initially, and can still be edited independently.
        new_rows = []
        for i, row in enumerate(activities, 1):
            try:
                start = max(0.0, float(row.get("start", 0.0)))
                end = max(start + 0.10, float(row.get("end", start + 1.0)))
            except Exception:
                continue

            narration = str(row.get("narration") or row.get("description") or row.get("activity") or "").strip()
            description = str(row.get("description") or row.get("activity") or "").strip()
            activity = str(row.get("activity") or "Activity").strip()
            if not narration:
                continue

            new_rows.append({
                "index": i,
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": "Speaker 1",
                "speaker_raw": "ACTIVITY",
                "gender": self.cmb_gender.currentText() if self.cmb_gender.currentText() in ("Male", "Female") else "Auto",
                "detected_gender": "Unknown",
                "gender_confidence": 0.0,
                "gender_male": 0.0,
                "gender_female": 0.0,
                "original": description or activity,
                "translated": narration,
                "tts_text": narration,
                "voice": "",
                "audio_path": "",
                "activity": activity,
                "activity_description": description,
                "activity_source": "Gemini Activity Analysis",
            })

        if not new_rows:
            if show_message:
                QMessageBox.warning(self, "Activity Timeline", "No usable activity rows were found.")
            return

        if replace:
            combined = new_rows
        else:
            combined = existing + new_rows

        # Re-index after append/replace and sort by actual video time.
        combined.sort(key=lambda x: (float(x.get("start", 0)), float(x.get("end", 0))))
        for idx, seg in enumerate(combined, 1):
            seg["index"] = idx

        self.segments[self.current_video] = combined
        self._load_table()
        self._load_activity_table()

        mode = "replaced transcript with" if replace else "added"
        self._status(
            f"Activity timeline: {mode} {len(new_rows)} activity rows. "
            "Subtitle/TTS text is ready for editing and voice generation."
        )
        if show_message:
            QMessageBox.information(
                self, "Activity Timeline Ready",
                f"{len(new_rows)} activity rows were {mode} the transcript.\n\n"
                "Next: review/edit Subtitle and TTS, then click ③ Generate Voice All.\n"
                "The original activity timestamps are preserved."
            )

    def _start_activity_full_auto(self):
        """One-click pipeline: Analyze Activities -> Voice -> Export."""
        if not self._require_license():
            return
        if not self.videos:
            QMessageBox.warning(self, "Activity FULL AUTO", "Import videos first.")
            return
        if len(self.videos) > MAX_VIDEOS:
            QMessageBox.warning(self, "Activity FULL AUTO", f"Maximum {MAX_VIDEOS} videos.")
            return
        if not _gemini_api_keys(self.settings):
            QMessageBox.warning(
                self, "Gemini API Key",
                "Add at least one Gemini API key in AI / API Settings first."
            )
            return

        self._collect_settings()
        self.activity_full_auto_active = True
        self.activity_full_auto_started_at = time.time()
        self.batch_export_failures.clear()
        self.batch_export_skipped.clear()
        self.btn_activity_full_auto.setEnabled(False)
        self.btn_activity.setEnabled(False)
        self.btn_activity_all.setEnabled(False)
        self._status(
            f"ACTIVITY FULL AUTO started for {len(self.videos)} video(s): "
            "Analyze Activities -> Generate Voice -> Export."
        )
        self.progress.setValue(0)
        self.batch_activity_queue = list(self.videos.keys())
        self.batch_activity_active = True
        self._start_next_batch_activity()

    def _start_activity_analysis(self):
        if not self.current_video:
            QMessageBox.warning(self, "Warning", "Import and select a video first.")
            return
        if not _gemini_api_keys(self.settings):
            QMessageBox.warning(self, "Gemini API Key", "Add at least one Gemini API key in AI / API Settings first.")
            return
        self.btn_activity.setEnabled(False)
        self.btn_activity_all.setEnabled(False)
        self.progress.setValue(0)
        settings = self._collect_settings()
        worker = ActivityAnalysisWorker(
            self.videos[self.current_video], self.current_video, settings=settings
        )
        worker.progress.connect(self._status)
        worker.finished.connect(self._activity_done)
        worker.error.connect(self._activity_error)
        self.workers.append(worker)
        worker.start()

    def _activity_done(self, name, rows):
        """Finish Activity Analysis and immediately build the Subtitle/TTS timeline.

        Activity Analysis is intended to be a complete workflow step, not just a
        report.  Gemini gives us reliable start/end timestamps plus narration,
        so as soon as analysis finishes for the currently selected video we
        convert those activities into the normal Dubby segment schema.  This
        makes the result visible in Timeline/Transcript immediately and makes
        it available to Translate/Edit/TTS/Export without another button click.
        """
        self.activities[name] = rows
        auto_loaded = False
        if name == self.current_video:
            self._load_activity_table()
            try:
                # Activity narration is already generated in the target language.
                # Replace the current transcript so Generate Voice All will not
                # accidentally speak both the old transcript and the activities.
                self._activity_to_timeline(replace=True, show_message=False)
                auto_loaded = True
            except Exception as exc:
                # Never turn a successful Gemini analysis into a failed job just
                # because timeline/UI loading had a problem.
                self._status(f"Activity timeline auto-load failed: {exc}")
                auto_loaded = False
        self.btn_activity.setEnabled(True)
        self.btn_activity_all.setEnabled(True)
        self.progress.setValue(100)
        if auto_loaded:
            self._status(
                f"Activity analysis complete: {name} → {len(rows)} activities. "
                "Timeline/Subtitle/TTS loaded automatically."
            )
        else:
            self._status(f"Activity analysis complete: {name} → {len(rows)} activities.")
        if getattr(self, "batch_activity_active", False):
            self._start_next_batch_activity()

    def _activity_error(self, error):
        self.btn_activity.setEnabled(True)
        self.btn_activity_all.setEnabled(True)
        self.batch_activity_active = False
        self.batch_activity_queue.clear()
        if getattr(self, "activity_full_auto_active", False):
            self.activity_full_auto_active = False
            self.btn_activity_full_auto.setEnabled(True)
            self._status(f"Activity FULL AUTO stopped: {error}")
        else:
            self._status(f"Activity analysis error: {error}")
        self.progress.setValue(0)
        QMessageBox.critical(self, "Activity Analysis Error", str(error))

    def _start_activity_all(self):
        if not self.videos:
            QMessageBox.warning(self, "Warning", "Import videos first.")
            return
        if not _gemini_api_keys(self.settings):
            QMessageBox.warning(self, "Gemini API Key", "Add at least one Gemini API key in AI / API Settings first.")
            return
        self.batch_activity_queue = list(self.videos.keys())
        self.batch_activity_active = True
        self.btn_activity.setEnabled(False)
        self.btn_activity_all.setEnabled(False)
        self.progress.setValue(0)
        self._status(f"Starting activity analysis: {len(self.batch_activity_queue)} video(s)...")
        self._start_next_batch_activity()

    def _start_next_batch_activity(self):
        if not self.batch_activity_active:
            return
        if not self.batch_activity_queue:
            self.batch_activity_active = False
            self.btn_activity.setEnabled(True)
            self.btn_activity_all.setEnabled(True)
            if getattr(self, "activity_full_auto_active", False):
                self.progress.setValue(35)
                self._status(
                    "Activity analysis complete. Starting automatic voice generation "
                    "for all activity narrations..."
                )
                self._start_tts_all(names=list(self.videos.keys()))
            else:
                self.progress.setValue(100)
                self._status("Batch activity analysis complete.")
            return
        name = self.batch_activity_queue.pop(0)
        self.current_video = name
        self.cmb_video.blockSignals(True)
        self.cmb_video.setCurrentText(name)
        self.cmb_video.blockSignals(False)
        settings = self._collect_settings()
        worker = ActivityAnalysisWorker(self.videos[name], name, settings=settings)
        worker.progress.connect(self._status)
        worker.finished.connect(self._activity_done)
        worker.error.connect(self._batch_activity_error)
        self.workers.append(worker)
        worker.start()

    def _batch_activity_error(self, error):
        self._status(f"Activity analysis skipped current video: {error}")
        self._start_next_batch_activity()

    def _start_transcribe(self):
        if not self.current_video:
            QMessageBox.warning(self, "Warning", "Import and select a video first.")
            return

        self.btn_transcribe.setEnabled(False)
        self.progress.setValue(0)
        default_gender = self.cmb_gender.currentText()
        settings = self._collect_settings()
        if self.cmb_transcribe_engine.currentText() == "Google Gemini + Analyze":
            worker = GeminiTranscribeWorker(
                self.videos[self.current_video], self.current_video,
                default_gender=default_gender, settings=settings
            )
        else:
            worker = TranscribeWorker(
                self.videos[self.current_video], self.current_video,
                self.cmb_model.currentText(), self._hardware_gpu(),
                default_gender=default_gender, settings=settings
            )
        worker.progress.connect(self._status)
        worker.finished.connect(self._transcribe_done)
        worker.error.connect(self._worker_error)
        self.workers.append(worker)
        worker.start()

    def _transcribe_done(self, name, segs, language, hardware):
        self.segments[name] = segs
        self._load_table() if name == self.current_video else None
        self.btn_transcribe.setEnabled(True)
        self.progress.setValue(35)

        # Summarize detected genders so the user can verify Auto worked.
        gender_counts = {}
        for s in segs:
            g = s.get("gender") or s.get("detected_gender") or "?"
            gender_counts[g] = gender_counts.get(g, 0) + 1
        gender_txt = ", ".join(f"{k}×{v}" for k, v in sorted(gender_counts.items()))
        gsum = hardware.get("gender_summary") or {}
        conf_bits = []
        for raw, info in gsum.items():
            conf_bits.append(
                f"{info.get('gender', '?')} {float(info.get('confidence') or 0):.0%}"
            )
        conf_txt = " | ".join(conf_bits) if conf_bits else "n/a"

        if hardware.get("diarization") == "Community-1":
            self._status(
                f"Transcribed {name}: {len(segs)} segs | "
                f"{hardware.get('speaker_count', 0)} speaker(s) | "
                f"Gender Auto: {gender_txt} ({conf_txt}) | "
                f"{hardware.get('device')}/{hardware.get('compute', 'auto')}"
            )
        else:
            err = str(hardware.get("diarization_error") or "no multi-speaker model")[:180]
            self._status(
                f"Transcribed {name}: {len(segs)} segs | "
                f"1 speaker (diarization off: {err}) | "
                f"Gender: {gender_txt} ({conf_txt})"
            )
        if self.batch_transcribe_active:
            self._start_next_batch_transcription()

    def _gender_done(self, name, result):
        # Legacy compatibility hook; v3 performs gender detection per speaker
        # during transcription and therefore does not start a second worker.
        self._status("Per-speaker gender detection is already included in transcription.")

    def _fill_gender_rows(self):
        if not self.current_video:
            return
        segs = self.segments.get(self.current_video, [])
        if not segs:
            return
        choice = self.cmb_fill_gender.currentText()
        if choice == "Default":
            choice = self.cmb_gender.currentText()
        if choice not in ("Male", "Female", "Auto"):
            return
        selected_rows = sorted(set(i.row() for i in self.table.selectedItems()))
        rows = selected_rows if selected_rows else list(range(len(segs)))
        self._snapshot()
        for r in rows:
            if r < len(segs):
                seg = segs[r]
                if choice == "Auto":
                    # Restore the ML result already computed during
                    # Transcribe + Analyze instead of erasing it.
                    detected = seg.get("detected_gender", "Unknown")
                    seg["gender"] = detected if detected in ("Male", "Female") else "Auto"
                else:
                    seg["gender"] = choice
                seg["voice"] = ""
        self._load_table()
        self._status(f"Gender applied to {len(rows)} row(s): {choice}.")

    def _copy_all_rows(self):
        if not self.current_video:
            return
        segs = self.segments.get(self.current_video, [])
        if not segs:
            return
        headers = ["#", "Start", "End", "Speaker", "Gender", "Original", "Subtitle", "TTS"]
        lines = ["\t".join(headers)]
        for i, seg in enumerate(segs, 1):
            lines.append("\t".join([
                str(seg.get("index", i)),
                str(seg.get("start", "")),
                str(seg.get("end", "")),
                str(seg.get("speaker", "Speaker 1")),
                str(seg.get("gender", "Auto")),
                str(seg.get("original", "")),
                str(seg.get("translated", "")),
                str(seg.get("tts_text", "")),
            ]))
        QApplication.clipboard().setText("\n".join(lines))
        self._status(f"Copied all {len(segs)} transcript rows to clipboard.")

    def _copy_column(self, key="translated"):
        if not self.current_video:
            return
        segs = self.segments.get(self.current_video, [])
        if not segs:
            return
        text = "\n".join(str(seg.get(key, "")) for seg in segs)
        QApplication.clipboard().setText(text)
        label = "Subtitle" if key == "translated" else "TTS"
        self._status(f"Copied {len(segs)} {label} rows to clipboard.")

    def _copy_tts_column(self):
        self._copy_column("tts_text")

    def _clipboard_lines(self):
        """Parse clipboard for ChatGPT / Excel / plain-line paste into Subtitle or TTS.

        Accepts:
          - one phrase per line
          - numbered lists: 1. text / 1) text / 1: text
          - markdown bullets: - text / * text
          - quoted lines
          - full Dubby TSV tables
          - blank lines are skipped (ChatGPT often adds them)
        """
        raw = QApplication.clipboard().text()
        if not raw or not str(raw).strip():
            return []

        # Full project table paste → handled by caller with column index.
        rows = [line.split("\t") for line in raw.splitlines() if line.strip()]
        if (
            rows and len(rows) >= 2 and len(rows[0]) >= 7
            and rows[0][0].strip() in ("#", "No", "Index", "index")
        ):
            return None  # signal: TSV table

        lines = []
        for line in raw.splitlines():
            line = line.rstrip("\r")
            s = line.strip()
            if not s:
                continue
            # Strip common ChatGPT / list prefixes.
            s = re.sub(r"^[\-\*\u2022]\s+", "", s)
            s = re.sub(r"^\d+\s*[\.\)\:\-]\s*", "", s)
            s = re.sub(r"^\[\d+\]\s*", "", s)
            # Strip wrapping quotes once.
            if len(s) >= 2 and (
                (s[0] == s[-1] == '"')
                or (s[0] == s[-1] == "'")
                or (s[0] == "\u201c" and s[-1] == "\u201d")
            ):
                s = s[1:-1].strip()
            # First column if someone pasted TSV without header.
            if "\t" in s:
                s = s.split("\t", 1)[0].strip()
            if s:
                lines.append(s)
        return lines

    def _paste_column_to_segments(self, key="translated", also_tts=False):
        """Paste ChatGPT / external translation into Subtitle or TTS with flexible alignment."""
        if not self.current_video:
            QMessageBox.information(self, "Paste", "Select a video first.")
            return
        segs = self.segments.get(self.current_video, [])
        if not segs:
            QMessageBox.information(self, "Paste", "Transcribe the video first.")
            return

        raw = QApplication.clipboard().text()
        values = self._clipboard_lines()

        # Full TSV table from "Copy All Rows"
        if values is None:
            rows = [line.split("\t") for line in raw.splitlines() if line.strip()]
            col = 6 if key == "translated" else 7
            # Header may shift columns — find by name.
            header = [h.strip().lower() for h in rows[0]]
            if key == "translated":
                for name in ("subtitle", "translated", "translation"):
                    if name in header:
                        col = header.index(name)
                        break
            else:
                for name in ("tts", "tts_text", "voice"):
                    if name in header:
                        col = header.index(name)
                        break
            values = [r[col].strip() if len(r) > col else "" for r in rows[1:]]

        if not values:
            QMessageBox.information(
                self, "Paste",
                "Clipboard is empty or could not be parsed.\n\n"
                "Tip: Copy Original → paste into ChatGPT → copy the translation "
                "(one line per subtitle) → Paste to Subtitle."
            )
            return

        selected = sorted(set(i.row() for i in self.table.selectedItems()))
        n_seg = len(segs)
        n_val = len(values)
        label = "Subtitle" if key == "translated" else "TTS"

        # Decide mapping
        if n_val == 1 and selected:
            targets = selected
            mapped = [values[0]] * len(targets)
        elif n_val == n_seg:
            targets = list(range(n_seg))
            mapped = values
        elif selected and n_val == len(selected):
            targets = selected
            mapped = values
        elif selected and n_val < len(selected):
            # Fill selected rows from the first, leave rest of selection untouched if short
            targets = selected[:n_val]
            mapped = values
        elif not selected and n_val < n_seg:
            # ChatGPT returned fewer lines — paste from the top, ask user.
            reply = QMessageBox.question(
                self, "Paste count mismatch",
                f"{label}: clipboard has {n_val} lines, transcript has {n_seg} rows.\n\n"
                f"Paste the {n_val} lines into the first {n_val} rows?\n"
                "(Tip: select specific rows first to paste only there.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            targets = list(range(n_val))
            mapped = values
        elif n_val > n_seg:
            reply = QMessageBox.question(
                self, "Paste count mismatch",
                f"{label}: clipboard has {n_val} lines, transcript has only {n_seg} rows.\n\n"
                f"Use the first {n_seg} lines?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            targets = list(range(n_seg))
            mapped = values[:n_seg]
        else:
            # selected empty, n_val != n_seg already handled; fallback top fill
            targets = list(range(min(n_val, n_seg)))
            mapped = values[:len(targets)]

        self._snapshot()
        for index, r in enumerate(targets):
            if r >= len(segs):
                continue
            value = mapped[index].strip() if index < len(mapped) else ""
            if key == "translated":
                segs[r]["translated"] = value
                if also_tts:
                    segs[r]["tts_text"] = value
                elif not str(segs[r].get("tts_text") or "").strip():
                    # Fill empty TTS only — do not overwrite a custom TTS line.
                    segs[r]["tts_text"] = value
            else:
                segs[r]["tts_text"] = value

        self._load_table()
        # Reselect targets for review
        try:
            self.table.clearSelection()
            for r in targets:
                self.table.selectRow(r)
        except Exception:
            pass
        self._status(
            f"Pasted {len(targets)} line(s) into {label}"
            + (" + TTS" if key == "translated" else "")
            + f" (clipboard had {n_val})."
        )

    def _paste_tts_column(self):
        self._paste_column_to_segments("tts_text")

    def _paste_subtitle_and_tts(self):
        """Paste ChatGPT translation into both Subtitle and TTS columns."""
        self._paste_column_to_segments("translated", also_tts=True)

    def _copy_original_column(self):
        """Copy Original lines for ChatGPT translation (one line per segment)."""
        self._copy_column("original")

    def _editor_khmer_font(self):
        """Return (family, size, QFont) for transcript Khmer text."""
        family = "Khmer UI"
        size = 22
        try:
            if hasattr(self, "cmb_editor_font"):
                family = self.cmb_editor_font.currentText().strip() or family
            elif hasattr(self, "cmb_font"):
                family = self.cmb_font.currentText().strip() or family
            if hasattr(self, "sp_editor_font_size"):
                size = int(self.sp_editor_font_size.value())
        except Exception:
            pass
        size = max(12, min(48, size))
        font = QFont(family, size)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        return family, size, font

    def _apply_editor_font(self):
        """Apply larger Khmer font to table Subtitle/TTS/Original + editor box."""
        family, size, font = self._editor_khmer_font()
        # Slightly larger for Subtitle / TTS body text (harder to read when small).
        body_font = QFont(family, size)
        body_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        edit_font = QFont(family, min(48, size + 4))
        edit_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)

        if hasattr(self, "table") and self.table is not None:
            self.table.setFont(body_font)
            row_h = max(44, int(size * 2.8))
            self.table.verticalHeader().setDefaultSectionSize(row_h)
            self.table.setStyleSheet(
                f"""
                QTableWidget {{
                    font-family: '{family}';
                    font-size: {size}px;
                }}
                QTableWidget::item {{
                    padding: 6px 8px;
                }}
                """
            )
            # Force font onto every text cell (Qt often ignores table font on items).
            for r in range(self.table.rowCount()):
                self.table.setRowHeight(r, row_h)
                for c in range(self.table.columnCount()):
                    item = self.table.item(r, c)
                    if item is not None:
                        # Subtitle / TTS / Original get the full Khmer body size.
                        if c in (5, 6, 7):
                            item.setFont(body_font)
                        else:
                            item.setFont(QFont(family, max(12, size - 2)))

            # Prefer width for Subtitle + TTS so Khmer is not clipped to "...".
            try:
                hdr = self.table.horizontalHeader()
                hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
                hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
                hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
                hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
                hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
                hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
                hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
                hdr.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
            except Exception:
                pass

        if hasattr(self, "txt_subtitle_edit") and self.txt_subtitle_edit is not None:
            self.txt_subtitle_edit.setFont(edit_font)

        self.settings["editor_font_family"] = family
        self.settings["editor_font_size"] = size

    def _selected_row_changed(self):
        if not self.current_video or not hasattr(self, "txt_subtitle_edit"):
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        segs = self.segments.get(self.current_video, [])
        if row >= len(segs):
            return
        self.txt_subtitle_edit.blockSignals(True)
        self.txt_subtitle_edit.setPlainText(segs[row].get("translated", ""))
        self.txt_subtitle_edit.blockSignals(False)
        rows=self.table.selectionModel().selectedRows()
        if rows and hasattr(self, "timeline_canvas"):
            self.timeline_canvas.set_selected(rows[0].row())

    def _apply_subtitle_editor(self):
        self._apply_editor_text("translated")

    def _apply_tts_editor(self):
        self._apply_editor_text("tts_text")

    def _apply_editor_text(self, key):
        if not self.current_video:
            return
        rows = sorted(set(i.row() for i in self.table.selectedItems()))
        if not rows:
            QMessageBox.information(self, "Subtitle Editor", "Select one or more transcript rows first.")
            return
        text = self.txt_subtitle_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Subtitle Editor", "Enter or paste text first.")
            return
        segs = self.segments.get(self.current_video, [])
        self._snapshot()
        for r in rows:
            if r < len(segs):
                segs[r][key] = text
                if key == "translated" and not segs[r].get("tts_text"):
                    segs[r]["tts_text"] = text
        self._load_table()
        self._status(f"Updated {key} for {len(rows)} row(s).")

    def _focus_transcript_review(self):
        """Top-bar task: open Transcript Review window (not under Program Monitor)."""
        dlg = getattr(self, "transcript_dialog", None)
        table = getattr(self, "table", None)
        if dlg is None:
            self._status("Transcript Review is not available.")
            return
        try:
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        except Exception as exc:
            self._status(f"Could not open Transcript Review: {exc}")
            return
        if table is not None:
            try:
                table.setFocus()
                if table.rowCount() > 0 and not table.selectedItems():
                    table.selectRow(0)
            except Exception:
                pass
        # Do not overwrite the fixed status area when opening Transcript Review.
        # The status area should keep showing the current processing/status message.

    def _toggle_timeline_maximize(self):

        self.timeline_maximized = not self.timeline_maximized
        left_scroll = getattr(self, "left_panel_scroll", None)

        if self.timeline_maximized:
            if left_scroll:
                left_scroll.hide()
            self.right_panel.hide()
            self.btn_timeline_max.setText("↙ Restore")
            self._status("Maximized workspace — Preview stays above Timeline.")
        else:
            if left_scroll:
                left_scroll.show()
            self.right_panel.show()
            self.btn_timeline_max.setText("⛶ Maximize")
            self._status("Workspace restored.")

    def _start_full_process(self):
        """Run the complete batch pipeline: transcribe -> translate -> TTS -> export."""
        if not self._require_license():
            return
        if not self.videos:
            QMessageBox.warning(self, "Full Auto", "Import videos first.")
            return
        if len(self.videos) > MAX_VIDEOS:
            QMessageBox.warning(self, "Full Auto", f"Maximum {MAX_VIDEOS} videos.")
            return
        self._collect_settings()
        self.full_process_active = True
        self.full_process_started_at = time.time()
        self.batch_export_failures.clear()
        self.batch_export_skipped.clear()
        self.btn_full_process.setEnabled(False)
        self._status(f"FULL AUTO started for {len(self.videos)} video(s). Step 1/4: transcription...")
        self.progress.setValue(0)
        self._start_transcribe_all()

    def _start_transcribe_all(self):
        if not self.videos:
            QMessageBox.warning(self, "Warning", "Import videos first.")
            return
        self.batch_transcribe_queue = list(self.videos.keys())
        self.batch_transcribe_active = True
        self.btn_transcribe_all.setEnabled(False)
        self.btn_transcribe.setEnabled(False)
        self.progress.setValue(0)
        self._status(f"Starting batch transcription: {len(self.batch_transcribe_queue)} video(s)...")
        self._start_next_batch_transcription()

    def _start_next_batch_transcription(self):
        if not self.batch_transcribe_active:
            return
        if not self.batch_transcribe_queue:
            self.batch_transcribe_active = False
            self.btn_transcribe_all.setEnabled(True)
            self.btn_transcribe.setEnabled(True)
            self.progress.setValue(25 if self.full_process_active else 100)
            if self.full_process_active:
                self._status("Step 1/4 complete. Starting Step 2/4: translation...")
                self._start_translate_all()
            else:
                self._status("Batch transcription complete.")
            return
        name = self.batch_transcribe_queue.pop(0)
        self.current_video = name
        self.cmb_video.blockSignals(True)
        self.cmb_video.setCurrentText(name)
        self.cmb_video.blockSignals(False)
        default_gender = self.cmb_gender.currentText()
        settings = self._collect_settings()
        if self.cmb_transcribe_engine.currentText() == "Google Gemini + Analyze":
            worker = GeminiTranscribeWorker(
                self.videos[name], name, default_gender=default_gender, settings=settings
            )
        else:
            worker = TranscribeWorker(
                self.videos[name], name, self.cmb_model.currentText(),
                self._hardware_gpu(), default_gender=default_gender, settings=settings
            )
        worker.progress.connect(self._status)
        worker.finished.connect(self._transcribe_done)
        worker.error.connect(self._batch_transcribe_error)
        self.workers.append(worker)
        worker.start()

    def _batch_transcribe_error(self, error):
        self._status(f"Batch transcription skipped current video: {error}")
        self._start_next_batch_transcription()

    def _start_translate_all(self, names=None):
        # QPushButton.clicked emits a boolean. When this method is connected
        # directly to clicked, that boolean must NOT be treated as a video
        # name list.
        if isinstance(names, bool):
            names = None

        if names is None:
            names = list(self.videos.keys())
        elif isinstance(names, str):
            names = [names]
        else:
            names = list(names)

        names = [n for n in names if self.segments.get(n)]
        if not names:
            QMessageBox.warning(self, "Translate All", "Transcribe at least one video first.")
            if self.full_process_active:
                self.full_process_active = False
                self.btn_full_process.setEnabled(True)
                self._status("FULL AUTO stopped: no transcription results were available.")
            return
        self.batch_translate_queue = names
        self.batch_translate_active = True
        self.btn_translate.setEnabled(False)
        self.btn_transcribe.setEnabled(False)
        self.btn_transcribe_all.setEnabled(False)
        self.btn_tts.setEnabled(False)
        self.progress.setValue(0)
        self._status(f"Starting batch translation: {len(names)} video(s)...")
        self._start_next_batch_translate()

    def _start_next_batch_translate(self):
        if not self.batch_translate_active:
            return
        if not self.batch_translate_queue:
            self.batch_translate_active = False
            self.btn_translate.setEnabled(True)
            self.btn_transcribe.setEnabled(True)
            self.btn_transcribe_all.setEnabled(True)
            self.btn_tts.setEnabled(True)
            self.progress.setValue(50 if self.full_process_active else 65)
            if self.full_process_active:
                self._status("Step 2/4 complete. Starting Step 3/4: voice generation...")
                self._start_tts_all()
            else:
                self._status("Batch translation complete. Ready for Generate Voice All.")
            return
        name = self.batch_translate_queue.pop(0)
        self.current_video = name
        self.cmb_video.blockSignals(True)
        self.cmb_video.setCurrentText(name)
        self.cmb_video.blockSignals(False)
        target = LANGUAGES[self.cmb_language.currentText()]
        provider = self.settings.get("translation_provider", "Google Translate")
        
        # Determine which provider to use
        if provider == "OpenAI ChatGPT":
            worker = TranslateWorker(
                name, self.segments[name], target,
                provider="openai",
                openai_key=self.settings.get("openai_api_key", ""),
                openai_model=self.settings.get("openai_model", "gpt-3.5-turbo"),
                openai_prompt=self.settings.get("translation_prompt", None)
            )
        else:
            # Default to Google Translate (existing behavior)
            worker = TranslateWorker(
                name, self.segments[name], target,
                provider="google"
            )
        
        worker.progress.connect(lambda t: self._status(t))
        worker.finished.connect(self._translate_done)
        worker.error.connect(self._batch_translate_error)
        self.workers.append(worker)
        worker.start()

    def _batch_translate_error(self, error):
        self._status(f"Translation skipped current video: {error}")
        self._start_next_batch_translate()

    def _translate_done(self, name, segs):
        self.segments[name] = segs
        if name == self.current_video:
            self._load_table()
        if self.batch_translate_active:
            self._status(f"Translated {name}. Continuing batch...")
            self._start_next_batch_translate()
        else:
            self.btn_translate.setEnabled(True)
            self.progress.setValue(65)
            self._status("Translation complete. You can edit Subtitle and TTS directly in the table.")

    def _start_tts_all(self, names=None):
        # QPushButton.clicked emits a boolean. Do not treat that signal
        # value as a list of video names.
        if not self._require_license():
            return
        if isinstance(names, bool):
            names = None

        if names is None:
            names = list(self.videos.keys())
        elif isinstance(names, str):
            names = [names]
        else:
            names = list(names)

        names = [n for n in names if self.segments.get(n)]
        missing = [n for n in names if not any(s.get("tts_text", "").strip() for s in self.segments.get(n, []))]
        if missing:
            QMessageBox.warning(
                self, "Generate Voice All",
                "These videos have no Subtitle/TTS text. Translate or paste the text first:\n\n"
                + "\n".join(missing)
            )
            if self.full_process_active and len(missing) == len(names):
                self.full_process_active = False
                self.btn_full_process.setEnabled(True)
                self._status("FULL AUTO stopped: no translated text was available for voice generation.")
            return
        if not names:
            QMessageBox.warning(self, "Generate Voice All", "Transcribe at least one video first.")
            if self.full_process_active:
                self.full_process_active = False
                self.btn_full_process.setEnabled(True)
            return
        self.batch_tts_queue = names
        self.batch_tts_active = True
        self.btn_tts.setEnabled(False)
        self.btn_translate.setEnabled(False)
        self.btn_transcribe.setEnabled(False)
        self.btn_transcribe_all.setEnabled(False)
        self.progress.setValue(0)
        self._status(f"Starting batch voice generation: {len(names)} video(s)...")
        self._start_next_batch_tts()

    def _start_next_batch_tts(self):
        if not self.batch_tts_active:
            return
        if not self.batch_tts_queue:
            self.batch_tts_active = False
            self.btn_tts.setEnabled(True)
            self.btn_translate.setEnabled(True)
            self.btn_transcribe.setEnabled(True)
            self.btn_transcribe_all.setEnabled(True)
            self.progress.setValue(75 if (self.full_process_active or self.activity_full_auto_active) else 80)
            if self.full_process_active:
                self._status("Step 3/4 complete. Starting Step 4/4: export...")
                self._start_export_all()
            elif self.activity_full_auto_active:
                self._status("Activity voice generation complete. Starting automatic export...")
                self._start_export_all()
            else:
                self._status("Batch voice generation complete. Ready to Export All.")
            return
        name = self.batch_tts_queue.pop(0)
        self.current_video = name
        self.cmb_video.blockSignals(True)
        self.cmb_video.setCurrentText(name)
        self.cmb_video.blockSignals(False)
        worker = TTSWorker(
            name, self.segments[name], self.cmb_language.currentText(),
            self.cmb_gender.currentText() if self.cmb_gender.currentText() != "Auto" else "Female",
            self.sld_speed.value() / 100.0,
            self._collect_settings()
        )
        worker.progress.connect(lambda p, t, n=name: (self.progress.setValue(p), self._status(f"{n}: {t}")))
        worker.finished.connect(self._tts_done)
        worker.error.connect(self._batch_tts_error)
        self.workers.append(worker)
        worker.start()

    def _batch_tts_error(self, error):
        self._status(f"Voice generation skipped current video: {error}")
        self._start_next_batch_tts()

    def _tts_done(self, name, segs):
        self.segments[name] = segs
        if name == self.current_video:
            self._load_table()
        if self.batch_tts_active:
            self._status(f"Generated voice for {name}. Continuing batch...")
            self._start_next_batch_tts()
        else:
            self.btn_tts.setEnabled(True)
            self.progress.setValue(80)
            self._status("Voice generation complete. Ready to export.")

    # ---------------- Export ----------------

    def _collect_settings(self):
        self.settings.update({
            "target_lang": LANGUAGES[self.cmb_language.currentText()],
            "target_lang_name": self.cmb_language.currentText(),
            # IMPORTANT: preserve Auto. The previous version converted Auto
            # to Female here, so TranscribeWorker never received Auto detection.
            "gender_default": self.cmb_gender.currentText(),
            "transcribe_engine": self.cmb_transcribe_engine.currentText() if hasattr(self, "cmb_transcribe_engine") else self.settings.get("transcribe_engine", "Local Whisper + Analyze"),
            "voice_speed": self.sld_speed.value() / 100.0,
            "edge_pitch_hz": self.sld_edge_pitch.value(),
            "edge_volume_pct": self.sld_edge_volume.value(),
            "edge_concurrency": self.sp_edge_concurrency.value(),
            "edge_max_rate_pct": int(self.settings.get("edge_max_rate_pct", 45) or 45),
            "tts_max_stretch": float(self.settings.get("tts_max_stretch", 0.38) or 0.38),
            "tts_pad_short": bool(self.settings.get("tts_pad_short", True)),
            "tts_provider": self.cmb_tts_provider.currentText(),
            "voxcpm_device": self.settings.get("voxcpm_device", "Auto"),
            "voxcpm_model": self.settings.get("voxcpm_model", "openbmb/VoxCPM2"),
            "voxcpm_style": self.settings.get("voxcpm_style", "Natural conversational voice, clear pronunciation, realistic emotion, natural pauses."),
            "huggingface_token": self.settings.get("huggingface_token", ""),
            "min_speakers": self.settings.get("min_speakers", 0),
            "max_speakers": self.settings.get("max_speakers", 0),
            "voxcpm_reference_wav": self.txt_voxcpm_ref.text().strip() if hasattr(self, "txt_voxcpm_ref") else self.settings.get("voxcpm_reference_wav", ""),
            "voxcpm_prompt_text": self.settings.get("voxcpm_prompt_text", ""),
            "keep_bgm": self.chk_bgm.isChecked(),
            "vocal_strength": self.cmb_strength.currentText(),
            "bgm_volume": self.sld_bgm.value() / 100.0,
            "bgm_duck_db": -float(self.sld_duck.value()),
            "add_subtitles": self.chk_subtitles.isChecked(),
            "subtitle_font_size": self.sp_sub_size.value(),
            "subtitle_color": self.sub_color,
            "subtitle_bar_height": self.sp_bar_h.value(),
            "subtitle_bar_opacity": self.sld_bar_op.value() / 100.0,
            "subtitle_blur_radius": self.sp_blur_radius.value() if hasattr(self, "sp_blur_radius") else 18,
            "subtitle_position_x": self.sp_sub_x.value() if hasattr(self, "sp_sub_x") else 50,
            "subtitle_position_y": self.sp_sub_y.value() if hasattr(self, "sp_sub_y") else 88,
            "font_family": self.cmb_font.currentText(),
            "add_title": self.chk_title.isChecked(),
            "title_text": self.txt_title.text(),
            "title_size": self.sp_title_size.value(),
            "title_color": self.title_color,
            "title_position": self.cmb_title_pos.currentText(),
            "title_position_x": self.sp_title_x.value(),
            "title_position_y": self.sp_title_y.value(),
            "add_logo": self.chk_logo.isChecked(),
            "logo_path": self.logo_path or "",
            "logo_width": self.sp_logo_w.value(),
            "logo_opacity": self.sld_logo_op.value() / 100.0,
            "logo_position": self.cmb_logo_pos.currentText(),
            "logo_position_x": self.sp_logo_x.value(),
            "logo_position_y": self.sp_logo_y.value(),
            "aspect_ratio": self.cmb_aspect.currentText(),
            "preset": self.cmb_preset.currentText(),
            "export_mode": self.cmb_export_mode.currentText() if hasattr(self, "cmb_export_mode") else self.settings.get("export_mode", "Turbo (FFmpeg + NVENC)"),
            "crf": self.sp_crf.value(),
            "threads": max(2, os.cpu_count() or 6),
            "export_hwaccel": self.cmb_export_hwaccel.currentText(),
            "shutdown_when_finished": bool(
                self.chk_shutdown.isChecked() if hasattr(self, "chk_shutdown") else False
            ),
            "output_folder": (
                self.txt_output_folder.text().strip()
                if hasattr(self, "txt_output_folder")
                else str(self.settings.get("output_folder") or OUTPUT_DIR)
            ),
        })
        return dict(self.settings)


    def _browse_output_folder(self):
        current = ""
        if hasattr(self, "txt_output_folder"):
            current = self.txt_output_folder.text().strip()
        if not current:
            current = str(self.settings.get("output_folder") or OUTPUT_DIR)
        folder = QFileDialog.getExistingDirectory(
            self, "Choose Output Folder", current or str(OUTPUT_DIR)
        )
        if folder:
            self.txt_output_folder.setText(folder)
            self.settings["output_folder"] = folder
            try:
                save_user_settings(self.settings)
            except Exception:
                pass
            self._status(f"Output folder: {folder}")

    def _open_output_folder(self):
        folder = (
            self.txt_output_folder.text().strip()
            if hasattr(self, "txt_output_folder")
            else ""
        ) or str(self.settings.get("output_folder") or OUTPUT_DIR)
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            QMessageBox.warning(self, "Output Folder", f"Could not open folder:\n{exc}")

    def _start_export(self):
        if not self._require_license():
            return
        if not self.current_video:
            QMessageBox.warning(self, "Warning", "Select a video.")
            return

        segs = self.segments.get(self.current_video, [])
        if not segs:
            QMessageBox.warning(self, "Warning", "Transcribe the video first.")
            return

        if not any(s.get("audio_path") for s in segs):
            QMessageBox.warning(
                self, "Warning",
                "Generate Voice first. This prevents exporting an empty dub."
            )
            return

        self.btn_export.setEnabled(False)
        self.btn_export_top.setEnabled(False)
        self.progress.setValue(0)
        settings = self._collect_settings()

        worker = ExportWorker(
            self.current_video,
            self.videos[self.current_video],
            segs,
            settings
        )
        worker.progress.connect(
            lambda p, t: (self.progress.setValue(p), self._status(t))
        )
        worker.finished.connect(self._export_done)
        worker.error.connect(self._worker_error)
        self.workers.append(worker)
        worker.start()

    def _start_export_all(self, clicked=False):
        # QPushButton.clicked emits a boolean. Export All does not use the
        # signal value, so explicitly ignore it and always process every
        # imported video.
        if not self._require_license():
            return
        if not self.videos:
            QMessageBox.warning(self, "Warning", "Import videos first.")
            return
        # Export All is deliberately tolerant: one failed/partial TTS video must
        # never prevent the other videos from exporting. Videos with at least one
        # generated segment are exported; videos with no audio are reported as skipped.
        self.batch_export_skipped = [
            name for name in self.videos
            if not any(
                s.get("audio_path") and Path(str(s.get("audio_path"))).exists()
                for s in self.segments.get(name, [])
            )
        ]
        self.batch_export_queue = [
            name for name in self.videos.keys() if name not in self.batch_export_skipped
        ]
        if not self.batch_export_queue:
            QMessageBox.warning(self, "Export All", "No video has usable generated voice audio. Generate Voice All first.")
            if self.full_process_active:
                self.full_process_active = False
                self.btn_full_process.setEnabled(True)
                self._status("FULL AUTO stopped: no generated voice audio was available to export.")
            if self.activity_full_auto_active:
                self.activity_full_auto_active = False
                self.btn_activity_full_auto.setEnabled(True)
                self.btn_activity.setEnabled(True)
                self.btn_activity_all.setEnabled(True)
                self._status("ACTIVITY FULL AUTO stopped: no generated voice audio was available to export.")
            return
        self.batch_export_active = True
        self.batch_export_total = len(self.batch_export_queue)
        self.batch_export_completed = 0
        self.batch_export_started_at = time.monotonic()
        self.btn_export_all.setEnabled(False)
        self.btn_export_top.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress.setValue(0)
        self._status(f"Starting batch export: {len(self.batch_export_queue)} video(s)...")
        self._start_next_batch_export()

    def _start_next_batch_export(self):
        if not self.batch_export_active:
            return
        if not self.batch_export_queue:
            self.batch_export_active = False
            self.btn_export_all.setEnabled(True)
            self.btn_export_top.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.progress.setValue(100)
            skipped = list(getattr(self, "batch_export_skipped", []))
            failed = list(getattr(self, "batch_export_failures", []))
            summary = f"Batch export complete: {len(self.videos) - len(skipped) - len(failed)} exported."
            if skipped:
                summary += " Skipped no-voice: " + ", ".join(skipped) + "."
            if failed:
                summary += " Failed: " + ", ".join(failed) + "."
            self._status(summary)
            if skipped or failed:
                QMessageBox.information(self, "Export All Complete", summary)
            if self.full_process_active:
                elapsed = time.time() - self.full_process_started_at
                self.full_process_active = False
                self.btn_full_process.setEnabled(True)
                self.progress.setValue(100)
                self._status(f"FULL AUTO COMPLETE: {len(self.videos) - len(skipped) - len(failed)}/{len(self.videos)} exported in {elapsed/60:.1f} min.")
                self._maybe_shutdown_pc()
            elif self.activity_full_auto_active:
                elapsed = time.time() - self.activity_full_auto_started_at
                self.activity_full_auto_active = False
                self.btn_activity_full_auto.setEnabled(True)
                self.btn_activity.setEnabled(True)
                self.btn_activity_all.setEnabled(True)
                self.progress.setValue(100)
                self._status(
                    f"ACTIVITY FULL AUTO COMPLETE: "
                    f"{len(self.videos) - len(skipped) - len(failed)}/{len(self.videos)} "
                    f"exported in {elapsed/60:.1f} min."
                )
                self._maybe_shutdown_pc()
            else:
                # Standalone Export All also respects the shutdown option.
                self._maybe_shutdown_pc()
            return
        name = self.batch_export_queue.pop(0)
        self.current_video = name
        self.cmb_video.blockSignals(True)
        self.cmb_video.setCurrentText(name)
        self.cmb_video.blockSignals(False)
        settings = self._collect_settings()
        worker = ExportWorker(name, self.videos[name], self.segments[name], settings)
        total = max(1, int(self.batch_export_total or 1))
        completed = int(self.batch_export_completed)
        worker.progress.connect(
            lambda p, t, n=name, done=completed, total_count=total:
            (
                self.progress.setValue(
                    min(99, max(0, int((done + max(0, min(100, p)) / 100.0) / total_count * 100)))
                ),
                self._status(
                    f"Export {done + 1}/{total_count} • {n}: {t}"
                ),
            )
        )
        worker.finished.connect(self._batch_export_done)
        worker.error.connect(lambda err, n=name: self._batch_export_error(n, err))
        self.workers.append(worker)
        worker.start()

    def _batch_export_done(self, name, video_path, srt_path):
        self.batch_export_completed += 1
        total = max(1, int(self.batch_export_total or 1))
        pct = min(99, int(self.batch_export_completed * 100 / total))
        self.progress.setValue(pct)
        self._status(
            f"Export {self.batch_export_completed}/{total} complete: {name}"
        )
        self._start_next_batch_export()

    def _batch_export_error(self, name, error):
        self.batch_export_failures.append(name)
        self.batch_export_completed += 1
        total = max(1, int(self.batch_export_total or 1))
        pct = min(99, int(self.batch_export_completed * 100 / total))
        self.progress.setValue(pct)
        self._status(
            f"Export {self.batch_export_completed}/{total} failed/skipped: {name} — {error}"
        )
        self._start_next_batch_export()

    def _export_done(self, name, video_path, srt_path):
        self.btn_export.setEnabled(True)
        self.btn_export_top.setEnabled(True)
        self.progress.setValue(100)
        self._status(f"Finished: {video_path}")

        msg = (
            f"Video:\n{video_path}\n\n"
            f"SRT:\n{srt_path}\n\n"
            "Open output folder?"
        )
        reply = QMessageBox.question(
            self, "Export Complete", msg,
            QMessageBox.StandardButton.Yes |
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                folder = str(Path(video_path).parent)
                if sys.platform.startswith("win"):
                    os.startfile(folder)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
            except Exception:
                pass

    # ---------------- Logo/color ----------------

    def _select_logo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Logo", "",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self.logo_path = path
            self.lbl_logo.setText(Path(path).name)
            self.chk_logo.setChecked(True)
            self._snapshot()
            self._refresh_preview_overlays()

    def _pick_sub_color(self):
        color = QColorDialog.getColor(QColor(self.sub_color), self)
        if color.isValid():
            self.sub_color = color.name()
            self.btn_sub_color.setStyleSheet(
                f"background: {self.sub_color}; color: black;"
            )
            self._snapshot()

    def _pick_title_color(self):
        color = QColorDialog.getColor(QColor(self.title_color), self)
        if color.isValid():
            self.title_color = color.name()
            self.btn_title_color.setStyleSheet(
                f"background: {self.title_color}; color: black;"
            )
            self._snapshot()

    # ---------------- API / project ----------------

    def _api_settings(self):
        dlg = ApiSettingsDialog(self, self.settings)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.settings.update(dlg.values())
            save_user_settings(self.settings)
            if hasattr(self, "cmb_tts_provider"):
                self.cmb_tts_provider.setCurrentText(self.settings.get("tts_provider", "Edge-TTS"))
            if hasattr(self, "txt_voxcpm_ref"):
                self.txt_voxcpm_ref.setText(self.settings.get("voxcpm_reference_wav", ""))
            if hasattr(self, "cmb_gpu"):
                self.cmb_gpu.setCurrentText(self.settings.get("gpu_mode", "Auto"))
            nkeys = len(self.settings.get("gemini_api_keys") or [])
            openai_ok = "configured" if self.settings.get("openai_api_key") else "not set"
            self._status(
                f"AI/API settings saved to user_settings.json. "
                f"Gemini keys: {nkeys}, OpenAI: {openai_ok}."
            )

    def _project_data(self):
        return {
            "version": APP_VERSION,
            "videos": self.videos,
            "segments": self.segments,
            "activities": self.activities,
            "current_video": self.current_video,
            "settings": self._collect_settings(),
            "logo_path": self.logo_path,
        }

    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", str(PROJECT_DIR / "project.json"),
            "Dubby Project (*.json)"
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self._project_data(), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            self.lbl_project.setText(Path(path).name)
            self._status(f"Project saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", str(PROJECT_DIR),
            "Dubby Project (*.json)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.videos = data.get("videos", {})
            self.segments = data.get("segments", {})
            self.activities = data.get("activities", {})
            self.current_video = data.get("current_video")
            self.settings.update(data.get("settings", {}))
            self.logo_path = data.get("logo_path") or None

            self.cmb_video.clear()
            self.cmb_video.addItems(list(self.videos.keys()))
            if self.current_video in self.videos:
                self.cmb_video.setCurrentText(self.current_video)

            self.cmb_language.setCurrentText(
                self.settings.get("target_lang_name", "Khmer")
            )
            self.cmb_gender.setCurrentText(
                self.settings.get("gender_default", "Female")
            )
            self._sync_ui_from_settings()
            self._load_table()
            self._load_activity_table()
            self.lbl_project.setText(Path(path).name)
            self._status("Project opened.")
        except Exception as e:
            QMessageBox.critical(self, "Open Error", str(e))

    def _sync_ui_from_settings(self):
        self.cmb_tts_provider.setCurrentText(self.settings.get("tts_provider", "Edge-TTS"))
        if hasattr(self, "txt_voxcpm_ref"):
            self.txt_voxcpm_ref.setText(self.settings.get("voxcpm_reference_wav", ""))
        self.sld_speed.setValue(int(float(self.settings.get("voice_speed", 1.0)) * 100))
        self.sld_edge_pitch.setValue(int(self.settings.get("edge_pitch_hz", 0)))
        self.sld_edge_volume.setValue(int(self.settings.get("edge_volume_pct", 0)))
        self.chk_bgm.setChecked(self.settings.get("keep_bgm", True))
        self.cmb_strength.setCurrentText(
            self.settings.get("vocal_strength", "strong")
        )
        self.sld_bgm.setValue(
            int(self.settings.get("bgm_volume", 0.25) * 100)
        )
        self.sld_duck.setValue(
            int(abs(self.settings.get("bgm_duck_db", -7)))
        )
        self.chk_subtitles.setChecked(
            self.settings.get("add_subtitles", False)
        )
        self.cmb_font.setCurrentText(
            self.settings.get("font_family", "Noto Sans Khmer")
        )
        self.sp_sub_size.setValue(
            max(10, min(160, int(self.settings.get("subtitle_font_size", 48))))
        )
        self.sp_bar_h.setValue(
            int(self.settings.get("subtitle_bar_height", 140))
        )
        self.sld_bar_op.setValue(
            int(self.settings.get("subtitle_bar_opacity", 0.65) * 100)
        )
        if hasattr(self, "sp_blur_radius"):
            self.sp_blur_radius.setValue(int(self.settings.get("subtitle_blur_radius", 18)))
        if hasattr(self, "sp_sub_x"):
            self.sp_sub_x.setValue(int(self.settings.get("subtitle_position_x", 50)))
        if hasattr(self, "sp_sub_y"):
            self.sp_sub_y.setValue(int(self.settings.get("subtitle_position_y", 88)))
        self.chk_title.setChecked(self.settings.get("add_title", False))
        self.txt_title.setText(self.settings.get("title_text", ""))
        self.sp_title_size.setValue(
            int(self.settings.get("title_size", 48))
        )
        self.cmb_title_pos.setCurrentText(
            self.settings.get("title_position", "top")
        )
        self.sp_title_x.setValue(int(self.settings.get("title_position_x", 50)))
        self.sp_title_y.setValue(int(self.settings.get("title_position_y", 10)))
        self.chk_logo.setChecked(self.settings.get("add_logo", False))
        self.sp_logo_w.setValue(
            int(self.settings.get("logo_width", 140))
        )
        self.sld_logo_op.setValue(
            int(self.settings.get("logo_opacity", 0.85) * 100)
        )
        self.cmb_logo_pos.setCurrentText(
            self.settings.get("logo_position", "top-right")
        )
        self.sp_logo_x.setValue(int(self.settings.get("logo_position_x", 90)))
        self.sp_logo_y.setValue(int(self.settings.get("logo_position_y", 10)))
        self._refresh_preview_overlays()
        if hasattr(self, "txt_output_folder"):
            folder = str(self.settings.get("output_folder") or OUTPUT_DIR)
            self.txt_output_folder.setText(folder)
        self.cmb_aspect.setCurrentText(
            self.settings.get("aspect_ratio", "Original")
        )
        self.cmb_preset.setCurrentText(
            self.settings.get("preset", "veryfast")
        )
        self.sp_crf.setValue(int(self.settings.get("crf", 20)))
        if hasattr(self, "chk_shutdown"):
            self.chk_shutdown.setChecked(bool(self.settings.get("shutdown_when_finished", False)))
        if self.logo_path:
            self.lbl_logo.setText(Path(self.logo_path).name)

    # ---------------- Undo/redo ----------------

    def _snapshot(self):
        if not self._recording_history:
            return
        state = json.dumps(
            {
                "segments": self.segments,
                "activities": self.activities,
                "settings": self._collect_settings(),
                "logo_path": self.logo_path,
            },
            ensure_ascii=False,
            sort_keys=True
        )
        if self.history and self.history[-1] == state:
            return
        self.history = self.history[:self.history_index + 1]
        self.history.append(state)
        self.history_index = len(self.history) - 1
        self.history = self.history[-30:]
        self.history_index = len(self.history) - 1

    def _restore_state(self, state):
        self._recording_history = False
        try:
            data = json.loads(state)
            self.segments = data.get("segments", {})
            self.activities = data.get("activities", {})
            self.settings.update(data.get("settings", {}))
            self.logo_path = data.get("logo_path")
            self._sync_ui_from_settings()
            self._load_table()
            self._load_activity_table()
        finally:
            self._recording_history = True

    def _undo(self):
        if self.history_index <= 0:
            return
        self.history_index -= 1
        self._restore_state(self.history[self.history_index])
        self._status("Undo")

    def _redo(self):
        if self.history_index >= len(self.history) - 1:
            return
        self.history_index += 1
        self._restore_state(self.history[self.history_index])
        self._status("Redo")

    # ---------------- Misc ----------------

    def _maybe_shutdown_pc(self, delay_sec: int = 120):
        """Schedule a PC shutdown after batch/full-auto work finishes.

        Only runs when the Export-tab checkbox is enabled. Uses a short delay
        so the user can still cancel (Windows: shutdown /a , Linux: shutdown -c).
        """
        if not getattr(self, "chk_shutdown", None) or not self.chk_shutdown.isChecked():
            return
        delay_sec = max(30, int(delay_sec or 120))
        minutes = max(1, delay_sec // 60)
        msg = (
            f"Dubby AI Studio finished. Shutting down in {minutes} minute(s). "
            "Cancel: Windows → shutdown /a   |   Linux → shutdown -c"
        )
        try:
            if sys.platform.startswith("win"):
                # /s = shutdown, /t = delay seconds, /c = comment shown to user
                subprocess.Popen(
                    ["shutdown", "/s", "/t", str(delay_sec), "/c", msg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                # Linux / macOS-style shutdown (requires privileges on some systems)
                subprocess.Popen(
                    ["shutdown", "-h", f"+{minutes}", msg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self._status(
                f"PC shutdown scheduled in {minutes} min. "
                "Cancel: Windows 'shutdown /a' or Linux 'shutdown -c'."
            )
            QMessageBox.information(
                self,
                "Shutdown scheduled",
                f"Processing finished.\n\n"
                f"This PC will shut down in about {minutes} minute(s).\n\n"
                "To cancel:\n"
                "  • Windows: open Command Prompt and run  shutdown /a\n"
                "  • Linux: run  shutdown -c",
            )
        except Exception as exc:
            self._status(f"Could not schedule shutdown: {exc}")
            QMessageBox.warning(
                self,
                "Shutdown failed",
                f"Could not schedule PC shutdown:\n{exc}\n\n"
                "You may need administrator privileges.",
            )

    def _status(self, text):
        # Keep status text in one place only: the QStatusBar.
        self.statusBar().showMessage(str(text))

    def _worker_error(self, msg):
        self.btn_transcribe.setEnabled(True)
        self.btn_transcribe_all.setEnabled(True)
        if hasattr(self, "btn_activity"):
            self.btn_activity.setEnabled(True)
        if hasattr(self, "btn_activity_all"):
            self.btn_activity_all.setEnabled(True)
        self.btn_translate.setEnabled(True)
        self.btn_tts.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_export_top.setEnabled(True)
        self.btn_export_all.setEnabled(True)
        self.batch_transcribe_active = False
        self.batch_export_active = False
        self.batch_translate_active = False
        self.batch_tts_active = False
        self.full_process_active = False
        self.activity_full_auto_active = False
        if hasattr(self, "btn_full_process"):
            self.btn_full_process.setEnabled(True)
        if hasattr(self, "btn_activity_full_auto"):
            self.btn_activity_full_auto.setEnabled(True)
        self.batch_transcribe_queue.clear()
        self.batch_activity_queue.clear()
        self.batch_export_queue.clear()
        self.batch_export_skipped.clear()
        self.batch_export_failures.clear()
        self.batch_export_total = 0
        self.batch_export_completed = 0
        self.batch_export_started_at = 0.0
        self.batch_translate_queue.clear()
        self.batch_tts_queue.clear()
        self._status("Error")
        QMessageBox.critical(self, "Processing Error", str(msg))

    def _clear_all(self, checked=False):
        """Reset the workspace without quitting the application.

        The optional *checked* argument absorbs the bool emitted by
        QPushButton.clicked so it is never mistaken for other state.
        Media player + combo signals are blocked carefully — an abrupt
        stop/clear of QMediaPlayer while QVideoSink is active can otherwise
        hard-crash the process (looks like the app “closed”).
        """
        try:
            reply = QMessageBox.question(
                self,
                "Clear Project",
                "Clear all imported videos, transcripts, activities, and editor state?\n\n"
                "API keys and AI settings are kept.\n"
                "This does not quit the application.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        except Exception:
            # If the dialog cannot show, still proceed with a safe clear.
            pass

        try:
            # ---- stop media safely (prevents native crash on some Qt builds) ----
            try:
                if hasattr(self, "player") and self.player is not None:
                    try:
                        self.player.stop()
                    except Exception:
                        pass
                    try:
                        # Detach source so the decoder releases the file handle.
                        self.player.setSource(QUrl())
                    except Exception:
                        pass
                if hasattr(self, "_tts_player") and self._tts_player is not None:
                    try:
                        self._tts_player.stop()
                        self._tts_player.setSource(QUrl())
                    except Exception:
                        pass
            except Exception:
                pass

            # ---- cancel batch pipelines (do not quit workers hard) ----
            self.batch_transcribe_active = False
            self.batch_export_active = False
            self.batch_translate_active = False
            self.batch_tts_active = False
            self.batch_activity_active = False
            self.full_process_active = False
            self.activity_full_auto_active = False
            try:
                self.batch_transcribe_queue.clear()
                self.batch_export_queue.clear()
                self.batch_export_skipped.clear()
                self.batch_export_failures.clear()
                self.batch_export_total = 0
                self.batch_export_completed = 0
                self.batch_export_started_at = 0.0
                self.batch_translate_queue.clear()
                self.batch_tts_queue.clear()
                self.batch_activity_queue.clear()
            except Exception:
                pass

            for btn_name in (
                "btn_full_process", "btn_activity_full_auto",
                "btn_transcribe", "btn_transcribe_all",
                "btn_translate", "btn_tts",
                "btn_export", "btn_export_top", "btn_export_all",
                "btn_activity", "btn_activity_all",
            ):
                btn = getattr(self, btn_name, None)
                if btn is not None:
                    try:
                        btn.setEnabled(True)
                    except Exception:
                        pass

            # ---- clear data ----
            self.videos.clear()
            self.segments.clear()
            self.activities.clear()
            self.current_video = None
            self.logo_path = None
            self._preview_playback_path = ""
            self._preview_proxy_retrying = False
            self._last_video_pixmap = QPixmap()
            self._preview_source_size = QSize(0, 0)

            # ---- UI (block combo signals so _on_video_changed is not fired) ----
            try:
                if hasattr(self, "cmb_video"):
                    self.cmb_video.blockSignals(True)
                    self.cmb_video.clear()
                    self.cmb_video.blockSignals(False)
            except Exception:
                pass

            try:
                if hasattr(self, "table"):
                    self.table.blockSignals(True)
                    self.table.setRowCount(0)
                    self.table.blockSignals(False)
            except Exception:
                pass

            try:
                if hasattr(self, "activity_table"):
                    self.activity_table.blockSignals(True)
                    self.activity_table.setRowCount(0)
                    self.activity_table.blockSignals(False)
                if hasattr(self, "txt_activity_narration"):
                    self.txt_activity_narration.clear()
                if hasattr(self, "lbl_activity_info"):
                    self.lbl_activity_info.setText("No activity analysis yet.")
            except Exception:
                pass

            try:
                if hasattr(self, "timeline_canvas"):
                    self.timeline_canvas.set_data([], 0.1, 0.0)
            except Exception:
                pass

            try:
                if hasattr(self, "video_widget"):
                    self.video_widget.clear()
                    self.video_widget.setText("No video")
                if hasattr(self, "lbl_preview_title"):
                    self.lbl_preview_title.hide()
                if hasattr(self, "lbl_preview_logo"):
                    self.lbl_preview_logo.hide()
                if hasattr(self, "lbl_preview_subtitle"):
                    self.lbl_preview_subtitle.hide()
            except Exception:
                pass

            try:
                if hasattr(self, "lbl_video_info"):
                    self.lbl_video_info.setText("No video selected")
                if hasattr(self, "lbl_project"):
                    self.lbl_project.setText("No project")
                if hasattr(self, "lbl_logo"):
                    self.lbl_logo.setText("No logo selected")
                if hasattr(self, "progress"):
                    self.progress.setValue(0)
                if hasattr(self, "btn_play"):
                    self.btn_play.setText("▶ Play")
                    self.btn_play.setEnabled(False)
                if hasattr(self, "lbl_time"):
                    self.lbl_time.setText("00:00 / 00:00")
                if hasattr(self, "sld_position"):
                    self.sld_position.blockSignals(True)
                    self.sld_position.setValue(0)
                    self.sld_position.blockSignals(False)
            except Exception:
                pass

            # Keep a small undo history baseline after clear.
            try:
                self.history = []
                self.history_index = -1
                self._snapshot()
            except Exception:
                pass

            self._status("Project cleared — app is still running. Import a video to continue.")
        except Exception as exc:
            # Never let Clear Project take down the whole process.
            try:
                self._status(f"Clear Project finished with a warning: {exc}")
            except Exception:
                pass
            try:
                QMessageBox.warning(
                    self,
                    "Clear Project",
                    f"Project was cleared, but something minor failed:\n{exc}\n\n"
                    "The app is still open.",
                )
            except Exception:
                pass

    def closeEvent(self, event):
        try:
            # Always persist API keys / AI settings on exit.
            try:
                save_user_settings(self.settings)
            except Exception:
                pass
            self.player.stop()
            for worker in self.workers:
                if worker.isRunning():
                    worker.quit()
                    worker.wait(1500)
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")

    window = DubbyAIStudio()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()