"""
app/engine/pipeline.py
======================
Two-step, single-decode frame extraction pipeline.

Step 1 — ffprobe PTS map  (run once per video, ~5 s overhead)
    ffprobe scans all video frames and records their actual
    presentation timestamps (best_effort_timestamp_time).
    This handles VFR videos, broken DTS, and any container quirks.
    The result is a dict  {sequential_frame_idx: timestamp_seconds}
    used as a timestamp fallback.

Step 2 — FFmpeg select-filter pipe  (single decode pass)
    FFmpeg decodes the video ONCE and applies:
      select='gt(scene,SCORE)'  →  only scene-changed frames flow through
      showinfo                  →  per-selected-frame PTS written to stderr
      scale=224:224             →  resize inside FFmpeg (GPU on many platforms)

    A background thread drains stderr and pushes parsed timestamps into a
    thread-safe queue. The main loop reads raw 224×224 RGB frames from
    stdout and pairs each one with its PTS from the queue (or the ffprobe
    fallback map if the queue is empty / parse failed).

    Result: 6-8× faster than the previous double-decode approach with the
    same (or better) semantic coverage.

Inference consumer — unchanged batch SigLIP inference over PIL images.
"""

import json
import logging
import queue
import re
import subprocess
import threading

try:
    import av as _av  # PyAV — FFmpeg C-library bindings (demux-only PTS map)
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False

import numpy as np
from PIL import Image

from app.engine.models import SigLipEngine
from app.core.config import settings

log = logging.getLogger(__name__)

# Output resolution fed to the model — resize in FFmpeg to minimise pipe I/O.
_MODEL_W = 224
_MODEL_H = 224
_BYTES_PER_FRAME = _MODEL_W * _MODEL_H * 3   # RGB24


# ---------------------------------------------------------------------------
# Step 1 helpers
# ---------------------------------------------------------------------------

def _get_video_info(video_path: str) -> dict:
    """
    ffprobe: FPS, resolution, duration.
    Returns {"fps": float, "width": int, "height": int, "duration": float}.
    Falls back to safe defaults on any error.
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                fps_str = stream.get("r_frame_rate", "30/1")
                num, den = fps_str.split("/")
                fps = float(num) / max(float(den), 1e-9)
                return {
                    "fps": fps,
                    "width": int(stream.get("width", 1920)),
                    "height": int(stream.get("height", 1080)),
                    "duration": float(stream.get("duration", 0.0) or 0.0),
                }
    except Exception as exc:
        log.warning("ffprobe stream info failed (%s); using defaults.", exc)
    return {"fps": 30.0, "width": 1920, "height": 1080, "duration": 0.0}


def _get_frame_pts_map(video_path: str) -> dict:
    """
    Step 1: Build a {sequential_frame_idx: timestamp_seconds} PTS map.

    Primary strategy — PyAV demux-only (no subprocess, no decode):
        Opens the container and iterates compressed video packets directly
        in Python memory via FFmpeg's C libav* bindings.  For a 2.5-hour
        H.264/H.265 MKV this completes in ~3-8 s and uses constant RAM.

        B-frame safety: packets are collected, then sorted by PTS before
        sequential indexing so that presentation order matches decode order.

        Broken-PTS safety: if a packet's PTS is AV_NOPTS_VALUE (None in
        PyAV), we fall back to its DTS.  If DTS is also None we skip that
        packet (the last-resort fps-estimate tier covers it).

    Fallback — empty dict:
        Returned on any error; callers use frame_idx / fps instead.
    """
    if not _AV_AVAILABLE:
        log.warning(
            "PyAV not installed (pip install av).  "
            "Timestamps will use frame_idx/fps."
        )
        return {}

    log.info("Step 1: Building PTS map via PyAV demux for '%s'…", video_path)
    try:
        container = _av.open(video_path)
        video_stream = next(
            (s for s in container.streams if s.type == "video"), None
        )
        if video_stream is None:
            log.warning("PyAV: no video stream found in '%s'.", video_path)
            container.close()
            return {}

        time_base = float(video_stream.time_base)  # e.g. 1/90000

        # ── Collect raw packet timestamps (demux only — no decode) ──────
        raw_pts: list = []  # list of float seconds, one per video packet
        for packet in container.demux(video_stream):
            # Skip flush packets (pts=None, dts=None, size=0)
            if packet.size == 0:
                continue

            # Prefer PTS (presentation timestamp); fall back to DTS
            ts_ticks = packet.pts
            if ts_ticks is None:
                ts_ticks = packet.dts
            if ts_ticks is None:
                # Truly unknown — skip; fps-estimate tier covers this frame
                continue

            raw_pts.append(ts_ticks * time_base)

        container.close()

        if not raw_pts:
            log.warning("PyAV: zero valid packet timestamps in '%s'.", video_path)
            return {}

        # ── Sort by PTS to restore presentation order (B-frame safe) ────
        raw_pts.sort()

        pts_map: dict = {i: ts for i, ts in enumerate(raw_pts)}
        log.info(
            "PTS map: %d packets indexed via PyAV demux.", len(pts_map)
        )
        return pts_map

    except Exception as exc:
        log.warning(
            "PyAV PTS map failed (%s). Timestamps will use frame_idx/fps.", exc
        )
        return {}


# ---------------------------------------------------------------------------
# Step 2 helper — stderr drain for showinfo timestamps
# ---------------------------------------------------------------------------

def _drain_showinfo_stderr(
    proc_stderr,
    ts_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """
    Background thread: parse 'showinfo' filter lines from FFmpeg stderr and
    push the extracted pts_time values into ts_queue.

    showinfo line example (normal):
      [Parsed_showinfo_1 @ 0x…] n:   0 pts:   0 pts_time:0.000 pos:…
    showinfo line example (broken PTS):
      [Parsed_showinfo_1 @ 0x…] n:   0 pts:   0 pts_time:N/A   pos:…

    IMPORTANT: when pts_time is N/A we push None (not silence) so the main
    producer loop gets an immediate signal and can use the PyAV fallback map
    without waiting for the 2-second queue timeout.
    """
    _pts_re = re.compile(r'pts_time:(\S+)')
    try:
        for raw_line in proc_stderr:
            if stop_event.is_set():
                break
            line = raw_line.decode("utf-8", errors="ignore")
            if "pts_time:" not in line:
                continue
            m = _pts_re.search(line)
            if m:
                raw_val = m.group(1)
                try:
                    ts_queue.put(float(raw_val))   # normal case: valid float
                except ValueError:
                    # pts_time:N/A or any non-numeric value → sentinel
                    # Push None so the main loop falls through to PyAV map
                    # immediately rather than stalling for the 2-second timeout.
                    ts_queue.put(None)
                    log.debug("showinfo pts_time='%s' (not a number); sent None sentinel.", raw_val)
    except Exception as exc:
        log.debug("stderr drain thread exited: %s", exc)


# ---------------------------------------------------------------------------
# VideoFrameProducer  (single FFmpeg pass, scene-filter + showinfo)
# ---------------------------------------------------------------------------

class VideoFrameProducer(threading.Thread):
    """
    Produces (image, timestamp, frame_idx) tuples from a video file using a
    single FFmpeg decode pass.

    FFmpeg filter chain applied:
        select='gt(scene,SCORE)' → showinfo → scale=224:224

    Timestamps are sourced from:
        Primary   — showinfo PTS parsed from FFmpeg stderr (VFR-correct)
        Fallback  — ffprobe PTS map built in Step 1
        Last resort — frame position / nominal FPS

    Sentinel ``None`` is placed on the queue when the producer is done.
    """

    def __init__(
        self,
        video_path: str,
        frame_queue: queue.Queue,
        batch_size: int | None = None,
        scene_score: float | None = None,
        fallback_fps: float | None = None,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.batch_size = batch_size or settings.INFERENCE_BATCH_SIZE
        self.scene_score = scene_score if scene_score is not None else settings.FFMPEG_SCENE_SCORE
        self.fallback_fps = fallback_fps if fallback_fps is not None else settings.FALLBACK_FPS

    # ------------------------------------------------------------------
    def run(self) -> None:
        info = _get_video_info(self.video_path)
        fps: float = info["fps"]

        # ── Step 1: Build PTS fallback map via ffprobe ─────────────────
        pts_map = _get_frame_pts_map(self.video_path)

        # ── Step 2: Single FFmpeg pass ─────────────────────────────────
        # Filter chain:
        #   select  → only frames with scene-change score > threshold
        #   showinfo → writes per-frame PTS to stderr (parsed in background)
        #   scale   → resize to model input size INSIDE FFmpeg (fast, low pipe I/O)
        vf = (
            f"select='gt(scene,{self.scene_score})',"
            f"showinfo,"
            f"scale={_MODEL_W}:{_MODEL_H}"
        )   
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", self.video_path,
            "-vf", vf,
            "-vsync", "0",           # VFR output — only emit selected frames
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-an",                   # No audio
            "pipe:1",
        ]

        log.info(
            "Step 2: FFmpeg single-pass | scene_score=%.2f | video='%s'",
            self.scene_score, self.video_path,
        )

        # Thread-safe queue for timestamps parsed from stderr
        ts_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()

        try:
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=_BYTES_PER_FRAME * 8,
            )
        except FileNotFoundError:
            log.error("FFmpeg not found. Make sure ffmpeg is on PATH.")
            self.frame_queue.put(None)
            return

        # Launch stderr drain thread BEFORE reading stdout
        stderr_thread = threading.Thread(
            target=_drain_showinfo_stderr,
            args=(proc.stderr, ts_queue, stop_event),
            daemon=True,
        )
        stderr_thread.start()

        selected_idx: int = 0
        frames_enqueued: int = 0

        try:
            while True:
                raw = proc.stdout.read(_BYTES_PER_FRAME)
                if len(raw) < _BYTES_PER_FRAME:
                    break  # EOF

                # ── Resolve timestamp ──────────────────────────────────
                # Priority 1: showinfo PTS from stderr (VFR-correct, exact)
                #   timeout reduced to 2.0 s (was 5.0 s) — long enough for
                #   any real pipeline lag, short enough to not block on bugs.
                #   ts_raw may be:
                #     float  → valid PTS in seconds          → use directly
                #     None   → showinfo reported pts_time:N/A → fall through
                #     (empty)→ stderr thread fell behind      → fall through
                _use_fallback = False
                try:
                    ts_raw = ts_queue.get(timeout=2.0)
                    if ts_raw is None:
                        # showinfo reported N/A PTS — use PyAV map / fps
                        log.debug(
                            "Frame %d: showinfo pts_time N/A; using fallback.",
                            selected_idx,
                        )
                        _use_fallback = True
                    else:
                        timestamp = ts_raw
                except queue.Empty:
                    # stderr thread fell behind (shouldn't happen normally)
                    log.debug(
                        "Frame %d: showinfo queue empty after 2 s; using fallback.",
                        selected_idx,
                    )
                    _use_fallback = True

                if _use_fallback:
                    # Priority 2: PyAV demux PTS map
                    if selected_idx in pts_map:
                        timestamp = pts_map[selected_idx]
                        log.debug(
                            "Frame %d: used PyAV pts_map fallback (%.3fs)",
                            selected_idx, timestamp,
                        )
                    else:
                        # Priority 3: nominal FPS estimate (last resort)
                        timestamp = selected_idx / fps if fps > 0 else 0.0
                        log.debug(
                            "Frame %d: using fps estimate (%.3fs)", selected_idx, timestamp,
                        )

                # ── Build PIL image (zero-copy numpy frombuffer) ───────
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((_MODEL_H, _MODEL_W, 3))
                pil_image = Image.fromarray(arr)

                self.frame_queue.put(
                    {
                        "image": pil_image,
                        "timestamp": timestamp,
                        "frame_idx": selected_idx,
                    }
                )
                frames_enqueued += 1
                selected_idx += 1

        except Exception as exc:
            log.error("Error reading FFmpeg pipe at frame %d: %s", selected_idx, exc)
        finally:
            stop_event.set()
            proc.stdout.close()
            proc.wait()
            stderr_thread.join(timeout=10)

        # ── Fallback: if FFmpeg scene filter selected 0 frames, ────────
        # fall back to uniform 1-FPS sampling so the video is never skipped.
        if frames_enqueued == 0:
            log.warning(
                "Scene filter returned 0 frames for '%s'. "
                "Falling back to uniform %.1f-FPS sampling.",
                self.video_path, self.fallback_fps,
            )
            self._uniform_fallback(fps, pts_map)
            self.frame_queue.put(None)
            return

        log.info(
            "Producer done | selected=%d frames | video='%s'",
            frames_enqueued, self.video_path,
        )
        self.frame_queue.put(None)  # Sentinel — signals consumer to finish

    # ------------------------------------------------------------------
    def _uniform_fallback(self, fps: float, pts_map: dict) -> None:
        """
        Uniform 1-FPS sampling fallback used when the scene filter returns 0
        frames (e.g., very short clip, static screen recording).
        Re-runs FFmpeg with a simple fps filter instead of the scene filter.
        """
        skip = max(1, int(fps / self.fallback_fps))
        vf_fallback = f"select='not(mod(n\\,{skip}))',scale={_MODEL_W}:{_MODEL_H}"
        ffmpeg_cmd = [
            "ffmpeg", "-i", self.video_path,
            "-vf", vf_fallback,
            "-vsync", "0",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo", "-an",
            "pipe:1",
        ]
        log.info("Fallback extraction: every %d frames (%.1f FPS)", skip, self.fallback_fps)
        try:
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=_BYTES_PER_FRAME * 8,
            )
            idx = 0
            while True:
                raw = proc.stdout.read(_BYTES_PER_FRAME)
                if len(raw) < _BYTES_PER_FRAME:
                    break
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((_MODEL_H, _MODEL_W, 3))
                pil_image = Image.fromarray(arr)
                timestamp = pts_map.get(idx * skip, (idx * skip) / fps if fps > 0 else 0.0)
                self.frame_queue.put(
                    {"image": pil_image, "timestamp": timestamp, "frame_idx": idx}
                )
                idx += 1
            proc.stdout.close()
            proc.wait()
            log.info("Fallback extracted %d frames.", idx)
        except Exception as exc:
            log.error("Uniform fallback failed: %s", exc)


# ---------------------------------------------------------------------------
# InferenceConsumer  (batch SigLIP inference — now returns frame_idx too)
# ---------------------------------------------------------------------------

class InferenceConsumer:
    """
    Consumes frames from the shared queue, accumulates them into batches,
    and runs SigLIP image-feature extraction.

    Returns a list of dicts:
        [{"vector": list[float], "timestamp": float, "frame_idx": int}, …]
    """

    def __init__(self, frame_queue: queue.Queue, batch_size: int | None = None):
        self.frame_queue = frame_queue
        self.batch_size = batch_size or settings.INFERENCE_BATCH_SIZE
        self.siglip = SigLipEngine()
        self.session, self.processor, self.device = self.siglip.get_components()

    def process_video(self) -> list:
        results: list = []
        batch: list = []

        while True:
            item = self.frame_queue.get()

            if item is None:          # Sentinel → flush remaining batch
                if batch:
                    results.extend(self._run_batch(batch))
                break

            batch.append(item)

            if len(batch) >= self.batch_size:
                results.extend(self._run_batch(batch))
                batch = []

        log.info("Consumer done | total embeddings=%d", len(results))
        return results

    # ------------------------------------------------------------------
    def _run_batch(self, batch: list) -> list:
        images = [item["image"] for item in batch]
        timestamps = [item["timestamp"] for item in batch]
        frame_indices = [item["frame_idx"] for item in batch]

        inputs = self.processor(
            images=images,
            return_tensors="np",
            padding="max_length",
        )
        pixel_values = inputs["pixel_values"].astype("float32")

        embeddings_np = self.siglip.get_image_features(pixel_values)

        return [
            {
                "vector": emb.tolist(),
                "timestamp": timestamps[i],
                "frame_idx": frame_indices[i],
            }
            for i, emb in enumerate(embeddings_np)
        ]