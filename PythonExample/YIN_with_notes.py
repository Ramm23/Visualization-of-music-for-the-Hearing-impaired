
import os
import re
import tempfile
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly, fftconvolve

# Optional: force GUI backend; if this fails, PNG saving still works.
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt


NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F",
                    "F#", "G", "G#", "A", "A#", "B"]
NOTE_TO_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}
SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],      # natural minor
    "chromatic": list(range(12)),
}


def _has_ffmpeg() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"],
                           capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


def load_audio_mono(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    ext = p.suffix.lower()

    # WAV can be read directly
    if ext == ".wav":
        x, sr = sf.read(str(p), always_2d=True)

    else:
        if not _has_ffmpeg():
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg and restart your terminal.")

        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # IMPORTANT on Windows: release handle so ffmpeg can write

        try:
            cmd = [
                "ffmpeg", "-y",
                "-hide_banner",
                "-nostdin",
                "-loglevel", "error",
                "-i", str(p),
                "-vn",
                "-ac", "1",
            ]

            # resample inside ffmpeg (recommended)
            if target_sr is not None:
                cmd += ["-ar", str(target_sr)]

            cmd += [tmp_wav]

            print("Resolved input path:", str(p), flush=True)
            print("Running:", " ".join(cmd), flush=True)

            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    "ffmpeg failed to decode/convert the file.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Return code: {r.returncode}\n"
                    f"stderr:\n{r.stderr.strip()}"
                )

            x, sr = sf.read(tmp_wav, always_2d=True)

        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

    # Convert to mono float32
    x = x.mean(axis=1).astype(np.float32)

    # If WAV and still want resample here (usually unnecessary if target_sr given)
    if target_sr is not None and sr != target_sr and ext == ".wav":
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        x = resample_poly(x, up, down).astype(np.float32)
        sr = target_sr

    return x, sr


# ----------------------------
# YIN core (steps 2–5)
# ----------------------------

def difference_function_fast(frame: np.ndarray, W: int, tau_max: int) -> np.ndarray:
    """
    FAST Step 2:
    d(tau) = E0 + E(tau) - 2 * sum_{j=0..W-1} x[j]*x[j+tau]
    The dot-products are computed for all taus at once using FFT convolution:
    corr = conv(b, reverse(a)) and corr[W-1 + tau] gives the needed dot.
    """
    # a = x[0..W-1], b = x[0..W+tau_max-1]
    a = frame[:W].astype(np.float64, copy=False)
    b = frame[:W + tau_max].astype(np.float64, copy=False)

    # cross-correlation via convolution b * reversed(a)
    corr_full = fftconvolve(b, a[::-1], mode="full")
    # acf[tau] for tau=0..tau_max
    acf = corr_full[W - 1: W - 1 + (tau_max + 1)]

    # energy terms
    E0 = np.dot(a, a)
    b2 = b * b
    cs = np.concatenate(([0.0], np.cumsum(b2)))
    taus = np.arange(0, tau_max + 1)
    E_tau = cs[taus + W] - cs[taus]  # sum of squares of b[tau:tau+W]

    d = E0 + E_tau - 2.0 * acf
    d[0] = 0.0
    return d.astype(np.float32)


def cmndf(d: np.ndarray) -> np.ndarray:
    dprime = np.empty_like(d, dtype=np.float32)
    dprime[0] = 1.0

    cumsum = np.cumsum(d[1:], dtype=np.float64)
    taus = np.arange(1, len(d), dtype=np.float64)
    denom = cumsum / taus

    out = np.ones_like(d[1:], dtype=np.float32)
    mask = denom > 1e-20
    out[mask] = (d[1:][mask] / denom[mask]).astype(np.float32)
    dprime[1:] = out
    return dprime


def absolute_threshold(dprime: np.ndarray, tau_min: int, tau_max: int, thresh: float) -> int:
    tau_max = min(tau_max, len(dprime) - 1)

    tau = None
    for t in range(tau_min, tau_max + 1):
        if dprime[t] < thresh:
            tau = t
            break

    if tau is None:
        return int(np.argmin(dprime[tau_min:tau_max + 1]) + tau_min)

    while tau + 1 <= tau_max and dprime[tau + 1] < dprime[tau]:
        tau += 1
    return tau


def parabolic_interpolation(y: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-20:
        return float(i)
    delta = 0.5 * (y0 - y2) / denom
    return float(i) + delta


def yin_one_frame(
    x: np.ndarray,
    sr: int,
    start: int,
    W: int,
    tau_min: int,
    tau_max: int,
    thresh: float
) -> tuple[float, float, float, float] | None:
    need = start + W + tau_max
    if start < 0 or need > len(x):
        return None

    frame = x[start:need]

    # FAST step 2
    d = difference_function_fast(frame, W, tau_max)

    # steps 3–5
    dprime = cmndf(d)
    tau_int = absolute_threshold(dprime, tau_min, tau_max, thresh)
    tau_hat = parabolic_interpolation(dprime, tau_int)

    aperiodicity = float(dprime[tau_int])
    confidence = float(np.clip(1.0 - aperiodicity, 0.0, 1.0))
    f0 = float(sr / tau_hat) if tau_hat > 0 else 0.0
    return f0, tau_hat, confidence, aperiodicity


# ----------------------------
# Step 6 (best local estimate) + tracking
# ----------------------------
def yin_track(
    x: np.ndarray,
    sr: int,
    fmin: float = 40.0,
    fmax: float = 2000.0,
    frame_size: int = 2048,
    hop_size: int = 1024,
    thresh: float = 0.1,
    conf_thresh: float = 0.6,
    # Step 6 controls (KEEP OFF for long files unless you really need it)
    local_radius: int = 0,
    local_step: int = 8,
    refine_radius: int = 3,
    # progress
    progress: bool = True,
    progress_every: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau_min = max(2, int(sr / fmax))
    tau_max = int(sr / fmin)

    times, f0s, confs = [], [], []

    max_start = len(x) - (frame_size + tau_max)
    if max_start <= 0:
        return np.array([]), np.array([]), np.array([])

    total_frames = (max_start // hop_size) + 1

    for idx, start in enumerate(range(0, max_start, hop_size), start=1):
        if progress and (idx % progress_every == 0 or idx == 1 or idx == total_frames):
            print(
                f"Tracking... frame {idx}/{total_frames} ({(idx/total_frames)*100:.1f}%)", flush=True)

        best = None  # (aperiodicity, f0, tau, conf, start_used)

        if local_radius > 0:
            # Step 6: pick best local estimate around start
            for u in range(start - local_radius, start + local_radius + 1, local_step):
                out = yin_one_frame(x, sr, u, frame_size,
                                    tau_min, tau_max, thresh)
                if out is None:
                    continue
                f0, tau, conf, ap = out
                if best is None or ap < best[0]:
                    best = (ap, f0, tau, conf, u)

            if best is None:
                continue

            ap, f0, tau, conf, u = best
            center = int(round(tau))
            tau_min2 = max(tau_min, center - refine_radius)
            tau_max2 = min(tau_max, center + refine_radius)

            out2 = yin_one_frame(x, sr, u, frame_size,
                                 tau_min2, tau_max2, thresh)
            if out2 is not None:
                f0, tau, conf, ap = out2
                best = (ap, f0, tau, conf, u)

            ap, f0, tau, conf, u = best
            start_used = u

        else:
            out = yin_one_frame(x, sr, start, frame_size,
                                tau_min, tau_max, thresh)
            if out is None:
                continue
            f0, tau, conf, ap = out
            start_used = start

        times.append(start_used / sr)
        confs.append(conf)
        f0s.append(f0 if conf >= conf_thresh else np.nan)

    return np.array(times), np.array(f0s), np.array(confs)


# ----------------------------
# Pitch -> note / key helpers
# ----------------------------
def hz_to_midi(freq_hz: float) -> float:
    if not np.isfinite(freq_hz) or freq_hz <= 0:
        return np.nan
    return 69.0 + 12.0 * np.log2(freq_hz / 440.0)


def midi_to_note_name(midi_note: int) -> str:
    pitch_class = midi_note % 12
    octave = (midi_note // 12) - 1
    return f"{NOTE_NAMES_SHARP[pitch_class]}{octave}"


def freq_to_note_info(freq_hz: float) -> dict:
    """
    Map a detected frequency to the nearest equal-tempered note.
    Returns the note name, octave, MIDI note number, and cent deviation.
    """
    if not np.isfinite(freq_hz) or freq_hz <= 0:
        return {
            "note": None,
            "octave": None,
            "note_with_octave": None,
            "midi": None,
            "cents_error": None,
        }

    midi_float = hz_to_midi(freq_hz)
    midi_round = int(np.round(midi_float))
    cents_error = float((midi_float - midi_round) * 100.0)
    note_name = NOTE_NAMES_SHARP[midi_round % 12]
    octave = (midi_round // 12) - 1

    return {
        "note": note_name,
        "octave": octave,
        "note_with_octave": f"{note_name}{octave}",
        "midi": midi_round,
        "cents_error": cents_error,
    }


def parse_key(key: str) -> tuple[int, str]:
    """
    Examples accepted:
        'C major'
        'A minor'
        'F# major'
        'Bb minor'
        'D'          -> defaults to major
        'chromatic'  -> all 12 notes treated as in-key
    """
    key = key.strip()
    if not key:
        raise ValueError("Key string cannot be empty.")

    key_upper = key.upper().strip()
    if key_upper == "CHROMATIC":
        return 0, "chromatic"

    m = re.match(r"^\s*([A-Ga-g])([#bB]?)(?:\s+(major|minor))?\s*$", key)
    if not m:
        raise ValueError(
            "Invalid key format. Use examples like 'C major', 'A minor', 'F# major', or 'Bb minor'."
        )

    tonic = (m.group(1) + m.group(2)).upper().replace("B", "b").upper()
    # repair flats after the forced uppercase conversion
    tonic = tonic.replace("BB", "Bb").replace("DB", "Db").replace(
        "EB", "Eb").replace("GB", "Gb").replace("AB", "Ab")
    mode = (m.group(3) or "major").lower()

    tonic_pc = NOTE_TO_PC[tonic.upper()]
    return tonic_pc, mode


def key_pitch_classes(key: str) -> set[int]:
    tonic_pc, mode = parse_key(key)
    intervals = SCALE_INTERVALS[mode]
    return {(tonic_pc + interval) % 12 for interval in intervals}


def is_note_in_key(note_name: str | None, key: str) -> bool | None:
    if note_name is None:
        return None
    pc = NOTE_TO_PC[note_name.upper()]
    return pc in key_pitch_classes(key)


def annotate_pitches(
    times: np.ndarray,
    f0: np.ndarray,
    conf: np.ndarray,
    key: str,
) -> list[dict]:
    """
    For every frame, convert YIN's detected frequency into:
    - note name
    - octave
    - nearest MIDI note
    - cents deviation
    - whether the note belongs to the user-provided key
    """
    annotations: list[dict] = []
    in_key_pcs = key_pitch_classes(key)

    for t, freq, c in zip(times, f0, conf):
        info = freq_to_note_info(freq)
        note_name = info["note"]
        midi_note = info["midi"]

        if midi_note is None:
            in_key = None
        else:
            in_key = (midi_note % 12) in in_key_pcs

        annotations.append({
            "time_s": float(t),
            "frequency_hz": None if not np.isfinite(freq) else float(freq),
            "confidence": float(c),
            "note": note_name,
            "octave": info["octave"],
            "note_with_octave": info["note_with_octave"],
            "midi": midi_note,
            "cents_error": info["cents_error"],
            "in_key": in_key,
        })

    return annotations


def compress_note_events(annotations: list[dict], min_duration_s: float = 0.05) -> list[dict]:
    """
    Merge consecutive frames that point to the same detected note.
    This gives you cleaner 'current note being played' events.
    """
    if not annotations:
        return []

    events: list[dict] = []
    current = None

    for row in annotations:
        label = row["note_with_octave"]

        if current is None:
            current = {
                "start_s": row["time_s"],
                "end_s": row["time_s"],
                "note": row["note"],
                "octave": row["octave"],
                "note_with_octave": row["note_with_octave"],
                "in_key": row["in_key"],
                "mean_frequency_hz": [] if row["frequency_hz"] is None else [row["frequency_hz"]],
                "mean_confidence": [row["confidence"]],
            }
            continue

        if label == current["note_with_octave"]:
            current["end_s"] = row["time_s"]
            if row["frequency_hz"] is not None:
                current["mean_frequency_hz"].append(row["frequency_hz"])
            current["mean_confidence"].append(row["confidence"])
        else:
            duration = current["end_s"] - current["start_s"]
            if duration >= min_duration_s:
                current["mean_frequency_hz"] = (
                    float(np.mean(current["mean_frequency_hz"]))
                    if current["mean_frequency_hz"] else None
                )
                current["mean_confidence"] = float(
                    np.mean(current["mean_confidence"]))
                events.append(current)

            current = {
                "start_s": row["time_s"],
                "end_s": row["time_s"],
                "note": row["note"],
                "octave": row["octave"],
                "note_with_octave": row["note_with_octave"],
                "in_key": row["in_key"],
                "mean_frequency_hz": [] if row["frequency_hz"] is None else [row["frequency_hz"]],
                "mean_confidence": [row["confidence"]],
            }

    if current is not None:
        duration = current["end_s"] - current["start_s"]
        if duration >= min_duration_s:
            current["mean_frequency_hz"] = (
                float(np.mean(current["mean_frequency_hz"]))
                if current["mean_frequency_hz"] else None
            )
            current["mean_confidence"] = float(
                np.mean(current["mean_confidence"]))
            events.append(current)

    return events


def detect_notes_in_audio(
    audio_path: str,
    key: str,
    target_sr: int = 22050,
    fmin: float = 40.0,
    fmax: float = 2000.0,
    frame_size: int = 2048,
    hop_size: int = 1024,
    thresh: float = 0.1,
    conf_thresh: float = 0.6,
    local_radius: int = 0,
    local_step: int = 8,
    refine_radius: int = 3,
    progress: bool = True,
    progress_every: int = 25,
) -> tuple[list[dict], list[dict]]:
    """
    End-to-end helper:
    1) Load audio
    2) Run YIN
    3) Convert frequencies to notes + octaves
    4) Mark whether each note is in the given key
    5) Return both frame-wise annotations and compressed note events
    """
    x, sr = load_audio_mono(audio_path, target_sr=target_sr)

    times, f0, conf = yin_track(
        x, sr,
        fmin=fmin,
        fmax=fmax,
        frame_size=frame_size,
        hop_size=hop_size,
        thresh=thresh,
        conf_thresh=conf_thresh,
        local_radius=local_radius,
        local_step=local_step,
        refine_radius=refine_radius,
        progress=progress,
        progress_every=progress_every,
    )

    frame_annotations = annotate_pitches(times, f0, conf, key=key)
    note_events = compress_note_events(frame_annotations)
    return frame_annotations, note_events


def plot_pitch(times: np.ndarray, f0: np.ndarray, conf: np.ndarray, fmax: float = 2000.0, out_png: str = "yin_pitch.png"):
    finite = np.isfinite(f0)

    plt.figure()
    plt.plot(times[finite], f0[finite], linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (Hz)")
    plt.ylim(0, fmax * 1.1)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(out_png, dpi=200)
    print(f"[OK] Saved pitch plot to: {os.path.abspath(out_png)}", flush=True)

    # try showing a window too
    try:
        plt.show()
    except Exception as e:
        print(
            "[INFO] plt.show() failed (no GUI backend). PNG saved instead.", flush=True)
        print("Reason:", e, flush=True)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    print("SCRIPT:", Path(__file__).resolve(), flush=True)
    print("CWD:", os.getcwd(), flush=True)

    path = r"PythonExample\Happy_Birthday.mp3"
    user_key = "C major"   # <- provide the key here

    # BIG SPEED WIN: downsample for analysis (still fine for pitch visualization)
    x, sr = load_audio_mono(path, target_sr=22050)
    print(
        f"Loaded: {len(x)} samples, sr={sr} Hz, duration={len(x)/sr:.2f} s", flush=True)

    times, f0, conf = yin_track(
        x, sr,
        fmin=40.0,
        fmax=2000.0,
        frame_size=2048,
        hop_size=1024,      # fewer frames -> faster, still good for plotting
        thresh=0.1,
        conf_thresh=0.6,
        # Step 6 OFF for speed (turn on later if you want)
        local_radius=0,
        local_step=8,
        refine_radius=3,
        progress=True,
        progress_every=25
    )

    voiced = np.isfinite(f0)
    print(f"Frames: {len(times)}", flush=True)
    print(
        f"Voiced frames: {voiced.sum()} ({(voiced.sum()/max(1, len(times)))*100:.1f}%)", flush=True)
    if len(conf) > 0:
        print(
            f"Confidence: min={conf.min():.3f}, median={np.median(conf):.3f}, max={conf.max():.3f}", flush=True)

    if len(times) == 0:
        raise RuntimeError(
            "No frames were processed. Try smaller frame_size/hop_size or check audio decode.")

    # Frame-wise note detection
    annotations = annotate_pitches(times, f0, conf, key=user_key)
    events = compress_note_events(annotations, min_duration_s=0.05)

    print(f"\nDetected note events for key = {user_key}")
    for event in events[:20]:
        print(
            f"{event['start_s']:.2f}s -> {event['end_s']:.2f}s | "
            f"{event['note_with_octave']} | in_key={event['in_key']} | "
            f"mean_f0={event['mean_frequency_hz']}"
        )

    plot_pitch(times, f0, conf, fmax=2000.0, out_png="yin_pitch.png")
