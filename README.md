# USPSA Clip Cutter

Automatically extract stage clips from USPSA / shooting-match action camera videos.

The tool scans a folder of videos, uses local speech recognition (faster-whisper) to find the range officer's commands ("Are you ready?", "Stand by", and the timer beep), and the end command ("If clear, hammer down and holster"). It then cuts the segment between the beep and the end command into a separate video file. Everything runs locally on your machine -- no API keys, no uploads, no cloud services.

---

## Requirements

- **Python 3.11 or newer**
- **ffmpeg and ffprobe** available on your system PATH
- Enough disk space for temporary WAV files and debug output (roughly 2x the size of one video per file being processed)
- **GPU is optional.** CPU mode works out of the box and is the default.

---

## Windows Installation

Follow these steps exactly. You do not need prior Python packaging experience.

### 1. Install ffmpeg

Open PowerShell and run:

```
winget install Gyan.FFmpeg
```

**Close PowerShell and open a new PowerShell window.** This is required so the new PATH entries take effect.

Verify the installation:

```
ffmpeg -version
ffprobe -version
```

Both commands should print version information. If you see "not recognized", close and reopen PowerShell again.

### 2. Create a virtual environment

Navigate to the folder where you cloned or downloaded this project, then run:

```
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Your prompt should now show `(.venv)` at the beginning.

### 3. Upgrade pip

```
python -m pip install --upgrade pip
```

### 4. Install the project

```
pip install -e .
```

This installs the project and all its dependencies (faster-whisper, numpy, scipy, rapidfuzz, tqdm, typer).

---

## Linux / macOS Installation

Install ffmpeg using your system package manager:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg
```

Then create a virtual environment and install:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

## Quick Start

The basic command is:

```
python -m video_stage_cutter run "C:\path\to\videos" "C:\path\to\clips"
```

The first argument is the folder containing your source videos (.mp4, .mov, .m4v). The second argument is the folder where cut clips will be saved.

---

## Recommended First Run

Start with a dry run to verify detection without actually cutting any video:

```
python -m video_stage_cutter run "C:\videos" "C:\clips" --dry-run
```

This will:
- Extract audio from each video
- Run speech recognition
- Detect start and end phrases
- Detect the timer beep
- Write `manifest.csv` and debug JSON files
- **Not** cut any video

Open `manifest.csv` in Excel or a text editor. Check the `start_offset`, `end_offset`, and `status` columns. If the detections look correct, run the actual cut.

---

## CPU Example

```
python -m video_stage_cutter run "C:\videos" "C:\clips" --model small --device cpu --compute-type int8
```

This uses the `small` Whisper model with int8 quantization. Faster but less accurate than the default `large-v3`.

## NVIDIA GPU Example

```
python -m video_stage_cutter run "C:\videos" "C:\clips" --model medium --device cuda --compute-type float16
```

This uses the `medium` model on your NVIDIA GPU with float16 precision. It is significantly faster than CPU mode but requires CUDA to be set up on your system. CUDA setup depends on your GPU driver and CUDA toolkit version -- if you get CUDA errors, fall back to CPU mode.

## Fast Cut Example

```
python -m video_stage_cutter run "C:\videos" "C:\clips" --fast-cut
```

Uses stream-copy instead of re-encoding. Much faster, but the clip may start a few frames early (on the nearest keyframe before the detected start). Use `--accurate-cut` (the default) if precise boundaries matter.

## Full Example with All Options

```
python -m video_stage_cutter run "C:\videos" "C:\clips" --model small --device cpu --compute-type int8 --accurate-cut --end-padding 3.0 --keep-wav --dry-run -v
```

---

## Expected Workflow

1. **Put videos in one folder.** Copy your action camera files (.mp4, .mov, .m4v) into a single input folder.
2. **Run a dry run.** Use `--dry-run` to generate detection results without cutting.
3. **Inspect manifest.csv.** Check that `start_offset` and `end_offset` look reasonable for each video. Check the `status` column for failures.
4. **Run the actual cut.** Remove the `--dry-run` flag to cut clips.
5. **Review clips.** Watch the output clips to verify boundaries.
6. **Tune if needed.** If detections are off, try `--model medium`, increase `--end-padding`, or check the debug JSON files for details.

---

## Output Files

### Clips

Cut clips are saved in the output folder with names like:

```
2026-05-27_14-32-18__GX010123__stage_clip.mp4
```

The format is: `<creation_time>__<original_filename>__stage_clip.mp4`. The creation time comes from the video's metadata (MP4/MOV `creation_time` tag). If that is unavailable, the file's modification time is used.

### manifest.csv

A CSV file in the output folder summarizing every processed video:

| Column | Description |
|---|---|
| source_file | Full path to the source video |
| creation_time | ISO 8601 timestamp from metadata or filesystem |
| duration | Duration of the cut clip in seconds |
| start_offset | Start time of the clip in the source video (seconds) |
| end_offset | End time of the clip in the source video (seconds) |
| start_reason | How the start was determined (e.g., `beep_after_standby`, `standby_end_fallback`) |
| end_reason | How the end was determined (e.g., `matched:hammer down and holster`) |
| confidence | Average detection confidence (0.00 to 1.00) |
| output_file | Path to the output clip |
| status | `ok`, `dry_run`, `skipped`, or `failed` |
| error_message | Error details if status is `failed` |

### Debug JSON Files

In the debug directory (`<output_dir>/debug/` by default, or the path you pass with `--debug-dir`):

- **`<filename>_transcript.json`** -- Full word-level transcript from Whisper. Contains each segment's start/end time, text, and individual word timings with confidence scores.
- **`<filename>_detection.json`** -- Detection results including all start/end phrase candidates with scores, beep candidates with timestamps and energy levels, the chosen start/end times, and the reasons for each choice.

---

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--model` | `large-v3` | Whisper model name. Options: `tiny`, `base`, `small`, `medium`, `large-v3`. Larger models are more accurate but slower. |
| `--device` | `cpu` | Device for Whisper inference. Use `cpu` or `cuda` (NVIDIA GPU). |
| `--compute-type` | `int8` | Numeric precision. `int8` for CPU, `float16` for GPU, `float32` for maximum accuracy. |
| `--accurate-cut` / `--fast-cut` | `--accurate-cut` | Re-encode for frame-accurate boundaries (slower) or stream-copy (fast, may start on a prior keyframe). |
| `--keep-wav` | off | Keep the extracted WAV files in the debug directory instead of deleting them after processing. |
| `--debug-dir` | `<output>/debug` | Custom directory for transcript and detection JSON files. |
| `--start-padding` | `10.0` | Seconds to include before the detected start (beep or "are you ready"). |
| `--end-padding` | `10.0` | Seconds to include after the detected end command. |
| `--min-clip-length` | `5.0` | Reject clips shorter than this many seconds. |
| `--max-clip-length` | `600.0` | Reject clips longer than this many seconds. |
| `--overwrite` | off | Overwrite existing output files. Without this, existing outputs are skipped. |
| `--dry-run` | off | Run detection and write manifest/debug files, but do not cut any video. |
| `--phrase-threshold` | `70.0` | Fuzzy matching threshold for phrase detection (0-100). Lower = more lenient, higher = stricter. |
| `--beep-search-before` | `0.25` | Seconds before "stand by" end to start searching for the timer beep. |
| `--beep-search-after` | `10.0` | Seconds after "stand by" end to stop searching for the timer beep. Clamped to first end command if found earlier. |
| `--workers` | `1` | Number of parallel workers for anchor detection. >1 spawns separate processes. GPU mode forces 1. |
| `-v` / `--verbose` | off | Enable debug-level logging for detailed output. |

---

## How Detection Works

1. **Audio extraction.** ffmpeg extracts a mono 16 kHz WAV from each video.
2. **Speech recognition.** faster-whisper (a local Whisper implementation) transcribes the audio with word-level timestamps.
3. **Phrase matching.** Each word sequence in the transcript is compared against known RO commands using fuzzy string matching (rapidfuzz). This handles variations in pronunciation and transcription errors.
4. **Beep detection.** After finding "stand by", the tool analyzes a short audio window using a spectrogram to find high-frequency energy spikes (the timer beep, typically 2500-5000 Hz).
5. **Clip boundaries.** The clip starts at the beep (or at the end of "stand by" if no beep is found). The clip ends after "hammer down and holster" (or the longest matching end phrase), plus configurable padding.
6. **Fallback clips.** If the end command is not found, the tool cuts a 3-minute clip after the detected start and marks it as incomplete. If a start/beep is not found but an end command is, the tool cuts the 3 minutes before the end. These fallback clips are automatically trimmed so they do not overlap any confirmed stage (where both start and end were found). If trimming makes a fallback clip shorter than `--min-clip-length`, it is skipped.
7. **Gunshot detection.** Loud transient amplitude spikes are detected as gunshots. These are used to validate that shooting actually occurred between the start and end commands.

---

## Troubleshooting

### "ffmpeg not found" or "ffprobe not found"

ffmpeg is not on your system PATH. On Windows, install it with `winget install Gyan.FFmpeg` and **restart your terminal**. Verify with `ffmpeg -version`.

### First run is very slow

The first time you run the tool, it downloads the Whisper model (the `small` model is about 500 MB). This is a one-time download. Subsequent runs use the cached model.

### Speech recognition misses commands

Range noise (gunfire, brass hitting the ground, wind) can interfere with speech recognition. Try:
- Use `--model medium` instead of `small` for better accuracy.
- Check the transcript JSON in the debug folder to see what Whisper actually heard.
- Lower the internal fuzzy matching threshold if needed (code change in `phrase_detect.py`).

### Beep detected at the wrong time

Check the `_detection.json` debug file. It lists all beep candidates with timestamps and energy levels. The tool picks the strongest beep after "stand by". If your timer has an unusual beep frequency, you can adjust the frequency band in `beep_detect.py` (default is 2500-5000 Hz).

### Clips are cut a little early or late

- Use `--accurate-cut` (the default) for frame-accurate boundaries. `--fast-cut` uses stream-copy which snaps to the nearest keyframe.
- Increase `--end-padding` if the clip cuts off before the shooter finishes holstering.
- Use `--start-padding` to include a moment before the beep.

### GPU / CUDA errors

If you get CUDA-related errors, fall back to CPU mode:

```
python -m video_stage_cutter run "C:\videos" "C:\clips" --device cpu --compute-type int8
```

CUDA requires a compatible NVIDIA GPU, the correct driver version, and the CUDA toolkit. CPU mode works everywhere.

### Paths with spaces

Always wrap paths in quotes:

```
python -m video_stage_cutter run "C:\My Videos\Match Day" "C:\My Videos\Clips"
```

### No clips produced

Open `manifest.csv` and check the `status` and `error_message` columns. Common reasons:
- `No start phrase detected` -- Whisper did not hear "are you ready" or "stand by". Try a larger model.
- `No end command detected` -- Whisper did not hear "hammer down". Check the transcript JSON.
- `Clip would be X.Xs, below minimum 5.0s` -- The detected segment is too short. This usually means a detection error.

Check the debug JSON files for detailed detection information including all candidates and scores.

---

## Tuning Advice

- **Use `--model medium` if `small` misses speech.** The medium model is significantly more accurate but about 3x slower on CPU.
- **Use `--accurate-cut` if clip boundaries matter.** This is the default and re-encodes the video for frame-perfect cuts.
- **Increase `--end-padding`** if your clips cut off too early after "hammer down and holster". The default is 2 seconds.
- **Keep debug files when tuning.** The transcript and detection JSON files show exactly what the tool detected and why it chose specific boundaries. Use `--keep-wav` to also preserve the audio for manual inspection.
- **Check confidence scores in manifest.csv.** Low confidence scores indicate uncertain detections.
- **Adjust beep search window** with `--beep-search-before` and `--beep-search-after` if the beep is detected at the wrong time. The default window is -0.25s to +10.0s around the end of "stand by".
- **Lower `--phrase-threshold`** (default 70) if speech recognition hears commands but fuzzy matching rejects them due to noise-garbled transcription.

---

## Privacy and Cost

- **Everything runs locally.** No data leaves your machine.
- **No OpenAI API key is required.** faster-whisper is a free, open-source local implementation of Whisper.
- **No video or audio is uploaded anywhere.** All processing happens on your CPU (or local GPU).
- **The Whisper model is downloaded once** from Hugging Face and cached locally.
- **CPU mode is free.** No special hardware required.
