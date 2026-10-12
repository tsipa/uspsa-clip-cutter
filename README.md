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

## Batched Transcription

```powershell
python -m video_stage_cutter "E:\cam\oct11" "E:\cam\oct11\crop" --device cuda --compress --stabilize --gpu-encode --batched --batch-size 2
```

`--batched` (alias `--batched-transcription`) opts into faster-whisper's `BatchedInferencePipeline`. `--batch-size` defaults to **2** and controls the number of audio chunks decoded together, not the number of video files or model copies. With the default `--workers 1`, the same model and pipeline are reused across files. The normal sequential transcription mode remains the default; `--batch-size` alone does not enable batching.

The full audio is covered by contiguous chunks of at most 30 seconds, with **VAD disabled**, preserving quiet range-officer commands instead of filtering them out before recognition. Segment and word timestamps remain relative to the original recording. The same English language, USPSA prompt, beam size 5 and word timestamps are used. Batched decoding does not carry the previous chunk's recognized text as context, so transcripts may differ, especially for words crossing chunk boundaries. Review detection results on representative footage before relying on a new recognition mode.

Larger batches use more GPU memory and are not guaranteed to be faster. Start at 2 on an 8 GB GTX 1070; reduce to `--batch-size 1` if necessary. `--workers` remains independent, and multiple workers each load their own model; use one worker when trying batching on a single GPU. `--reuse-transcripts` still bypasses recognition for cached files, irrespective of transcription mode. No encoding, stabilization or compute-type settings are changed by `--batched`.

## Fast Cut Example

```
python -m video_stage_cutter run "C:\videos" "C:\clips" --fast-cut
```

Uses stream-copy instead of re-encoding. Much faster, but the clip may start a few frames early (on the nearest keyframe before the detected start). Use `--accurate-cut` (the default) if precise boundaries matter.

## Full Example with All Options

```
python -m video_stage_cutter run "C:\videos" "C:\clips" --model small --device cpu --compute-type int8 --accurate-cut --end-padding 3.0 --keep-wav --dry-run -v
```

## Compression and Stabilization

```powershell
python -m video_stage_cutter "C:\videos" "C:\clips" --compress
python -m video_stage_cutter "C:\videos" "C:\clips" --compress --stabilize
python -m video_stage_cutter "C:\videos" "C:\clips" --device cuda --compress --stabilize --gpu-encode
```

`--compress` enables this ffmpeg output profile:

- **Up to 1920x1080 (1080p), 30 FPS**, preserving the display aspect ratio. Larger sources are downscaled; smaller sources keep their resolution. Dimensions are rounded down to even values for H.265. No padding to a full 1080p canvas is added.
- **H.265 / HEVC** (`libx265`), **preset `slow`**, **CRF 26**, 8-bit `yuv420p`.
- **AAC audio**, 96 kbit/s, 48 kHz, stereo, for MP4 compatibility. Sources without audio produce silent video without an added audio stream.
- **MP4** with `hvc1` and `+faststart` for progressive playback.

Compression uses constant quality, with **no file size or bitrate target**. Around 20–30 MB per minute is a guide (roughly 2.6–3.9 Mbit/s of video plus AAC), not a guarantee; movement, noise, detail and resolution affect the result. CRF 26 is retained as a starting point for the quality/size tradeoff, rather than imposing a bitrate cap on fast motion. Ten-minute clips keep the same quality settings and can be much larger. Tune quality/size with `--compress-crf 28` for smaller files or `--compress-crf 23` for better quality. The `slow` preset takes more encoding time.

`--gpu-encode` selects NVIDIA `hevc_nvenc` and implies `--compress`. It retains the same resolution, 30 FPS, AAC 96k, MP4 and stabilization settings. The GPU profile uses preset `p7`, tuning `hq`, VBR with target quality **CQ 26** and no bitrate target (`-b:v 0`), full-resolution multipass, lookahead 32 and spatial AQ. HEVC B-frames and temporal AQ are omitted for compatibility with GTX 1070/Pascal. Adjust GPU quality with `--gpu-cq 28` for smaller files or a lower value for better quality; CQ is separate from CPU CRF, so their numbers do not guarantee identical quality or size. `--compress-crf` controls the CPU profile only. Encoding uses NVENC; decoding, scaling and vid.stab remain on CPU. `--device cuda` independently selects GPU speech recognition. The requested encoder and a short real NVENC encode are checked before transcription; a missing GPU/driver produces an actionable error. `--dry-run` skips this check and all encoding.

On Windows, if Whisper needs the NVIDIA DLL directories on PATH, run this in the same PowerShell session (using this project's virtual environment):

```powershell
$env:PATH += ";C:\Users\Ts\beep\uspsa-clip-cutter\.venv\Lib\site-packages\nvidia\cublas\bin"
$env:PATH += ";C:\Users\Ts\beep\uspsa-clip-cutter\.venv\Lib\site-packages\nvidia\cudnn\bin"
& "C:\Users\Ts\beep\uspsa-clip-cutter\.venv\Scripts\python.exe" -m video_stage_cutter "E:\cam\oct11" "E:\cam\oct11\crop" --device cuda --compress --stabilize --gpu-encode
```

`--stabilize` separately enables two-pass image stabilization (`vidstabdetect` then `vidstabtransform`): shakiness 5, accuracy 15, fixed 5% zoom (`optzoom=0:zoom=5`), and bicubic interpolation. Both passes process the same original frame sequence and resolution; with `--compress`, scaling happens before stabilization and conversion to 30 FPS happens only after `vidstabtransform`. The default smoothing radius is approximately one second on each side: 30 source frames at 30 FPS, 60 at 60 FPS, or the nearest integer for other rates. `--stabilize-smoothing` remains expressed in frames at 30 FPS, so `15` means approximately half a second per side. A zero radius stays zero; the scaled radius is capped at vid.stab's limit of 1000 source frames, with a warning. Source FPS uses the first video stream's average frame rate, falling back to its nominal rate. For variable-rate footage the temporal window is approximate; for concatenation it uses the rate of the file containing the clip start, assuming compatible camera chapters at the same rate. A fixed modest zoom prevents isolated large movements from determining the crop for the whole clip. Strong movements can still expose black borders (`crop=black`); the crop does not grow to hide them. Stabilization at the source rate adds processing time. Without `--compress`, stabilization retains the existing H.264/AAC encoding profile and source resolution/frame rate.

Both options force re-encoding even when `--fast-cut` is supplied. They work for single-file clips and stages spanning multiple source files. For concatenated clips, explicit source durations allow input seeking to the relevant file/keyframe; input and output duration limits bound processing to the selected interval plus decoder preroll. Audio and video share the seek origin. As with the existing concat demuxer, sources must have compatible streams, codecs and time bases. Use a full ffmpeg build with `libx265`, AAC, and (for stabilization) `libvidstab`; required capabilities are checked before transcription. With `--dry-run`, encoding and stabilization are skipped.

---

## Windows Sleep Prevention

On Windows, the CLI automatically prevents **idle system sleep** throughout detection, transcription, cutting and manifest writing. It restores the prior thread execution state on completion, errors or Ctrl+C, without changing the Windows power plan. The display can still turn off. Manual sleep, closing a laptop lid and shutting down remain under Windows/user control. If Windows rejects the request, the program logs a warning and continues. Other platforms are unaffected.

## Restoring an Interrupted Run

Transcripts and progress checkpoints are stored in `<output>/debug` (or `--debug-dir`). Completed files use `<stem>_transcript.json`. During recognition, `<stem>_transcript.partial.json` is saved after every yielded segment, including its words and the current decoder-window offset. Writes are flushed and replaced atomically so an interruption cannot truncate the previous checkpoint. The partial file is removed only after the completed transcript is saved successfully.

```powershell
python -m video_stage_cutter "E:\cam\oct11" "E:\cam\oct11\crop" --device cuda --compress --stabilize --gpu-encode --batched --batch-size 2 --restore
```

`--restore` reuses completed transcripts (including those from earlier versions) and resumes partial files. It replays the last decoder window, retaining the completed prefix without duplicate words or compressed timestamps. In batched mode this usually reprocesses at most the last 30-second chunk; standard mode can also restart from a decoder-window boundary. Recognition happens again for an unfinished window because its remaining segments may not have been yielded before interruption. Partial checkpoints are validated against the source path, size, modification time and Whisper model; invalid/stale partials are logged and re-transcribed. Old completed transcripts without checkpoint metadata retain the existing cache-reuse behavior.

Checkpoints are written automatically in new runs; `--restore` is only needed when restarting. Transcription begun in an older running process cannot gain checkpoints retroactively. `--reuse-transcripts` remains available for completed files; a partial checkpoint takes precedence over an older completed transcript. Existing video outputs are skipped unless `--overwrite` is supplied. If interrupted FFmpeg encoding left an incomplete output file, use `--overwrite` to regenerate it. Restoration covers transcription, not FFmpeg's internal encoder state.

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
| `--batched` / `--batched-transcription` | off | Recognize audio chunks in batches using `BatchedInferencePipeline`, without VAD. |
| `--batch-size` | `2` | Audio chunks per inference batch (minimum 1); used with `--batched`. Larger batches require more memory. |
| `--accurate-cut` / `--fast-cut` | `--accurate-cut` | Re-encode for frame-accurate boundaries (slower) or stream-copy (fast, may start on a prior keyframe). |
| `--compress` | off | Encode up to 1920x1080/30 FPS H.265 slow CRF 26 + AAC 96k without upscaling. Constant quality, no size target; overrides `--fast-cut`. |
| `--compress-crf` | `26` | Quality for `--compress`, 0–51. Lower = better quality/larger files; higher = smaller files. |
| `--gpu-encode` | off | Use NVIDIA NVENC HEVC p7/hq. Implies `--compress`; independent of Whisper `--device`. |
| `--gpu-cq` | `26` | GPU target quality, 1–51. Lower = better quality/larger files; higher = smaller files. Separate from CPU CRF. |
| `--stabilize` | off | Apply two-pass vid.stab image stabilization; overrides `--fast-cut`. |
| `--stabilize-smoothing` | `30` | Smoothing radius in 30-FPS-equivalent frames, 0–1000. Scaled to source FPS: default 30 becomes 60 source frames at 60 FPS (~1 second per side). |
| `--keep-wav` | off | Keep the extracted WAV files in the debug directory instead of deleting them after processing. |
| `--debug-dir` | `<output>/debug` | Custom directory for transcript and detection JSON files. |
| `--start-padding` | `10.0` | Seconds to include before the detected start (beep or "are you ready"). |
| `--end-padding` | `10.0` | Seconds to include after the detected end command. |
| `--min-clip-length` | `5.0` | Reject clips shorter than this many seconds. |
| `--max-clip-length` | `600.0` | Reject clips longer than this many seconds. |
| `--overwrite` | off | Overwrite existing output files. Without this, existing outputs are skipped. |
| `--dry-run` | off | Run detection and write manifest/debug files, but do not cut any video. |
| `--restore` | off | Reuse completed transcripts and resume partial recognition from checkpoints in the output debug directory. |
| `--phrase-threshold` | `70.0` | Fuzzy matching threshold for phrase detection (0-100). Lower = more lenient, higher = stricter. |
| `--beep-search-before` | `0.25` | Seconds before "stand by" end to start searching for the timer beep. |
| `--beep-search-after` | `10.0` | Seconds after "stand by" end to stop searching for the timer beep. Clamped to first end command if found earlier. |
| `-v` / `--verbose` | off | Enable debug-level logging for detailed output. |

---

## How Detection Works

1. **Audio extraction.** ffmpeg extracts a mono 16 kHz WAV from each video.
2. **Speech recognition.** All files are transcribed with faster-whisper (word-level timestamps). A domain-specific prompt biases the model toward RO commands.
3. **Global keyword detection.** All transcripts are combined on a single timeline. Individual keywords are found via fuzzy matching (rapidfuzz): "make ready", "are you ready", "stand by", "finished", "unload", "show clear", "hammer down", "holster", "range is clear".
4. **Sequence matching.** Keywords are grouped into ordered sequences matching the expected RO protocol. Start: make ready -> are you ready -> stand by. End: finished -> unload -> show clear -> hammer down -> holster -> range is clear. Keywords must appear in chronological order; out-of-order keywords are ignored.
5. **Beep detection.** After finding "stand by", a spectrogram analysis searches for a tonal beep (2500-5000 Hz). Candidates are scored by a composite of energy, tonality, broadband ratio, and duration. Broadband impulses (gunshots) are rejected.
6. **Confidence scoring.** Each stage gets a base confidence from its best anchor score, then boosted by corroborating evidence: +0.3 for beep, +0.2 for having both start and end, +0.1 for standby, +0.05 per gunshot near boundaries (max +0.2). Stages with effective confidence below 0.4 are dropped. This filters out isolated words like "holster" (0.10) or "ready" (0.10) that appear in casual conversation.
7. **Fallback clips.** If only start is found: 3-minute clip after start. If only end is found: 3 minutes before end. Fallbacks are trimmed to avoid overlapping confirmed stages. If trimming makes a clip shorter than `--min-clip-length`, it is skipped.
8. **Gunshot detection.** Amplitude spikes are detected as gunshots. Gunshots near stage boundaries boost confidence, confirming that shooting actually occurred.

---

## Troubleshooting

### "ffmpeg not found" or "ffprobe not found"

ffmpeg is not on your system PATH. On Windows, install it with `winget install Gyan.FFmpeg` and **restart your terminal**. Verify with `ffmpeg -version`.

### First run is very slow

The first time you run the tool, it downloads the Whisper model (the default `large-v3` model is about 3 GB). This is a one-time download. Subsequent runs use the cached model. Use `--model small` (~500 MB) for faster downloads and lower memory usage.

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
- **Increase `--end-padding`** if your clips cut off too early after "hammer down and holster". The default is 10 seconds.
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
