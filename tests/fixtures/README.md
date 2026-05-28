# Test Fixtures

## Synthetic fixtures (committed to git)

Small JSON transcript files and generated WAV audio used by unit tests.
These do not require ffmpeg, faster-whisper, or real video files.

- `transcripts/*.json` -- synthetic Whisper-style transcript segments with
  word-level timestamps. Used to test phrase detection, stage assembly,
  and fallback/overlap logic without running speech recognition.

## Real video fixtures (NOT committed)

Large real action-camera videos should never be committed to this repo.
Place them in:

```
local_fixtures/real/
```

This directory is in `.gitignore`.

Tests that require real video files should be marked with the `real_video`
pytest marker:

```python
@pytest.mark.real_video
def test_with_real_gopro_footage():
    ...
```

Run only real-video tests:

```
pytest -m real_video
```

Skip real-video tests (default CI behavior):

```
pytest -m "not real_video"
```
