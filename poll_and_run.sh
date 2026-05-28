#!/bin/bash
# Poll ~/may24/ until stable, then run video_stage_cutter
set -e

DIR=~/may24
OUT=~/may24_clips
VENV=~/beep/.venv

PREV_SIZE=""
STABLE=0

echo "Polling $DIR until upload complete..."

while true; do
    SIZE=$(du -sb "$DIR" 2>/dev/null | cut -f1)
    COUNT=$(ls "$DIR"/*.MP4 "$DIR"/*.mp4 2>/dev/null | wc -l)
    echo "$(date +%H:%M:%S) files=$COUNT size=$(numfmt --to=iec $SIZE 2>/dev/null || echo $SIZE)"

    if [ "$SIZE" = "$PREV_SIZE" ] && [ -n "$SIZE" ] && [ "$SIZE" != "0" ]; then
        STABLE=$((STABLE + 1))
        echo "  stable ${STABLE}/6"
        if [ $STABLE -ge 6 ]; then
            echo ""
            echo "===== UPLOAD COMPLETE ====="
            ls -lhS "$DIR"/*.MP4 "$DIR"/*.mp4 2>/dev/null
            break
        fi
    else
        STABLE=0
    fi

    PREV_SIZE="$SIZE"
    sleep 30
done

echo ""
echo "===== RUNNING VIDEO STAGE CUTTER ====="
source "$VENV/bin/activate"
python -m video_stage_cutter "$DIR" "$OUT" --keep-wav -v --overwrite 2>&1 | tee ~/may24_run.log

echo ""
echo "===== DONE ====="
echo "Clips: $OUT"
echo "Log: ~/may24_run.log"
echo "Manifest: $OUT/manifest.csv"
