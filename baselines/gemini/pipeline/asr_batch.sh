#!/bin/bash
# ASR transcription for moshi and gemini inference results (dev_release_flac).
#
# For every <instance>/<speaker> folder, transcribes:
#   - output.wav                  (model response)
#   - speaker_{1,2}_audio.flac    (input speaker audio)
# output.flac is skipped (duplicate of output.wav, would collide on output.json).
#
# Transcripts mirror the audio hierarchy, e.g.:
#   audio_output/gemini/gemini-3.1/20/speaker_1/output.wav
#     -> asr_results/gemini/20/speaker_1/output.json
#   audio_output/gemini/gemini-3.1/20/speaker_1/speaker_1_audio.flac
#     -> asr_results/gemini/20/speaker_1/speaker_1_audio.json

set -e

AUDIO_ROOT=/bathrooms/kcire/sesame/results/audio_output
ASR_ROOT=/bathrooms/kcire/sesame/results/asr_results

FILENAMES="output.wav speaker_1_audio.flac speaker_2_audio.flac"

# --local_attention + --chunk_sec: files here are 10-15 min; full attention
# OOMs past ~11 min, and even local attention with timestamps needs multi-GiB
# decode buffers on a 20GB GPU, so transcribe in 5-min segments

# Moshi results
python asr_generic.py \
    --input_dir "$AUDIO_ROOT/moshi" \
    --output_dir "$ASR_ROOT/moshi" \
    --filenames $FILENAMES \
    --local_attention \
    --chunk_sec 300 \
    --skip_existing

# Gemini results
python asr_generic.py \
    --input_dir "$AUDIO_ROOT/gemini/gemini-3.1" \
    --output_dir "$ASR_ROOT/gemini" \
    --filenames $FILENAMES \
    --local_attention \
    --chunk_sec 300 \
    --skip_existing

# ── Previous runs (kept for reference) ────────────────────────────────────────
# python asr_generic.py --input_dir /bathrooms/kcire/BD_models/covo_audio/mpbench_covo_out --output_dir /livingrooms/kcire/bd/asr_results/gemini_elevenlabs_greet_v1prompt/covo_audio
# python asr_generic.py --input_dir /bathrooms/kcire/BD_models/glm4voice/mpbench_glm4voice_out --output_dir /livingrooms/kcire/bd/asr_results/gemini_elevenlabs_greet_v1prompt/glm4voice
# python asr_generic.py --input_dir /livingrooms/kcire/bd/inference_result_pro_filtered_eleven_labs_v2prompt/xai_output --output_dir /livingrooms/kcire/bd/asr_results/gemini_elevenlabs_greet_v2prompt/grok_new
# python asr_generic.py --input_dir /livingrooms/kcire/bd/inference_result_elevenlabs_v1prompt/xai_output --output_dir /livingrooms/kcire/bd/asr_results/gemini_elevenlabs_greet_v1prompt/grok_new
# python asr_generic.py --input_dir /livingrooms/kcire/bd/inference_result_elevenlabs_v1prompt_slow_start/xai_output --output_dir /livingrooms/kcire/bd/asr_results/gemini_elevenlabs_greet_v1prompt_slow_start/grok_new
