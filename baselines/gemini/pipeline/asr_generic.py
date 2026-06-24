#!/usr/bin/env python3
"""
Generic ASR Batch Transcription Script

This script transcribes all audio files found under an input directory
and saves JSON transcripts to an output directory while preserving
the original folder hierarchy.

Example:
    Input:
        /path/to/audio/
        ├── folder1/
        │   ├── audio1.wav
        │   └── subfolder/
        │       └── audio2.mp3
        └── folder2/
            └── audio3.flac

    Output (with --output_dir /path/to/transcripts):
        /path/to/transcripts/
        ├── folder1/
        │   ├── audio1.json
        │   └── subfolder/
        │       └── audio2.json
        └── folder2/
            └── audio3.json

Usage:
    python asr_generic.py --input_dir /path/to/audio --output_dir /path/to/transcripts
    python asr_generic.py --input_dir /path/to/audio --output_dir /path/to/transcripts --skip_existing
    python asr_generic.py --input_dir /path/to/audio --output_dir /path/to/transcripts --extensions .wav .mp3
"""

import os
import gc
import json
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Set

import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
from tqdm import tqdm


# Supported audio extensions
DEFAULT_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def _transcribe_segment(waveform, sr: int, asr_model) -> List[Dict]:
    """Transcribe one audio segment, return NeMo word timestamps (seconds)."""
    # Create temporary file for transcription (NeMo expects .wav)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, waveform, sr)
        try:
            asr_outputs = asr_model.transcribe([tmp.name], timestamps=True)
        finally:
            os.unlink(tmp.name)

    word_timestamps = asr_outputs[0].timestamp["word"]
    # Free GPU memory held by the hypothesis/decoding buffers; without this,
    # usage accumulates across files until CUDA OOM
    del asr_outputs
    gc.collect()
    torch.cuda.empty_cache()
    return word_timestamps


def transcribe_audio_file(audio_path: str, asr_model, chunk_sec: float = 0.0) -> Dict:
    """
    Transcribe a single audio file and return the result dict.

    Args:
        audio_path: Path to the audio file
        asr_model: Loaded NeMo ASR model
        chunk_sec: If > 0, split audio into segments of this many seconds and
                   transcribe each separately (word timestamps are offset by
                   the segment start, so output timing stays global)

    Returns:
        Dict with 'text' and 'chunks' keys containing transcription and word timestamps
    """
    # Read the audio file
    waveform, sr = sf.read(audio_path)

    # Convert to mono if needed
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Split into segments if chunking is enabled
    if chunk_sec and len(waveform) > chunk_sec * sr:
        seg_len = int(chunk_sec * sr)
        segments = [
            (i / sr, waveform[i : i + seg_len])
            for i in range(0, len(waveform), seg_len)
        ]
    else:
        segments = [(0.0, waveform)]

    # Build output dict
    chunks = []
    text = ""
    for offset_sec, seg in segments:
        for w in _transcribe_segment(seg, sr, asr_model):
            word = w["word"]
            text += word + " "
            chunks.append({
                "text": word,
                "timestamp": [w["start"] + offset_sec, w["end"] + offset_sec],
            })

    return {
        "text": text.strip(),
        "chunks": chunks,
    }


def find_audio_files(
    input_dir: str,
    extensions: Set[str],
    filenames: Optional[Set[str]] = None
) -> List[Dict]:
    """
    Recursively find all audio files in the input directory.

    Args:
        input_dir: Root directory to search for audio files
        extensions: Set of file extensions to include (e.g., {'.wav', '.mp3'})
        filenames: If given, only files with these exact names are included
                   (e.g., {'output.wav', 'speaker_1_audio.flac'})

    Returns:
        List of dicts with audio file information
    """
    audio_files = []
    input_dir = os.path.abspath(input_dir)

    for root, dirs, files in os.walk(input_dir):
        # Skip hidden directories (e.g., .batch_work in gemini results)
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            # Check filename whitelist
            if filenames is not None and filename not in filenames:
                continue
            # Check extension (case-insensitive)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in extensions:
                continue
            
            audio_path = os.path.join(root, filename)
            
            # Calculate relative path from input_dir
            relative_dir = os.path.relpath(root, input_dir)
            if relative_dir == ".":
                relative_dir = ""
            
            # Output filename (same name but .json extension)
            output_filename = os.path.splitext(filename)[0] + ".json"
            
            audio_files.append({
                "audio_path": audio_path,
                "filename": filename,
                "relative_dir": relative_dir,
                "output_filename": output_filename,
            })
    
    return audio_files


def process_audio_file(
    audio_info: Dict,
    asr_model,
    output_dir: str,
    skip_existing: bool = True,
    chunk_sec: float = 0.0
) -> Optional[str]:
    """
    Process a single audio file and save the transcript.
    
    Args:
        audio_info: Audio file info dict
        asr_model: Loaded ASR model
        output_dir: Root directory for saving JSON transcripts
        skip_existing: Skip if JSON already exists
    
    Returns:
        Path to output JSON file, or None if failed
    """
    # Construct output path
    if audio_info["relative_dir"]:
        output_subdir = os.path.join(output_dir, audio_info["relative_dir"])
    else:
        output_subdir = output_dir
    
    json_path = os.path.join(output_subdir, audio_info["output_filename"])
    
    # Check if already processed
    if skip_existing and os.path.exists(json_path):
        return json_path
    
    # Create output directory
    os.makedirs(output_subdir, exist_ok=True)
    
    try:
        # Transcribe
        transcript = transcribe_audio_file(audio_info["audio_path"], asr_model, chunk_sec=chunk_sec)

        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(transcript, f, indent=2)

        return json_path
    except Exception as e:
        print(f"Error processing {audio_info['audio_path']}: {e}")
        error_log = os.path.join(output_dir, "asr_errors.log")
        with open(error_log, "a") as f:
            f.write(f"{audio_info['audio_path']}: {type(e).__name__}: {e}\n")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generic batch transcription for any audio file hierarchy"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing audio files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for JSON transcripts (will mirror input hierarchy)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help="Audio file extensions to process (default: .wav .mp3 .flac .ogg .m4a .aac .wma)"
    )
    parser.add_argument(
        "--filenames",
        type=str,
        nargs="+",
        default=None,
        help="Only process files with these exact names "
             "(e.g., output.wav speaker_1_audio.flac speaker_2_audio.flac). "
             "Default: process all audio files"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip audio files that already have JSON transcripts"
    )
    parser.add_argument(
        "--local_attention",
        action="store_true",
        help="Switch the encoder to local attention for long audio "
             "(full attention OOMs on files longer than ~11 min)"
    )
    parser.add_argument(
        "--att_context_size",
        type=int,
        default=256,
        help="Left/right context size for local attention (default: 256)"
    )
    parser.add_argument(
        "--chunk_sec",
        type=float,
        default=0.0,
        help="Split audio into segments of this many seconds and transcribe "
             "each separately, offsetting word timestamps (0 = no chunking). "
             "Use for long files on small GPUs"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list files that would be processed, don't run ASR"
    )
    
    args = parser.parse_args()
    
    # Parse extensions
    if args.extensions:
        extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions}
        extensions = {ext.lower() for ext in extensions}
    else:
        extensions = DEFAULT_AUDIO_EXTENSIONS
    
    filenames = set(args.filenames) if args.filenames else None

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Audio extensions: {', '.join(sorted(extensions))}")
    if filenames:
        print(f"Filename filter: {', '.join(sorted(filenames))}")

    # Find all audio files
    print("\nScanning for audio files...")
    audio_files = find_audio_files(args.input_dir, extensions, filenames)
    print(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("No audio files found. Please check the input directory and extensions.")
        return
    
    # Dry run - just list files
    if args.dry_run:
        print("\nDry run - files that would be processed:")
        for audio_info in audio_files[:20]:  # Show first 20
            rel_path = os.path.join(audio_info["relative_dir"], audio_info["filename"])
            print(f"  {rel_path}")
        if len(audio_files) > 20:
            print(f"  ... and {len(audio_files) - 20} more files")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ASR model
    print("\nLoading ASR model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    ).cuda()
    print("ASR model loaded")

    if args.local_attention:
        # NVIDIA-recommended setup for long-audio inference with parakeet:
        # local attention keeps memory linear in audio length
        ctx = args.att_context_size
        print(f"Switching encoder to local attention (context {ctx}) for long audio")
        asr_model.change_attention_model("rel_pos_local_attn", [ctx, ctx])
        asr_model.change_subsampling_conv_chunking_factor(1)
    
    # Process files
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for audio_info in tqdm(audio_files, desc="Transcribing"):
        # Check if should skip
        if args.skip_existing:
            if audio_info["relative_dir"]:
                output_subdir = os.path.join(args.output_dir, audio_info["relative_dir"])
            else:
                output_subdir = args.output_dir
            json_path = os.path.join(output_subdir, audio_info["output_filename"])
            
            if os.path.exists(json_path):
                skipped_count += 1
                continue
        
        result = process_audio_file(
            audio_info,
            asr_model,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            chunk_sec=args.chunk_sec
        )
        
        if result is None:
            error_count += 1
        else:
            processed_count += 1
    
    print(f"\nProcessing complete:")
    print(f"  - Processed: {processed_count}")
    print(f"  - Skipped (existing): {skipped_count}")
    print(f"  - Errors: {error_count}")
    print(f"\nTranscripts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

