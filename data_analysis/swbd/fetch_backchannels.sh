#!/usr/bin/env bash
# Fetch the VAP / Ekstedt Switchboard backchannel list required by
# per_conversation_swbd.py (~6.5 MB, not committed).
#
# Source: https://github.com/ErikEkstedt/VoiceActivityProjection
#         (dataset_swb/backchannels.csv)
set -euo pipefail

URL="https://raw.githubusercontent.com/ErikEkstedt/VoiceActivityProjection/main/dataset_swb/backchannels.csv"
SHA256="fb535e437da49eebf90d88d81a6ba2eaab664394c149a4a80b917bf3aa3910de"
out="${1:-backchannels.csv}"

echo "Downloading backchannels.csv -> ${out}"
if command -v wget >/dev/null 2>&1; then
    wget -O "${out}" "${URL}"
else
    curl -fsSL "${URL}" -o "${out}"
fi

if command -v sha256sum >/dev/null 2>&1; then
    echo "${SHA256}  ${out}" | sha256sum -c -
else
    echo "${SHA256}  ${out}" | shasum -a 256 -c -
fi
echo "Done."
