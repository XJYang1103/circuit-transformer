#!/usr/bin/env bash
#
# Fetch the Circuit Transformer datasets from HuggingFace into
# ~/data/circuit-transformer/.
#
#   raw/     the archives exactly as published -- keep as provenance
#   train/   unzipped training data   (5,080,042 .json files)
#   test/    unzipped test data       (random_circuits/, IWLS_FFWs/)
#
# The training archive is 3.5 GiB and expands to just over 5 million small
# files, so the unzip step is a minutes-long operation. Pass --no-unzip to
# download now and expand later.
#
# Usage:
#   scripts/download_data.sh                 download, then unzip both archives
#   scripts/download_data.sh --no-unzip      download only
#   scripts/download_data.sh --unzip-only    unzip archives already in raw/
#   scripts/download_data.sh --verify        check sha256 sums against the pins
#   scripts/download_data.sh --help
#
# Alternate endpoint, useful outside the US/EU:
#   HF_ENDPOINT=https://hf-mirror.com scripts/download_data.sh
#
# Custom destination:
#   DATA_ROOT=/mnt/big/circuit-transformer scripts/download_data.sh
#
# The training data lands in the directory that configs/train_deepsyn.yaml
# points `paths.data_dir` at.
#
set -euo pipefail

REPO_ID="snowkylin/circuit-transformer"
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
BASE_URL="${ENDPOINT%/}/datasets/${REPO_ID}/resolve/main"
DATA_ROOT="${DATA_ROOT:-$HOME/data/circuit-transformer}"

TRAIN_ARCHIVE="8_inputs_2_outputs_random_deepsyn.zip"
TEST_ARCHIVE="test_data.zip"

# sha256 of the archives as published, recorded 2026-10-20. A mismatch means
# upstream replaced the file, which is worth knowing before training on it.
TRAIN_SHA256="9cdef0d4ad0623fb54805c219c9feb9f41963584faa2c5a546f0c531827c88a2"
TEST_SHA256="33eec532f892b527b7a5c5cffce07a870b589fefa2c06a8dcde12e211f725bcc"

DO_DOWNLOAD=1
DO_UNZIP=1
DO_VERIFY=0

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}

for arg in "$@"; do
    case "$arg" in
        --no-unzip)   DO_UNZIP=0 ;;
        --unzip-only) DO_DOWNLOAD=0 ;;
        --verify)     DO_VERIFY=1; DO_DOWNLOAD=0; DO_UNZIP=0 ;;
        -h|--help)    usage; exit 0 ;;
        *)            echo "Unknown option: $arg" >&2; echo >&2; usage >&2; exit 2 ;;
    esac
done

if [ "$DO_DOWNLOAD" = 1 ] && ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download the datasets." >&2
    exit 1
fi

download() {
    local name="$1"
    local dest="$DATA_ROOT/raw/$name"
    local part="$dest.part"

    if [ -s "$dest" ]; then
        echo "  [skip] $name (already in raw/)"
        return 0
    fi

    mkdir -p "$DATA_ROOT/raw"
    echo "  [get ] $name"
    # -L: HF redirects to a CDN. -C -: resume a partial download if one exists.
    curl -fL --retry 3 --retry-delay 5 -C - -o "$part" "$BASE_URL/$name"
    mv "$part" "$dest"
}

verify() {
    local name="$1" expected="$2"
    local file="$DATA_ROOT/raw/$name"
    local actual

    if [ ! -s "$file" ]; then
        echo "  [MISS] $name is not in raw/"
        return 1
    fi

    actual="$(sha256sum "$file" | cut -d' ' -f1)"
    if [ "$actual" = "$expected" ]; then
        echo "  [ ok ] $name"
        return 0
    fi

    echo "  [BAD ] $name"
    echo "         expected $expected"
    echo "         actual   $actual"
    return 1
}

extract() {
    local archive="$1" dest="$2"

    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$archive" -d "$dest"
    else
        python3 - "$archive" "$dest" <<'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
PY
    fi
}

unzip_one() {
    local name="$1" dest="$2"
    local archive="$DATA_ROOT/raw/$name"

    if [ ! -s "$archive" ]; then
        echo "  [MISS] raw/$name not found -- drop --unzip-only to download it"
        return 1
    fi

    if [ -n "$(ls -A "$dest" 2>/dev/null || true)" ]; then
        echo "  [skip] $dest/ is not empty (remove it to force a re-extract)"
        return 0
    fi

    echo "  [unzip] $name -> $dest/  (millions of small files, this takes a while)"
    mkdir -p "$dest"
    extract "$archive" "$dest"
}

echo "Circuit Transformer datasets"
echo "  endpoint : $ENDPOINT"
echo "  data root: $DATA_ROOT"
echo

if [ "$DO_DOWNLOAD" = 1 ]; then
    echo "Downloading:"
    download "$TRAIN_ARCHIVE"
    download "$TEST_ARCHIVE"
    download "README.md"
    echo
fi

if [ "$DO_VERIFY" = 1 ]; then
    echo "Verifying:"
    status=0
    verify "$TRAIN_ARCHIVE" "$TRAIN_SHA256" || status=1
    verify "$TEST_ARCHIVE" "$TEST_SHA256" || status=1
    exit "$status"
fi

if [ "$DO_UNZIP" = 1 ]; then
    echo "Extracting:"
    unzip_one "$TRAIN_ARCHIVE" "$DATA_ROOT/train"
    unzip_one "$TEST_ARCHIVE" "$DATA_ROOT/test"
    echo
fi

echo "Layout:"
echo "  $DATA_ROOT/raw/    archives, exactly as published"
echo "  $DATA_ROOT/train/  training .json files"
echo "                     -> paths.data_dir in configs/train_deepsyn.yaml"
echo "  $DATA_ROOT/test/   random_circuits/ and IWLS_FFWs/"
