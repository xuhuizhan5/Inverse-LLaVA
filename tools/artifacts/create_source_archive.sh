#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <absolute-repository-path> <absolute-output.tar.gz>" >&2
  exit 2
fi

repo_root="$1"
destination="$2"
if [[ "$repo_root" != /* || ! -d "$repo_root" || -L "$repo_root" ]]; then
  echo "Repository must be an absolute, existing, unsymlinked directory." >&2
  exit 2
fi
if [[ "$destination" != /* || "$destination" != *.tar.gz ]]; then
  echo "Destination must be an absolute .tar.gz path." >&2
  exit 2
fi
if [[ -e "$destination" || -L "$destination" ]]; then
  echo "Refusing to overwrite an existing destination: $destination" >&2
  exit 2
fi
if [[ ! -d "$(dirname "$destination")" ]]; then
  echo "Destination parent does not exist: $(dirname "$destination")" >&2
  exit 2
fi

repo_root="$(realpath "$repo_root")"
source_details="$(python3 "$repo_root/scripts/hash_execution_source.py" "$repo_root" --details)"
inventory="$(mktemp "${TMPDIR:-/tmp}/invllava-source-inventory.XXXXXX")"
listing="$(mktemp "${TMPDIR:-/tmp}/invllava-source-listing.XXXXXX")"
trap 'rm -f -- "$inventory" "$listing"' EXIT
python3 "$repo_root/scripts/hash_execution_source.py" "$repo_root" --list > "$inventory"

# COPYFILE_DISABLE prevents macOS tar from serializing extended attributes as
# AppleDouble files that later appear as real `._*` files on Linux.
COPYFILE_DISABLE=1 tar --no-xattrs --owner 0 --group 0 -C "$repo_root" -czf "$destination" \
  -T "$inventory"
tar -tzf "$destination" > "$listing"

if ! cmp -s "$inventory" "$listing"; then
  echo "Archive inventory differs from the reviewed execution-source inventory." >&2
  exit 1
fi
if grep -Eq '(^|/)\._|(^|/)\.DS_Store$' "$listing"; then
  echo "Archive contains platform metadata and is unsafe to stage." >&2
  exit 1
fi
required_entries=(
  pyproject.toml
  requirements.txt
  scripts/hash_execution_source.py
  src/invllava/__init__.py
  src/invllava/artifacts/source.py
  scripts/launch_train.py
)
for required_entry in "${required_entries[@]}"; do
  if ! grep -Fxq "$required_entry" "$listing"; then
    echo "Archive is missing a required execution file: $required_entry" >&2
    exit 1
  fi
done

archive_sha256="$(shasum -a 256 "$destination" | awk '{print $1}')"
printf '%s\n' "$source_details"
printf 'archive_sha256=%s bytes=%s path=%s\n' \
  "$archive_sha256" "$(wc -c < "$destination" | tr -d ' ')" "$destination"
