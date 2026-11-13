#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
base="$repo_root/issues/ginsbach_asplos18/primary_sources"
original="$repo_root/third_party/ginsbach-original"
mkdir -p "$base" "$original"

paper_url=https://www.pure.ed.ac.uk/ws/portalfiles/portal/54253267/asplos.pdf
curl -fL --retry 3 "$paper_url" -o "$base/ginsbach_asplos18.pdf"
sha256sum "$base/ginsbach_asplos18.pdf" > "$base/SHA256SUMS"
pdftotext -layout "$base/ginsbach_asplos18.pdf" "$base/ginsbach_asplos18.txt"

clone_exact() {
  local url=$1
  local branch=$2
  local commit=$3
  local destination=$4
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none --single-branch --branch "$branch" "$url" "$destination"
  fi
  git -C "$destination" fetch origin "$branch"
  git -C "$destination" checkout --detach "$commit"
  test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
}

clone_exact https://github.com/asplos18ginsbach/IDL-Demo.git master \
  1edfb4086a07f1efe2f6b4cf568fe58a2feecfb8 "$original/IDL-Demo"
clone_exact https://github.com/cc18ginsbach/llvm.git cc18ginsbach \
  5d26ac5896055927238d4c3ec01608ae07b7f9d0 "$original/llvm"
clone_exact https://github.com/asplos18ginsbach/clang.git cc18ginsbach \
  818167bc99a0f5538a2676bf876e5c598d1b449b "$original/llvm/tools/clang"

{
  printf 'paper_sha256\t'
  sha256sum "$base/ginsbach_asplos18.pdf" | awk '{print $1}'
  for path in "$original/IDL-Demo" "$original/llvm" "$original/llvm/tools/clang"; do
    printf '%s\t%s\t%s\n' "$path" "$(git -C "$path" rev-parse HEAD)" \
      "$(git -C "$path" show -s --format=%cI HEAD)"
  done
} > "$base/acquisition_versions.tsv"
