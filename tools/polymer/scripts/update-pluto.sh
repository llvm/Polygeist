#!/usr/bin/env bash
# Sync pluto as a submodule.

set -o errexit
set -o pipefail
set -o nounset

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"
POLYMER_ROOT="${DIR}/../"

cd "${POLYMER_ROOT}"
git submodule sync
git submodule update --init --recursive "${POLYMER_ROOT}/pluto"
cd - &>/dev/null
