#!/bin/sh
set -eu

echo "PREFLIGHT_BEGIN"
echo "POWER_MODE"
nvpmodel -q 2>&1 || true
echo "CLOCK_STATE"
jetson_clocks --show 2>&1 || true
echo "ACTIVE_CPU_COMMANDS"
ps -eo comm=,pcpu= | awk '$2 >= 10.0 { print }' | sort -k2,2nr || true
echo "TEGRastats"
if command -v tegrastats >/dev/null 2>&1; then
  timeout 2 tegrastats --interval 500 2>&1 || true
else
  echo "unavailable"
fi
echo "PREFLIGHT_END"
