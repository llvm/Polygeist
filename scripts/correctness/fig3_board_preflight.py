#!/usr/bin/env python3
"""Read-only, credential-free idle/clock/thermal probe for CPU measurements."""

import json
import os
from pathlib import Path
import re
import subprocess
import time


def processes():
    result = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = path.read_text()
            end = text.rindex(")")
            fields = text[end + 2:].split()
            result[int(path.parent.name)] = (
                text[text.index("(") + 1:end], int(fields[11]) + int(fields[12]))
        except (OSError, ValueError, IndexError):
            pass
    return result


def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return "unavailable"


before = processes()
start = time.monotonic()
time.sleep(2)
after = processes()
elapsed = time.monotonic() - start
busy = []
for pid, (name, ticks) in after.items():
    if pid in before and pid != os.getpid():
        percent = 100 * (ticks - before[pid][1]) / os.sysconf("SC_CLK_TCK") / elapsed
        if percent > 10:
            busy.append(dict(pid=pid, command=name, cpu_percent=percent))
clocks = [{"policy": p.name, **{name: read(p / name) for name in (
    "scaling_governor", "scaling_min_freq", "scaling_max_freq", "scaling_cur_freq")}}
    for p in Path("/sys/devices/system/cpu/cpufreq").glob("policy*")]
temperatures = [{"sensor": read(p / "type"), "millicelsius": read(p / "temp")}
                for p in Path("/sys/class/thermal").glob("thermal_zone*")]
fans = {str(p): read(p) for p in Path("/sys/class/hwmon").glob("hwmon*/pwm*")
        if p.is_file()}
probe = subprocess.run(["timeout", "3", "tegrastats", "--interval", "1000"],
                       capture_output=True, text=True)
stats = probe.stdout + probe.stderr
gpu_busy = any(int(v) > 0 for v in re.findall(r"(?:GR3D_FREQ|IGPU_FREQ)\s+(\d+)%", stats))
fixed = bool(clocks) and all(p["scaling_min_freq"] == p["scaling_max_freq"]
                             and p["scaling_min_freq"].isdigit() for p in clocks)
hot = any(t["millicelsius"].isdigit() and int(t["millicelsius"]) >= 85000
          for t in temperatures)
result = dict(timestamp=time.time(), load=os.getloadavg(), busy=busy,
              clocks=clocks, temperatures=temperatures, fan_settings=fans,
              power_mode=read(Path("/var/lib/nvpmodel/status")),
              software=read(Path("/etc/nv_tegra_release")),
              tegrastats=stats, fixed_cpu_clocks=fixed, gpu_busy=gpu_busy,
              temperature_limit_c=85, accepted=not busy and not gpu_busy and fixed and not hot)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["accepted"] else 1)
