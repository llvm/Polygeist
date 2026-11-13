// RUN: /usr/bin/python3 %S/../Inputs/aten_final_routes/check.py | FileCheck %s
// CHECK: PASS: unnamed depthwise, reverse, mean and prefix routes; invalid domains rejected
