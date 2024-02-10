#!/usr/bin/env python3

# Find all the log files under this directory and look for pass timing.
import os
import argparse
import glob

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('work_dir', type=str)
  args = parser.parse_args()

  results = []
  pluto_results = []
  for fp in glob.glob(os.path.join(args.work_dir, '**', '*.log')):

    with open(fp, 'r') as f:
      lines = [line.strip() for line in f.readlines()]
      has_timing_report = any(('timing report' in line) for line in lines)
      has_pluto = any(('[pluto] Timing statistics' in line) for line in lines)

      if has_timing_report:
        pos = next(i for i, line in enumerate(lines) if ('timing report' in line))
        line = next(x for x in lines[pos:] if 'PlutoTransformPass' in x)

        total_time = float(lines[pos+2].split()[-2])
        pluto_pass_time = float(line.split()[-4])
        pluto_time = float(next(x for x in lines if 'Pluto schedule elapsed' in x).split()[-1][:-1])
        name = '.'.join(os.path.basename(fp).split('.')[:4])

        results.append('{name},{total_time},{pluto_pass_time},{pluto_time:.4f}'.format(name=name, total_time=total_time, pluto_pass_time=pluto_pass_time, pluto_time=pluto_time))
      
      elif has_pluto:
        times = [str(float(y)) for y in next(x for x in lines if 'All times' in x).split()[-4:]]
        pluto_results.append('.'.join(os.path.basename(fp).split('.')[:-1]) + ',' + ','.join(times))

  
  results.sort()
  for line in results:
    print(line)
  pluto_results.sort()
  for line in pluto_results:
    print(line)

if __name__ == '__main__':
  main()
