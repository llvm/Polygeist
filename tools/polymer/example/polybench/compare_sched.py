#!/usr/bin/env python3

def get_scheds(file):
  scheds = []
  with open(file, 'r') as f:
    while True:
      line = f.readline()
      if not line:
        break
      if not line.startswith('T'):
        continue
      line = line.strip()

      # Process the schedule string
      sched = line.split(':')[1].strip() 
      sched = sched[1:-1].split(',')
      sched = [s.strip() for s in sched]
      scheds.append(sched)
  return scheds

def compare_sched(file1, file2):
  scheds1 = get_scheds(file1)
  scheds2 = get_scheds(file2)

  for sched1, sched2 in zip(scheds1, scheds2):
    if len(sched1) != len(sched2):
      return False

    imap = {}
    for i in range(len(sched1)):
      if sched1[i].isalpha():
        imap[sched1[i]] = sched2[i]

    for i in range(len(sched1)):
      s1 = sched1[i]
      for k, v in imap.items():
        s1 = s1.replace(k, v) 
      if s1 != sched2[i]:
        return False

  return True
