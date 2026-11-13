#!/usr/bin/env python3
"""Extract the pinned NVIDIA PyTorch AArch64 headers/libraries for cross builds.
Only the public image's torch installation is extracted; no image code runs.
Requires skopeo and a Python version with tarfile's data extraction filter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

IMAGE='nvcr.io/nvidia/pytorch@sha256:8b8ce6c8b0d34fea26602d52a0c1a4364339773e900047cbd9f3511eb7b79e64'
PREFIX='usr/local/lib/python3.12/dist-packages/torch/'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--use-cache',action='store_true',help='Verify and reuse an already downloaded pinned image')
    args=p.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.use_cache:
        subprocess.run(['skopeo','copy','--override-os','linux','--override-arch','arm64',
                        'docker://'+IMAGE,'dir:'+str(args.cache)],check=True)
    manifest_bytes=(args.cache/'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest()!=IMAGE.rsplit(':',1)[1]:
        raise RuntimeError('Manifest does not match the pinned image')
    manifest=json.loads(manifest_bytes)
    counts={}
    for layer in manifest['layers']:
        digest=layer['digest'].split(':',1)[1]
        blob=args.cache/digest
        hasher=hashlib.sha256()
        with blob.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                hasher.update(chunk)
        checksum=hasher.hexdigest()
        if checksum!=digest: raise RuntimeError('Layer checksum mismatch')
        with tarfile.open(blob,'r:*') as archive:
            members=[m for m in archive.getmembers() if m.name.lstrip('./').startswith(PREFIX)]
            if members:
                archive.extractall(args.output,members=members,filter='data')
                counts[digest]=len(members)
    root=args.output/PREFIX
    for file in ['include/ATen/ATen.h','lib/libtorch_cuda.so']:
        if not (root/file).is_file():raise RuntimeError('Missing '+file)
    (args.output/'provenance.json').write_text(json.dumps(dict(image=IMAGE,torch_root=str(root),layers=counts),indent=2)+'\n')
    print('ATEN_NATIVE_TORCH_ROOT='+str(root))

if __name__=='__main__':main()
