from pathlib import Path
import subprocess,json
O=Path(__file__).resolve().parent;R=O.parents[1];D=O/'staged-cub-scalar-returns';objs=[];cmds=[]
with (D/'native-build.log').open('w') as log:
 for s in json.loads((D/'manifest/cub_reduce_value.json').read_text())['specs']:
  out=D/(s['name']+'.aarch64.o');cmd=['aarch64-linux-gnu-gcc','-O0','-ffp-contract=off','-DNETLIB_SPEC='+s['name']]+['-D'+k+'='+str(v) for k,v in s['defines'].items()]+['-c',str(D/'ref_c/cub_reduce_value.c'),'-o',str(out)];cmds.append(cmd);subprocess.run(cmd,check=True,stdout=log,stderr=log);objs.append(str(out))
 cmd=['/usr/local/cuda-12.6/bin/nvcc','-O2','-std=c++17','-arch=sm_87','-ccbin=aarch64-linux-gnu-g++','-target-dir','sbsa-linux','-I'+str(R/'third_party/cuda_headers/cuda_cccl/include/cccl'),'-I'+str(R/'runtime'),'-c',str(D/'runtime.cu'),'-o',str(D/'runtime.o')];cmds.append(cmd);subprocess.run(cmd,check=True,stdout=log,stderr=log)
 cmd=['aarch64-linux-gnu-g++','-O0','-std=c++17','-ffp-contract=off','-I/usr/local/cuda-12.6/targets/sbsa-linux/include',str(D/'native_validation.cpp'),str(D/'runtime.o')]+objs+['-L/usr/local/cuda-12.6/targets/sbsa-linux/lib','-lcudart','-Wl,-rpath,/home/nvidia/cuda-12.6/lib64','-o',str(D/'native_validation')];cmds.append(cmd);subprocess.run(cmd,check=True,stdout=log,stderr=log)
(D/'native-build.json').write_text(json.dumps(dict(commands=cmds,passed=True),indent=2)+'\n');print('native scalar-return build passed',flush=True)
