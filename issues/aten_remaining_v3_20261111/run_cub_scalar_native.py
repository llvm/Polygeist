from pathlib import Path
import importlib.util,hashlib,json,subprocess
OUT=Path(__file__).resolve().parent/'staged-cub-scalar-returns';s=importlib.util.spec_from_file_location('board',Path.home()/'.config/polygeist-silicon/reproduction_20261109/remote.py');B=importlib.util.module_from_spec(s);s.loader.exec_module(B)
p=OUT/'native_validation';remote='/tmp/aten-cub-scalar-returns-native-validation-20261111';assert B.upload(p,remote)==0
rc,log=B.run('chmod +x '+remote+'; LD_LIBRARY_PATH=/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu POLYGEIST_CUB_TRACE=1 timeout 60 '+remote,timeout=65)
(OUT/'native-gpu.log').write_text(log);(OUT/'native-gpu.json').write_text(json.dumps(dict(rc=rc,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),host_uptime=subprocess.check_output(['uptime'],text=True),vendor_calls=log.count('POLYGEIST_CUB_CALL')),indent=2)+'\n');print('GPU rc',rc);print('\n'.join(log.splitlines()[-7:]))
