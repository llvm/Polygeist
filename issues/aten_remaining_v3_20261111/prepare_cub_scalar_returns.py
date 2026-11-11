from pathlib import Path
import json
O=Path(__file__).resolve().parent;D=O/'staged-cub-scalar-returns';D.mkdir(exist_ok=True)
for s in ['ref_c','manifest','raised']:(D/s).mkdir(exist_ok=True)
s=(O/'staged-cub-reductions/ref_c/cub_reduce.c').read_text().replace('void NETLIB_SPEC(int n, SCALAR seed, SCALAR input[], SCALAR output[1])','SCALAR NETLIB_SPEC(int n, SCALAR seed, SCALAR input[])').replace('output[0] = retval;', 'return retval;')
(D/'ref_c/cub_reduce_value.c').write_text(s)
m=json.loads((O/'staged-cub-reductions/manifest/cub_reduce.json').read_text());m['routine']='cub_reduce_value';m['adaptation']=m['adaptation'].replace(', and expose the return as one output element','; retain its scalar return')
m['specs']=[x for x in m['specs'] if x['defines']['REDUCE_PRODUCT']==0]
for x in m['specs']:
 x['name']=x['name'].replace('cub_reduce_sum_','cub_reduce_value_sum_');x['args'].remove('output');cfg=x['target']['runtime_adapter'];cfg['entry']=cfg['entry'].replace('cub_reduce_sum_','cub_reduce_value_sum_');cfg['arguments'].remove('output');cfg['arrays'].pop('output');x['assumptions'][0]='positive n <= INT_MAX, contiguous read-only input and scalar result'
(D/'manifest/cub_reduce_value.json').write_text(json.dumps(m,indent=2)+'\n')
