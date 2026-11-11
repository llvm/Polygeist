from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'staged-cub';(OUT/'ref_c').mkdir(exist_ok=True);(OUT/'manifest').mkdir(exist_ok=True)
source=ROOT/'third_party/cuda_headers/cuda_cccl/include/cccl/cub/thread/thread_scan.cuh';text=source.read_text();start=text.index('ThreadScanInclusive(T inclusive, T* input, T* output, ScanOp scan_op, detail::constant_t<LENGTH>')
start=text.index('{',start);end=text.index('\n}',start);body=text[start+1:end]
assert 'inclusive = scan_op(inclusive, input[i]);' in body and 'output[i] = inclusive;' in body
body=body.replace('  _CCCL_PRAGMA_UNROLL_FULL()\n','').replace('LENGTH','n').replace('scan_op(inclusive, input[i])','inclusive + input[i]').replace('\n  return inclusive;','')
ref='''/* C specialization of pinned CUB ThreadScanInclusive with explicit seed.
 * T=float, ScanOp=cuda::std::plus<float>, LENGTH is a runtime extent.
 * The unused returned aggregate is discarded; each output and operation order
 * are preserved. DeviceScan::InclusiveScanInit is the corresponding API;
 * its parallel FP association is separately checked within declared tolerance. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n, float inclusive, float input[], float output[]) {
'''+body+'\n}\n'
(OUT/'ref_c/cub_inclusive_scan.c').write_text(ref)
headers=[source,ROOT/'third_party/cuda_headers/cuda_cccl/include/cccl/cub/device/device_scan.cuh']
manifest=dict(routine='cub_inclusive_scan',source='Pinned public CUB seeded ThreadScanInclusive implementation',upstream_files=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in headers],adaptation='T=float and ScanOp=cuda::std::plus<float>; runtime extent replaces template LENGTH; discarded final aggregate return. The seed remains a parameter. DeviceScan::InclusiveScanInit uses the same seed and inclusive output contract, with parallel reassociation checked separately.',specs=[dict(name='cub_inclusive_scan_f32',target=dict(library='CUB',call='DeviceScanInclusiveScanInit',backends=['cuda'],runtime_adapter=dict(schema=1,kind='typed_c_wrapper',entry='polygeist_cub_scan_inclusive_seed_f32',arguments=['n','inclusive','input','output'],arrays={'input':['n'],'output':['n']},bounds={'n':{'min':1,'max':2147483647}},alias_policy='disjoint')),args=['n','inclusive','input','output'],assumptions=['positive n <= INT_MAX; contiguous disjoint FP32 input and output','parallel backend reassociation requires declared numerical tolerance; no strict bitwise backend equivalence claim'],deviations=[])])
(OUT/'manifest/cub_inclusive_scan.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'upstream-specialization.cpp').write_text('template<int LENGTH,typename T,typename ScanOp>\nT original(T inclusive,T*input,T*output,ScanOp scan_op)\n'+text[start:end+2].replace('_CCCL_PRAGMA_UNROLL_FULL()','')+'\n')
print('Staged seeded CUB reference and pinned provenance')
