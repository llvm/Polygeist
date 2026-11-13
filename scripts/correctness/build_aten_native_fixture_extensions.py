#!/usr/bin/env python3
"""Cross-build real ATen baselines for the extracted fixtures missing in the
original native harness. These are benchmark implementations, never matcher
specifications or automatic compiler replacements. Several extracted stages
require ATen compositions, not a call to the complete named parent operator.
The original C fixture supplies the independent complete-output reference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import aten_pointwise_graph_silicon as bench

ROOT = Path(__file__).resolve().parents[2]
RECIPES = {
    "aten_batch_norm_transform_cpu": """
      auto x4=x_t.view({N,C,H,W});
      auto result=out_t.view({N,C,H,W});
      auto scaled=(x4-mean_t.view({1,C,1,1}))*invstd_t.view({1,C,1,1})*weight_t.view({1,C,1,1});
      at::add_out(result,scaled,bias_t.view({1,C,1,1}));
    """,
    "aten_cauchy_cpu": "at::add_out(out_t,sigma*at::tan(3.14159265358979323846f*(uniform_t-0.5f)),median);",
    "aten_conv2d_columns_cpu": "auto result=col_t.view({1,C*K*K,(H-2)*(W-2)}); at::im2col_out(result,x_t.view({1,C,H,W}),{K,K},{1,1},{0,0},{1,1});",
    "aten_convert_csr_to_coo_cpu": """
      static auto rows=at::arange(R,ptr_t.options());
      defer_output(row_t,at::repeat_interleave(rows,ptr_t.slice(0,1,R+1)-ptr_t.slice(0,0,R),0,N));
    """,
    "aten_depthwise_conv3x3_cpu": "auto result=out_t.view({B,C,H,W}); at::convolution_out(result,x_t.view({B,C,H,W}),weight_t.view({C,1,3,3}),bias_t,{1,1},{1,1},{1,1},false,{0,0},C);",
    "aten_dirichlet_grad_cpu": "at::mul_out(out_t,x_t,at::log(x_t+0.001f)-alpha_t/total_t);",
    "aten_embedding_bag_counts_cpu": "defer_output(out_t,at::bincount(index_t.to(at::kLong),{},E));",
    "aten_exponential_cpu": "at::div_out(out_t,-at::log1p(-uniform_t),lambda);",
    "aten_gamma_transform_cpu": "auto d=alpha_t-0.3333333f; auto v=1.0f+normal_t/at::sqrt(9.0f*d); at::mul_out(out_t,d*v*v,v);",
    "aten_geometric_cpu": "at::ceil_out(out_t,at::log1p(-uniform_t)/std::log(1.0f-probability));",
    "aten_log_normal_cpu": "at::exp_out(out_t,mean+std*standard_normal_t);",
    "aten_normal_cpu": "at::add_out(out_t,std*standard_normal_t,mean);",
    "aten_quant_col_offsets_cpu": "at::sub_out(out_t,at::sum(w_t.view({K,N}),{0},false,at::kInt),zero*K);",
    "aten_slow_conv3d_backward_input_cpu": "auto result=out_t.view({1,C,D+2,H+2,W+2}); at::convolution_out(result,g_t.view({1,O,D,H,W}),w_t.view({O,C,K,K,K}),{},{1,1,1},{0,0,0},{1,1,1},true,{0,0,0},1);",
    "aten_slow_conv3d_backward_weight_cpu": """
      static auto dummy=at::empty({O,C,K,K,K},x_t.options());
      auto grads=at::convolution_backward(g_t.view({1,O,D,H,W}),
        x_t.view({1,C,D+2,H+2,W+2}),dummy,{}, {1,1,1},{0,0,0},{1,1,1},
        false,{0,0,0},1,{false,true,false});
      defer_output(out_t,std::get<1>(grads));
    """,
    "aten_sparse_bmm_cpu": "auto result=out_t.view({B,M,N}); at::bmm_out(result,a_t.view({B,M,K}),b_t.view({B,K,N}));",
    "aten_sparse_intersection_apply_cpu": "at::mul_out(out_t,a_t,b_t);",
    "aten_sparse_intersection_launch_cpu": "at::mul_out(out_t,a_t,b_t);",
    "aten_sparse_matmul_cpu": """
      static auto sparse=at::sparse_csr_tensor(ap_t,ac_t,av_t,{R,R},av_t.options().layout(at::kSparseCsr));
      auto result=out_t.view({R,C}); at::mm_out(result,sparse,b_t.view({R,C}));
    """,
    "aten_sparse_matmul_csr_to_coo_cpu": """
      static auto rows=at::arange(R,ptr_t.options());
      defer_output(row_t,at::repeat_interleave(rows,ptr_t.slice(0,1,R+1)-ptr_t.slice(0,0,R),0,N));
    """,
    "aten_sspaddmm_cpu": """
      static auto indices=at::stack({row_t,col_t}).to(at::kLong);
      static auto sparse=at::sparse_coo_tensor(indices,val_t,{R,K},val_t.options().layout(at::kSparse)).coalesce();
      auto result=out_t.view({R,C}); at::mm_out(result,sparse,b_t.view({K,C}));
    """,
    "aten_standard_gamma_grad_cpu": "at::add_out(out_t,(x_t-a_t)/(a_t+0.001f),at::log(x_t+0.001f));",
    "aten_transform_bias_rescale_qkv_cpu": """
      auto input=qkv_t.view({B,S,3,H,D}); auto offsets=bias_t.view({3,H,D});
      defer_output(q_t,(input.select(2,0)+offsets.select(0,0)).permute({0,2,1,3})*scale);
      defer_output(k_t,(input.select(2,1)+offsets.select(0,1)).permute({0,2,1,3}));
      defer_output(v_t,(input.select(2,2)+offsets.select(0,2)).permute({0,2,1,3}));
    """,
    "aten_uniform_cpu": "at::add_out(out_t,(to-from)*uniform01_t,from);",
    "aten_upsample_bilinear2d": """
      // The extracted fixture uses source coordinates (oh/2, ow/2), not
      // PyTorch interpolate's default half-pixel convention.
      static auto hi=at::arange(2*H,input_t.options().dtype(at::kLong));
      static auto wi=at::arange(2*W,input_t.options().dtype(at::kLong));
      static auto h0=at::floor_divide(hi,2), w0=at::floor_divide(wi,2);
      static auto h1=at::clamp_max(h0+1,H-1), w1=at::clamp_max(w0+1,W-1);
      static auto fh=(at::remainder(hi,2).to(at::kFloat)*0.5f).view({1,1,2*H,1});
      static auto fw=(at::remainder(wi,2).to(at::kFloat)*0.5f).view({1,1,1,2*W});
      auto x=input_t.view({B,C,H,W});
      auto top=x.index_select(2,h0), bottom=x.index_select(2,h1);
      defer_output(output_t,(1-fh)*((1-fw)*top.index_select(3,w0)+fw*top.index_select(3,w1))+
                       fh*((1-fw)*bottom.index_select(3,w0)+fw*bottom.index_select(3,w1)));
    """,
    "aten_xor_sum_cpu": """
      // ATen has elementwise XOR but no public XOR reduction; use a tree
      // of genuine ATen calls and report this as an extracted-stage composition.
      auto current=x_t.view({R,K});
      for(int64_t width=K;width>1;width=(width+1)/2) {
        auto pairs=at::bitwise_xor(current.slice(1,0,width-1,2),current.slice(1,1,width,2));
        current=(width%2) ? at::cat({pairs,current.slice(1,width-1,width)},1) : pairs;
      }
      defer_output(out_t,current);
    """,
}


def adapter(name, cfg):
    types = {'ptr': ('float*', 'at::kFloat'), 'iptr': ('int*', 'at::kInt'),
             'bptr': ('signed char*', 'at::kChar'), 'dptr': ('double*', 'at::kDouble'),
             'scalar': ('float', None), 'dscalar': ('double', None), 'iscalar': ('int', None)}
    signature = ', '.join(types[k][0]+' '+n for n,k,*_ in cfg['args'])
    lines = ['#include <ATen/ATen.h>', '#include <ATen/Context.h>', '#include <cmath>',
             '#include <cstdio>', '#include <exception>', '#include <vector>',
             'static std::vector<std::pair<at::Tensor,at::Tensor>> pending_outputs;',
             'static void defer_output(at::Tensor dest, at::Tensor result) { pending_outputs.emplace_back(dest,result); }',
             'extern "C" void polygeist_native_materialize() { for(auto &p:pending_outputs) p.first.view(p.second.sizes()).copy_(p.second); pending_outputs.clear(); }',
             f'extern "C" void {name}({signature}) {{', 'try {',
             'pending_outputs.clear();',
             'static bool configured=[] { at::globalContext().setAllowTF32CuBLAS(false); at::globalContext().setAllowTF32CuDNN(false); return true; }();']
    lines += [f'constexpr int64_t {k}={v};' for k,v in cfg['dims'].items()]
    for n,k,size,*_ in cfg['args']:
        if types[k][1]:
            lines += [f'static auto {n}_t=at::from_blob({n},{{{size}}},at::TensorOptions().dtype({types[k][1]}).device(at::kCUDA));']
    lines += [RECIPES[name], '} catch(const std::exception &e) { std::fprintf(stderr,"ATen failure: %s\\n",e.what()); std::abort(); }', '}']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kernel',action='append')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch=Path(os.environ.get('ATEN_NATIVE_TORCH_ROOT','/tmp/aten_native_sysroot_24_12/usr/local/lib/python3.12/dist-packages/torch'))
    cuda=Path(os.environ.get('ATEN_NATIVE_CUDA_ROOT','/usr/local/cuda-12.6/targets/sbsa-linux'))
    cases=[]; failures=[]
    previous=args.output/'manifest.json'
    if args.kernel and previous.exists():
        saved=json.loads(previous.read_text())
        cases=[c for c in saved['cases'] if c['kernel'] not in args.kernel]
        failures=[c for c in saved['failures'] if c['kernel'] not in args.kernel]
    for name in args.kernel or RECIPES:
        cfg=bench._cfg_for(name); out=args.output/name; out.mkdir(exist_ok=True)
        source=out/'reference.c'; bench.scaled_source(name,cfg,source)
        harness=bench.harness_text(name,cfg).replace('raised_resident_wall_us','native_aten_resident_wall_us')
        (out/'harness.c').write_text(harness)
        (out/'adapter.cpp').write_text(adapter(name,cfg))
        commands=[
            ['aarch64-linux-gnu-gcc','-O3','-c',str(source),f'-D{name}={name}_reference','-o',str(out/'reference.o')],
            ['aarch64-linux-gnu-gcc','-O2','-DBENCH_RESIDENT_ONLY','-DBENCH_NATIVE_DEFERRED_OUTPUTS','-c',str(out/'harness.c'),'-o',str(out/'harness.o')],
            ['aarch64-linux-gnu-g++','-O2','-std=c++17','-DHAVE_SVE256_CPU_DEFINITION',
             '-I'+str(torch/'include'),'-I'+str(torch/'include/torch/csrc/api/include'),
             '-I'+str(cuda/'include'),str(out/'adapter.cpp'),str(out/'reference.o'),str(out/'harness.o'),
             '-L'+str(torch/'lib'),'-L'+str(cuda/'lib'),'-Wl,--allow-shlib-undefined',
             '-Wl,--no-as-needed','-ltorch','-ltorch_cpu','-ltorch_cuda','-lc10_cuda','-lc10',
             '-lcudart','-lpthread','-ldl','-lm','-o',str(out/name)]]
        rc=0
        with (out/'build.log').open('w') as log:
            for command in commands:
                rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
                if rc: break
        row=dict(kernel=name,config=cfg,commands=commands,scope='real ATen composition of extracted fixture',exit_code=rc)
        if rc: failures.append(row)
        else:
            row['executable']=str(out/name)
            row['sha256']=hashlib.sha256((out/name).read_bytes()).hexdigest()
            cases.append(row)
        (args.output/'manifest.json').write_text(json.dumps(dict(cases=cases,failures=failures),indent=2)+'\n')
        print(name,'FAIL' if rc else 'BUILT',flush=True)
    raise SystemExit(bool(failures))


if __name__=='__main__': main()
