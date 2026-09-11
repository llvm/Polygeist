#!/usr/bin/env python3
"""Generate specialized standalone-C forms for remaining ATen loop bodies.

The fixtures preserve the numerical loop carried by the named upstream body
while fixing ranks, extents, dtypes, and optional modes so cgeist can expose a
static loop nest.  PyTorch allocation, dispatch, shape checks, and Tensor-list
orchestration deliberately remain outside the fixture.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "issues/aten_c_kernels"
MANIFEST = OUT / "generated_remaining_provenance.csv"
E: list[tuple[str, str, str, str]] = []


def add(name: str, source: str, token: str, code: str) -> None:
    E.append((f"aten_{name}", source, token, code.strip() + "\n"))


def reductions() -> None:
    s = "aten/src/ATen/native/ReduceOps.cpp"
    add("cumprod_backward_cpu", s, "cumprod_backward", """
#define N 128
void aten_cumprod_backward_cpu(float x[N],float prod[N],float grad[N],float out[N]){for(int i=0;i<N;++i){float v=0;for(int j=i;j<N;++j){float p=1;for(int k=0;k<=j;++k)if(k!=i)p*=x[k];v+=grad[j]*p;}out[i]=v;}}
""")
    add("cummax_cummin_cpu", s, "cummax_cummin_helper", """
#define R 16
#define N 64
void aten_cummax_cummin_cpu(float x[R][N],int is_max,float out[R][N],int index[R][N]){for(int r=0;r<R;++r){float v=x[r][0];int q=0;for(int i=0;i<N;++i){if((is_max&&x[r][i]>=v)||(!is_max&&x[r][i]<=v)){v=x[r][i];q=i;}out[r][i]=v;index[r][i]=q;}}}
""")
    add("diff_cpu", s, "diff_helper", "#define N 128\nvoid aten_diff_cpu(float x[N],float out[N-1]){for(int i=0;i<N-1;++i)out[i]=x[i+1]-x[i];}")
    add("gradient_cpu", s, "gradient_helper", "#define N 128\nvoid aten_gradient_cpu(float x[N],float h,float out[N]){out[0]=(x[1]-x[0])/h;for(int i=1;i<N-1;++i)out[i]=(x[i+1]-x[i-1])/(2*h);out[N-1]=(x[N-1]-x[N-2])/h;}")
    add("gradient_float_cpu", s, "gradient_helper_float", "#define N 128\nvoid aten_gradient_float_cpu(float x[N],float coord[N],float out[N]){out[0]=(x[1]-x[0])/(coord[1]-coord[0]);for(int i=1;i<N-1;++i)out[i]=(x[i+1]-x[i-1])/(coord[i+1]-coord[i-1]);out[N-1]=(x[N-1]-x[N-2])/(coord[N-1]-coord[N-2]);}")
    add("trace_cpu", s, "trace_cpu", "#ifndef N\n#define N 64\n#endif\nvoid aten_trace_cpu(float x[N][N],float out[1]){float v=0;for(int i=0;i<N;++i)v+=x[i][i];out[0]=v;}")
    add("allany_dims_cpu", s, "allany_dims_default", "#define R 32\n#define C 64\nvoid aten_allany_dims_cpu(int x[R][C],int all,int out[R]){for(int r=0;r<R;++r){int v=all;for(int c=0;c<C;++c)v=all?(v&&x[r][c]):(v||x[r][c]);out[r]=v;}}")
    add("std_var_all_cpu", s, "std_var_all_cpu", "#define N 1024\nvoid aten_std_var_all_cpu(float x[N],float out[1]){float m=0;for(int i=0;i<N;++i)m+=x[i];m/=N;float v=0;for(int i=0;i<N;++i){float d=x[i]-m;v+=d*d;}out[0]=v/(N-1);}")
    add("equal_cpu", s, "cpu_equal", "#define N 1024\nvoid aten_equal_cpu(float a[N],float b[N],int out[1]){int v=1;for(int i=0;i<N;++i)v=v&&(a[i]==b[i]);out[0]=v;}")


def blas() -> None:
    s = "aten/src/ATen/native/cpu/BlasKernel.cpp"
    add("blas_scale_cpu", s, "scale_", "#define N 1024\nvoid aten_blas_scale_cpu(float x[N],float a){for(int i=0;i<N;++i)x[i]*=a;}")
    add("blas_sum_cpu", s, "sum", "#ifndef N\n#define N 1024\n#endif\nvoid aten_blas_sum_cpu(float x[N],float out[1]){float v=0;for(int i=0;i<N;++i)v+=x[i];out[0]=v;}")
    for name, token, body in [
        ("gemm_transa_cpu", "gemm_transa_", "a[k][i]"),
        ("gemm_transb_cpu", "gemm_transb_impl", "a[i][k]"),
        ("gemm_transab_cpu", "gemm_transab_", "a[k][i]"),
        ("gemm_notrans_cpu", "gemm_notrans_", "a[i][k]"),
    ]:
        b = "b[j][k]" if "transb" in name or "transab" in name else "b[k][j]"
        add(name, s, token, f"#define M 32\n#define N 48\n#define K 40\nvoid aten_{name}(float a[M][K],float b[K][N],float c[M][N]){{for(int i=0;i<M;++i)for(int j=0;j<N;++j)for(int k=0;k<K;++k)c[i][j]+={body}*{b};}}")
    add("blas_axpy_cpu", s, "cpublas_axpy_impl", "#define N 1024\nvoid aten_blas_axpy_cpu(float a,float x[N],float y[N]){for(int i=0;i<N;++i)y[i]+=a*x[i];}")
    add("blas_copy_cpu", s, "cpublas_copy_impl", "#define N 1024\nvoid aten_blas_copy_cpu(float x[N],float y[N]){for(int i=0;i<N;++i)y[i]=x[i];}")
    s2 = "aten/src/ATen/native/CPUBlas.cpp"
    add("cpu_blas_gemm_cpu", s2, "gemm", "#define M 24\n#define N 32\n#define K 40\nvoid aten_cpu_blas_gemm_cpu(float a[M][K],float b[K][N],float c[M][N]){for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=a[i][k]*b[k][j];c[i][j]=v;}}")
    add("cpu_blas_gemm_batched_cpu", s2, "gemm_batched_generic", "#define B 4\n#define M 16\n#define N 20\n#define K 24\nvoid aten_cpu_blas_gemm_batched_cpu(float a[B][M][K],float b[B][K][N],float c[B][M][N]){for(int q=0;q<B;++q)for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=a[q][i][k]*b[q][k][j];c[q][i][j]=v;}}")
    add("cpu_blas_gemm_strided_batched_cpu", s2, "gemm_batched_with_stride_generic", "#define B 4\n#define M 16\n#define N 20\n#define K 24\nvoid aten_cpu_blas_gemm_strided_batched_cpu(float a[B][M][K],float b[B][K][N],float c[B][M][N]){for(int q=0;q<B;++q)for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=a[q][i][k]*b[q][k][j];c[q][i][j]=v;}}")


def factories_and_spectral() -> None:
    s = "aten/src/ATen/native/TensorFactories.cpp"
    add("eye_cpu", s, "eye_out_cpu", "#define N 64\nvoid aten_eye_cpu(float out[N][N]){for(int i=0;i<N;++i)for(int j=0;j<N;++j)out[i][j]=i==j;}")
    add("randperm_cpu", s, "randperm_cpu", "#define N 256\nvoid aten_randperm_cpu(unsigned bits[N],int out[N]){for(int i=0;i<N;++i)out[i]=i;for(int i=N-1;i>0;--i){int j=bits[i]%(i+1);int t=out[i];out[i]=out[j];out[j]=t;}}")
    add("tril_indices_cpu", s, "tril_indices_cpu", "#define N 32\n#define K (N*(N+1)/2)\nvoid aten_tril_indices_cpu(int row[K],int col[K]){for(int i=0;i<N;++i)for(int j=0;j<=i;++j){int p=i*(i+1)/2+j;row[p]=i;col[p]=j;}}")
    add("triu_indices_cpu", s, "triu_indices_cpu", "#define N 32\n#define K (N*(N+1)/2)\nvoid aten_triu_indices_cpu(int row[K],int col[K]){int p=0;for(int i=0;i<N;++i)for(int j=i;j<N;++j){row[p]=i;col[p++]=j;}}")
    add("zeros_cpu", s, "zeros_symint", "#define N 4096\nvoid aten_zeros_cpu(float out[N]){for(int i=0;i<N;++i)out[i]=0;}")
    s = "aten/src/ATen/native/SpectralOps.cpp"
    add("fftshift_cpu", s, "fft_fftshift", "#define N 256\nvoid aten_fftshift_cpu(float x[N],float out[N]){for(int i=0;i<N;++i)out[i]=x[(i+N/2)%N];}")
    add("ifftshift_cpu", s, "fft_ifftshift", "#define N 255\nvoid aten_ifftshift_cpu(float x[N],float out[N]){for(int i=0;i<N;++i)out[i]=x[(i+(N+1)/2)%N];}")
    add("as_complex_cpu", s, "as_complex", "#define N 512\nvoid aten_as_complex_cpu(float x[N][2],float re[N],float im[N]){for(int i=0;i<N;++i){re[i]=x[i][0];im[i]=x[i][1];}}")
    add("fft_conjugate_symmetry_cpu", s, "_fft_fill_with_conjugate_symmetry_", "#define N 256\nvoid aten_fft_conjugate_symmetry_cpu(float re[N],float im[N]){for(int i=N/2+1;i<N;++i){re[i]=re[N-i];im[i]=-im[N-i];}}")


def indexing_sorting() -> None:
    s = "aten/src/ATen/native/Bucketization.cpp"
    add("lower_bound_cpu", s, "cus_lower_bound", "#define N 256\n#define M 128\nvoid aten_lower_bound_cpu(float x[N],float v[M],int out[M]){for(int q=0;q<M;++q){int l=0,r=N;while(l<r){int m=(l+r)/2;if(x[m]<v[q])l=m+1;else r=m;}out[q]=l;}}")
    add("upper_bound_cpu", s, "cus_upper_bound", "#define N 256\n#define M 128\nvoid aten_upper_bound_cpu(float x[N],float v[M],int out[M]){for(int q=0;q<M;++q){int l=0,r=N;while(l<r){int m=(l+r)/2;if(x[m]<=v[q])l=m+1;else r=m;}out[q]=l;}}")
    add("searchsorted_cpu", s, "searchsorted_cpu_contiguous", "#define N 256\n#define M 128\nvoid aten_searchsorted_cpu(float x[N],float v[M],int out[M]){for(int q=0;q<M;++q){int l=0,r=N;while(l<r){int m=(l+r)/2;if(x[m]<v[q])l=m+1;else r=m;}out[q]=l;}}")
    s = "aten/src/ATen/native/Sorting.cpp"
    add("quick_select_cpu", s, "quick_select_template", "#define N 127\nvoid aten_quick_select_cpu(float x[N],int k,float out[1]){for(int i=0;i<=k;++i){int b=i;for(int j=i+1;j<N;++j)if(x[j]<x[b])b=j;float t=x[i];x[i]=x[b];x[b]=t;}out[0]=x[k];}")
    add("kthvalue_cpu", s, "kthvalue_out_impl_cpu", "#define R 16\n#define N 63\nvoid aten_kthvalue_cpu(float x[R][N],int k,float out[R]){for(int r=0;r<R;++r){for(int i=0;i<=k;++i){int b=i;for(int j=i+1;j<N;++j)if(x[r][j]<x[r][b])b=j;float t=x[r][i];x[r][i]=x[r][b];x[r][b]=t;}out[r]=x[r][k];}}")
    add("median_indices_cpu", s, "median_with_indices_impl", "#define R 16\n#define N 63\nvoid aten_median_indices_cpu(float x[R][N],float out[R],int idx[R]){for(int r=0;r<R;++r){int k=N/2;for(int i=0;i<=k;++i){int b=i;for(int j=i+1;j<N;++j)if(x[r][j]<x[r][b])b=j;float t=x[r][i];x[r][i]=x[r][b];x[r][b]=t;}out[r]=x[r][k];idx[r]=k;}}")
    s = "aten/src/ATen/native/SummaryOps.cpp"
    add("bincount_cpu", s, "_bincount_cpu_template", "#define N 1024\n#define B 64\nvoid aten_bincount_cpu(int x[N],float w[N],float out[B]){for(int b=0;b<B;++b)out[b]=0;for(int i=0;i<N;++i)out[x[i]]+=w[i];}")
    add("flip_tensor_transform_cpu", "aten/src/ATen/native/TensorTransformations.cpp", "flip", "#define R 32\n#define C 64\nvoid aten_flip_tensor_transform_cpu(float x[R][C],float out[R][C]){for(int i=0;i<R;++i)for(int j=0;j<C;++j)out[i][j]=x[R-1-i][C-1-j];}")


def convolutions() -> None:
    add("col2im_cpu", "aten/src/ATen/native/Col2Im.cpp", "col2im_out_cpu_template", "#define C 2\n#define H 8\n#define W 8\n#define KH 3\n#define KW 3\nvoid aten_col2im_cpu(float col[C][KH][KW][H][W],float out[C][H+2][W+2]){for(int p=0;p<C*(H+2)*(W+2);++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx)for(int y=0;y<H;++y)for(int x=0;x<W;++x)out[c][y+ky][x+kx]+=col[c][ky][kx][y][x];}")
    add("conv2d_columns_cpu", "aten/src/ATen/native/ConvolutionMM2d.cpp", "compute_columns2d", "#define C 3\n#define H 16\n#define W 16\n#define K 3\nvoid aten_conv2d_columns_cpu(float x[C][H][W],float col[C][K][K][H-2][W-2]){for(int c=0;c<C;++c)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)for(int y=0;y<H-2;++y)for(int z=0;z<W-2;++z)col[c][ky][kx][y][z]=x[c][y+ky][z+kx];}")
    s = "aten/src/ATen/native/ConvolutionMM3d.cpp"
    add("conv3d_columns_cpu", s, "compute_columns3d", "#define C 2\n#define D 8\n#define H 9\n#define W 10\n#define K 3\nvoid aten_conv3d_columns_cpu(float x[C][D][H][W],float col[C][K][K][K][D-2][H-2][W-2]){for(int c=0;c<C;++c)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)for(int z=0;z<D-2;++z)for(int y=0;y<H-2;++y)for(int q=0;q<W-2;++q)col[c][kz][ky][kx][z][y][q]=x[c][z+kz][y+ky][q+kx];}")
    add("slow_conv3d_forward_cpu", s, "slow_conv3d_forward_out_cpu", "#define C 2\n#define O 3\n#define D 8\n#define H 9\n#define W 10\n#define K 3\nvoid aten_slow_conv3d_forward_cpu(float x[C][D][H][W],float w[O][C][K][K][K],float out[O][D-2][H-2][W-2]){for(int o=0;o<O;++o)for(int z=0;z<D-2;++z)for(int y=0;y<H-2;++y)for(int q=0;q<W-2;++q){float v=0;for(int c=0;c<C;++c)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)v+=x[c][z+kz][y+ky][q+kx]*w[o][c][kz][ky][kx];out[o][z][y][q]=v;}}")
    add("slow_conv3d_backward_input_cpu", s, "slow_conv3d_backward_out_cpu_template", "#define C 2\n#define O 3\n#define D 6\n#define H 7\n#define W 8\n#define K 3\nvoid aten_slow_conv3d_backward_input_cpu(float g[O][D][H][W],float w[O][C][K][K][K],float out[C][D+2][H+2][W+2]){for(int p=0;p<C*(D+2)*(H+2)*(W+2);++p)((float*)out)[p]=0;for(int o=0;o<O;++o)for(int c=0;c<C;++c)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int x=0;x<W;++x)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)out[c][z+kz][y+ky][x+kx]+=g[o][z][y][x]*w[o][c][kz][ky][kx];}")
    add("slow_conv3d_backward_weight_cpu", s, "slow_conv3d_backward_parameters_out_cpu_template", "#define C 2\n#define O 3\n#define D 6\n#define H 7\n#define W 8\n#define K 3\nvoid aten_slow_conv3d_backward_weight_cpu(float x[C][D+2][H+2][W+2],float g[O][D][H][W],float out[O][C][K][K][K]){for(int p=0;p<O*C*K*K*K;++p)((float*)out)[p]=0;for(int o=0;o<O;++o)for(int c=0;c<C;++c)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q)out[o][c][kz][ky][kx]+=x[c][z+kz][y+ky][q+kx]*g[o][z][y][q];}")
    add("dropout_feature_noise_cpu", "aten/src/ATen/native/Dropout.cpp", "make_feature_noise", "#define B 8\n#define C 16\n#define H 8\n#define W 8\nvoid aten_dropout_feature_noise_cpu(float x[B][C][H][W],float mask[B][C],float scale,float out[B][C][H][W]){for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int y=0;y<H;++y)for(int z=0;z<W;++z)out[b][c][y][z]=x[b][c][y][z]*mask[b][c]*scale;}")


def sparse_and_shape() -> None:
    s = "aten/src/ATen/native/SparseTensorUtils.cpp"
    add("sparse_flatten_indices_cpu", s, "flatten_indices_by_dims", "#define N 512\n#define D 3\nvoid aten_sparse_flatten_indices_cpu(int idx[D][N],int size[D],int out[N]){for(int n=0;n<N;++n){int v=0;for(int d=0;d<D;++d)v=v*size[d]+idx[d][n];out[n]=v;}}")
    add("sparse_coo_to_csr_cpu", s, "coo_to_csr", "#define N 512\n#define R 64\nvoid aten_sparse_coo_to_csr_cpu(int row[N],int out[R+1]){int p=0;for(int r=0;r<=R;++r){while(p<N&&row[p]<r)++p;out[r]=p;}}")
    add("sparse_full_coo_indices_cpu", s, "full_coo_indices", "#define R 16\n#define C 32\nvoid aten_sparse_full_coo_indices_cpu(int row[R*C],int col[R*C]){int p=0;for(int r=0;r<R;++r)for(int c=0;c<C;++c){row[p]=r;col[p++]=c;}}")
    s = "aten/src/ATen/native/TensorConversions.cpp"
    add("convert_coo_to_csr_cpu", s, "convert_indices_from_coo_to_csr_cpu", "#define N 512\n#define R 64\nvoid aten_convert_coo_to_csr_cpu(int row[N],int out[R+1]){int p=0;for(int r=0;r<=R;++r){while(p<N&&row[p]<r)++p;out[r]=p;}}")
    add("convert_csr_to_coo_cpu", s, "convert_indices_from_csr_to_coo_cpu", "#define N 512\n#define R 64\nvoid aten_convert_csr_to_coo_cpu(int ptr[R+1],int col[N],int row[N]){for(int r=0;r<R;++r)for(int p=ptr[r];p<ptr[r+1];++p)row[p]=r;}")
    s = "aten/src/ATen/native/TensorShape.cpp"
    add("fast_cat_dim0_cpu", s, "fastCatOutDim0", "#define B 4\n#define N 256\nvoid aten_fast_cat_dim0_cpu(float x[B][N],float out[B*N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b*N+i]=x[b][i];}")
    add("block_diag_cpu", s, "block_diag", "#define B 4\n#define N 16\nvoid aten_block_diag_cpu(float x[B][N][N],float out[B*N][B*N]){for(int i=0;i<B*N;++i)for(int j=0;j<B*N;++j)out[i][j]=0;for(int b=0;b<B;++b)for(int i=0;i<N;++i)for(int j=0;j<N;++j)out[b*N+i][b*N+j]=x[b][i][j];}")
    add("narrow_copy_dense_cpu", s, "narrow_copy_dense_cpu_out", "#define R 32\n#define C 64\n#define S 8\n#define L 16\nvoid aten_narrow_copy_dense_cpu(float x[R][C],float out[R][L]){for(int i=0;i<R;++i)for(int j=0;j<L;++j)out[i][j]=x[i][S+j];}")
    add("repeat_tensor_shape_cpu", s, "repeat", "#define N 64\n#define R 4\nvoid aten_repeat_tensor_shape_cpu(float x[N],float out[R][N]){for(int r=0;r<R;++r)for(int i=0;i<N;++i)out[r][i]=x[i];}")
    add("split_copy_cpu", s, "split_copy_Tensor_out", "#define N 128\n#define S 4\nvoid aten_split_copy_cpu(float x[N],float out[S][N/S]){for(int s=0;s<S;++s)for(int i=0;i<N/S;++i)out[s][i]=x[s*(N/S)+i];}")
    add("copy_tensor_array_cpu", s, "copy_tensor_array_to_out", "#define B 4\n#define N 64\nvoid aten_copy_tensor_array_cpu(float x[B][N],float out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=x[b][i];}")
    add("unbind_copy_cpu", s, "unbind_copy_int_out", "#define B 4\n#define N 64\nvoid aten_unbind_copy_cpu(float x[B][N],float out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=x[b][i];}")


def misc() -> None:
    s = "aten/src/ATen/native/Itertools.cpp"
    add("triu_mask_cpu", s, "_triu_mask", "#define M 32\n#define N 32\nvoid aten_triu_mask_cpu(int mask[M][N],int diagonal){for(int i=0;i<M;++i)for(int j=0;j<N;++j)mask[i][j]=j-i>=diagonal;}")
    add("cartesian_prod_cpu", s, "cartesian_prod", "#define A 16\n#define B 12\nvoid aten_cartesian_prod_cpu(float a[A],float b[B],float out[A*B][2]){for(int i=0;i<A;++i)for(int j=0;j<B;++j){int p=i*B+j;out[p][0]=a[i];out[p][1]=b[j];}}")
    add("combinations_cpu", s, "combinations", "#define N 32\n#define K (N*(N-1)/2)\nvoid aten_combinations_cpu(float x[N],float out[K][2]){for(int i=0;i<N;++i)for(int j=i+1;j<N;++j){int p=i*(2*N-i-1)/2+(j-i-1);out[p][0]=x[i];out[p][1]=x[j];}}")
    s = "aten/src/ATen/native/SobolEngineOps.cpp"
    add("sobol_draw_cpu", s, "_sobol_engine_draw", "#define N 256\n#define D 8\nvoid aten_sobol_draw_cpu(unsigned state[D],unsigned dirs[D][32],float out[N][D]){for(int n=0;n<N;++n){int bit=0,q=n;while(q&1){++bit;q>>=1;}for(int d=0;d<D;++d){state[d]^=dirs[d][bit];out[n][d]=state[d]*(1.0f/4294967296.0f);}}}")
    add("sobol_fast_forward_cpu", s, "_sobol_engine_ff_", "#define N 256\n#define D 8\nvoid aten_sobol_fast_forward_cpu(unsigned state[D],unsigned dirs[D][32]){for(int n=0;n<N;++n){int bit=0,q=n;while(q&1){++bit;q>>=1;}for(int d=0;d<D;++d)state[d]^=dirs[d][bit];}}")
    add("sobol_scramble_cpu", s, "_sobol_engine_scramble_", "#define D 8\nvoid aten_sobol_scramble_cpu(unsigned dirs[D][32],unsigned shift[D]){for(int d=0;d<D;++d)for(int b=0;b<32;++b)dirs[d][b]^=shift[d]>>(b&7);}")
    add("sobol_initialize_cpu", s, "_sobol_engine_initialize_state_", "#define D 8\nvoid aten_sobol_initialize_cpu(unsigned dirs[D][32],unsigned state[D]){for(int d=0;d<D;++d)state[d]=dirs[d][0];}")
    add("rowwise_prune_cpu", "aten/src/ATen/native/RowwisePrune.cpp", "_rowwise_prune_helper", "#define R 64\n#define C 32\nvoid aten_rowwise_prune_cpu(float x[R][C],float threshold,int keep[R]){for(int r=0;r<R;++r){float v=0;for(int c=0;c<C;++c){float a=x[r][c];v+=a<0?-a:a;}keep[r]=v>threshold;}}")
    add("joint_scaling_cpu", "aten/src/ATen/native/ScaledBlas.cpp", "get_joint_scaling", "#define N 1024\nvoid aten_joint_scaling_cpu(float a[N],float b[N],float out[1]){float ma=0,mb=0;for(int i=0;i<N;++i){float x=a[i]<0?-a[i]:a[i],y=b[i]<0?-b[i]:b[i];if(x>ma)ma=x;if(y>mb)mb=y;}out[0]=ma*mb;}")


def remaining_major() -> None:
    s = "aten/src/ATen/native/TensorAdvancedIndexing.cpp"
    add("unsafe_index_cpu", s, "_unsafe_index", "#define N 512\nvoid aten_unsafe_index_cpu(float x[N],int idx[N],float out[N]){for(int i=0;i<N;++i)out[i]=x[idx[i]];}")
    add("index_put_impl_cpu", s, "_index_put_impl_", "#define N 512\nvoid aten_index_put_impl_cpu(float out[N],int idx[N],float value[N]){for(int i=0;i<N;++i)out[idx[i]]=value[i];}")
    add("index_reduce_impl_cpu", s, "index_reduce_func_impl", "#define N 512\n#define O 64\nvoid aten_index_reduce_impl_cpu(float x[N],int idx[N],float out[O]){for(int o=0;o<O;++o)out[o]=0;for(int i=0;i<N;++i)out[idx[i]]+=x[i];}")
    add("index_select_dim1_cpu", s, "index_select_out_cpu_dim1_", "#define R 32\n#define C 64\n#define K 16\nvoid aten_index_select_dim1_cpu(float x[R][C],int idx[K],float out[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][k]=x[r][idx[k]];}")
    add("index_select_out_cpu", s, "index_select_out_cpu_", "#define R 32\n#define C 64\n#define K 16\nvoid aten_index_select_out_cpu(float x[R][C],int idx[K],float out[K][C]){for(int k=0;k<K;++k)for(int c=0;c<C;++c)out[k][c]=x[idx[k]][c];}")
    add("masked_scatter_backward_cpu", s, "masked_scatter_backward_symint", "#define N 512\nvoid aten_masked_scatter_backward_cpu(float grad[N],int mask[N],float out[N]){int p=0;for(int i=0;i<N;++i)if(mask[i])out[p++]=grad[i];}")
    add("count_nonzero_impl_cpu", s, "count_nonzero_impl", "#define R 32\n#define C 64\nvoid aten_count_nonzero_impl_cpu(float x[R][C],int out[R]){for(int r=0;r<R;++r){int n=0;for(int c=0;c<C;++c)n+=x[r][c]!=0;out[r]=n;}}")
    add("count_nonzero_cpu", s, "count_nonzero_cpu", "#define N 2048\nvoid aten_count_nonzero_cpu(float x[N],int out[1]){int n=0;for(int i=0;i<N;++i)n+=x[i]!=0;out[0]=n;}")
    add("nonzero_out_cpu", s, "nonzero_out_cpu", "#define R 32\n#define C 64\nvoid aten_nonzero_out_cpu(float x[R][C],int row[R*C],int col[R*C],int count[1]){int p=0;for(int r=0;r<R;++r)for(int c=0;c<C;++c)if(x[r][c]!=0){row[p]=r;col[p++]=c;}count[0]=p;}")

    s = "aten/src/ATen/native/Unfold3d.cpp"
    add("unfold3d_zero_copy_cpu", s, "Unfold3dZeroPaddingCopyKernelImpl", "#define C 2\n#define D 8\n#define H 9\n#define W 10\n#define K 3\nvoid aten_unfold3d_zero_copy_cpu(float x[C][D][H][W],float out[C][K][K][K][D][H][W]){for(int c=0;c<C;++c)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx){int iz=z+kz-1,iy=y+ky-1,ix=q+kx-1;out[c][kz][ky][kx][z][y][q]=(iz>=0&&iz<D&&iy>=0&&iy<H&&ix>=0&&ix<W)?x[c][iz][iy][ix]:0;}}")
    add("unfold3d_copy_cpu", s, "Unfold3dCopyKernelImpl", "#define C 2\n#define D 6\n#define H 7\n#define W 8\n#define K 3\nvoid aten_unfold3d_copy_cpu(float x[C][D+2][H+2][W+2],float out[C][K][K][K][D][H][W]){for(int c=0;c<C;++c)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)out[c][kz][ky][kx][z][y][q]=x[c][z+kz][y+ky][q+kx];}")
    add("unfold3d_zero_acc_cpu", s, "Unfold3dZeroPaddingAccKernelImpl", "#define C 2\n#define D 8\n#define H 9\n#define W 10\n#define K 3\nvoid aten_unfold3d_zero_acc_cpu(float x[C][K][K][K][D][H][W],float out[C][D][H][W]){for(int p=0;p<C*D*H*W;++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx){int iz=z+kz-1,iy=y+ky-1,ix=q+kx-1;if(iz>=0&&iz<D&&iy>=0&&iy<H&&ix>=0&&ix<W)out[c][iz][iy][ix]+=x[c][kz][ky][kx][z][y][q];}}")
    add("unfold3d_acc_cpu", s, "Unfold3dAccKernelImpl", "#define C 2\n#define D 6\n#define H 7\n#define W 8\n#define K 3\nvoid aten_unfold3d_acc_cpu(float x[C][K][K][K][D][H][W],float out[C][D+2][H+2][W+2]){for(int p=0;p<C*(D+2)*(H+2)*(W+2);++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)out[c][z+kz][y+ky][q+kx]+=x[c][kz][ky][kx][z][y][q];}")

    s = "aten/src/ATen/native/Normalization.cpp"
    add("batch_norm_transform_cpu", s, "batch_norm_cpu_transform_input_template", "#define N 8\n#define C 16\n#define H 16\n#define W 16\nvoid aten_batch_norm_transform_cpu(float x[N][C][H][W],float mean[C],float invstd[C],float weight[C],float bias[C],float out[N][C][H][W]){for(int n=0;n<N;++n)for(int c=0;c<C;++c)for(int y=0;y<H;++y)for(int z=0;z<W;++z)out[n][c][y][z]=(x[n][c][y][z]-mean[c])*invstd[c]*weight[c]+bias[c];}")
    add("batch_norm_stats_cpu", s, "batch_norm_cpu_update_stats_template", "#define N 8\n#define C 16\n#define H 16\n#define W 16\nvoid aten_batch_norm_stats_cpu(float x[N][C][H][W],float mean[C],float var[C]){for(int c=0;c<C;++c){float m=0;for(int n=0;n<N;++n)for(int y=0;y<H;++y)for(int z=0;z<W;++z)m+=x[n][c][y][z];m/=N*H*W;float v=0;for(int n=0;n<N;++n)for(int y=0;y<H;++y)for(int z=0;z<W;++z){float d=x[n][c][y][z]-m;v+=d*d;}mean[c]=m;var[c]=v/(N*H*W);}}")
    add("batch_norm_backward_template_cpu", s, "batch_norm_backward_cpu_template", "#define N 8\n#define C 16\n#define H 16\n#define W 16\nvoid aten_batch_norm_backward_template_cpu(float grad[N][C][H][W],float x[N][C][H][W],float mean[C],float invstd[C],float out[N][C][H][W]){for(int c=0;c<C;++c){float sg=0,sgx=0;for(int n=0;n<N;++n)for(int y=0;y<H;++y)for(int z=0;z<W;++z){sg+=grad[n][c][y][z];sgx+=grad[n][c][y][z]*(x[n][c][y][z]-mean[c]);}for(int n=0;n<N;++n)for(int y=0;y<H;++y)for(int z=0;z<W;++z)out[n][c][y][z]=invstd[c]*(grad[n][c][y][z]-sg/(N*H*W)-(x[n][c][y][z]-mean[c])*invstd[c]*invstd[c]*sgx/(N*H*W));}}")
    add("batch_norm_cpu_entry", s, "batch_norm_cpu", "#define N 2048\nvoid aten_batch_norm_cpu_entry(float x[N],float scale,float bias,float out[N]){for(int i=0;i<N;++i)out[i]=x[i]*scale+bias;}")

    s = "aten/src/ATen/native/Linear.cpp"
    add("flatten_nd_linear_cpu", s, "_flatten_nd_linear", "#define B 16\n#define M 32\n#define K 64\n#define N 48\nvoid aten_flatten_nd_linear_cpu(float x[B][M][K],float w[K][N],float out[B][M][N]){for(int b=0;b<B;++b)for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=x[b][i][k]*w[k][j];out[b][i][j]=v;}}")
    add("sumproduct_pair_cpu", s, "sumproduct_pair", "#define B 8\n#define M 16\n#define K 32\n#define N 24\nvoid aten_sumproduct_pair_cpu(float a[B][M][K],float b[B][K][N],float out[B][M][N]){for(int q=0;q<B;++q)for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=a[q][i][k]*b[q][k][j];out[q][i][j]=v;}}")
    add("trilinear_cpu", s, "_trilinear", "#define B 8\n#define I 16\n#define J 20\n#define K 24\nvoid aten_trilinear_cpu(float a[B][I],float w[I][J][K],float b[B][J],float out[B][K]){for(int n=0;n<B;++n)for(int k=0;k<K;++k){float v=0;for(int i=0;i<I;++i)for(int j=0;j<J;++j)v+=a[n][i]*w[i][j][k]*b[n][j];out[n][k]=v;}}")
    add("bilinear_cpu", s, "bilinear", "#define B 8\n#define I 16\n#define J 20\n#define O 24\nvoid aten_bilinear_cpu(float a[B][I],float w[O][I][J],float b[B][J],float out[B][O]){for(int n=0;n<B;++n)for(int o=0;o<O;++o){float v=0;for(int i=0;i<I;++i)for(int j=0;j<J;++j)v+=a[n][i]*w[o][i][j]*b[n][j];out[n][o]=v;}}")

    s = "aten/src/ATen/native/LinearAlgebra.cpp"
    add("vector_norm_out_cpu", s, "linalg_vector_norm_out", "#define R 32\n#define C 64\nextern float sqrtf(float);void aten_vector_norm_out_cpu(float x[R][C],float out[R]){for(int r=0;r<R;++r){float v=0;for(int c=0;c<C;++c)v+=x[r][c]*x[r][c];out[r]=sqrtf(v);}}")
    add("linalg_powsum_cpu", s, "linalg__powsum", "#define R 32\n#define C 64\nextern float powf(float,float);void aten_linalg_powsum_cpu(float x[R][C],float p,float out[R]){for(int r=0;r<R;++r){float v=0;for(int c=0;c<C;++c)v+=powf(x[r][c]<0?-x[r][c]:x[r][c],p);out[r]=v;}}")
    add("kron_impl_cpu", s, "KronImpl", "#define A 16\n#define B 12\n#define C 8\n#define D 10\nvoid aten_kron_impl_cpu(float x[A][B],float y[C][D],float out[A*C][B*D]){for(int a=0;a<A;++a)for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int d=0;d<D;++d)out[a*C+c][b*D+d]=x[a][b]*y[c][d];}")
    add("kron_out_cpu", s, "kron_out", "#define A 16\n#define B 12\n#define C 8\n#define D 10\nvoid aten_kron_out_cpu(float x[A][B],float y[C][D],float out[A*C][B*D]){for(int a=0;a<A;++a)for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int d=0;d<D;++d)out[a*C+c][b*D+d]=x[a][b]*y[c][d];}")
    add("int_mm_out_cpu", s, "_int_mm_out_cpu", "#define M 32\n#define N 48\n#define K 64\nvoid aten_int_mm_out_cpu(signed char a[M][K],signed char b[K][N],int out[M][N]){for(int i=0;i<M;++i)for(int j=0;j<N;++j){int v=0;for(int k=0;k<K;++k)v+=(int)a[i][k]*(int)b[k][j];out[i][j]=v;}}")


def sparse_math() -> None:
    s = "aten/src/ATen/native/sparse/SparseTensorMath.cpp"
    add("sparse_norm_cpu", s, "norm_sparse", "#define N 1024\nextern float sqrtf(float);void aten_sparse_norm_cpu(float value[N],float out[1]){float v=0;for(int i=0;i<N;++i)v+=value[i]*value[i];out[0]=sqrtf(v);}")
    add("sparse_add_values_cpu", s, "add_out_sparse_contiguous", "#define N 1024\nvoid aten_sparse_add_values_cpu(float a[N],float b[N],float alpha,float out[N]){for(int i=0;i<N;++i)out[i]=a[i]+alpha*b[i];}")
    add("dense_sparse_add_cpu", s, "add_out_dense_sparse_cpu", "#define R 64\n#define C 64\n#define N 512\nvoid aten_dense_sparse_add_cpu(float dense[R][C],int row[N],int col[N],float value[N],float out[R][C]){for(int r=0;r<R;++r)for(int c=0;c<C;++c)out[r][c]=dense[r][c];for(int i=0;i<N;++i)out[row[i]][col[i]]+=value[i];}")
    add("sparse_dense_intersection_cpu", s, "intersection_binary_op_sparse_dense_out", "#define R 64\n#define C 64\n#define N 512\nvoid aten_sparse_dense_intersection_cpu(float dense[R][C],int row[N],int col[N],float value[N],float out[N]){for(int i=0;i<N;++i)out[i]=value[i]*dense[row[i]][col[i]];}")
    add("sparse_mul_cpu", s, "mul_out_sparse_cpu", "#define N 1024\nvoid aten_sparse_mul_cpu(float a[N],float b[N],float out[N]){for(int i=0;i<N;++i)out[i]=a[i]*b[i];}")
    add("sparse_addmm_cpu", s, "s_addmm_out_sparse_dense_worker", "#define R 64\n#define C 48\n#define N 512\nvoid aten_sparse_addmm_cpu(int row[N],int col[N],float value[N],float dense[64][C],float out[R][C]){for(int r=0;r<R;++r)for(int c=0;c<C;++c)out[r][c]=0;for(int p=0;p<N;++p)for(int c=0;c<C;++c)out[row[p]][c]+=value[p]*dense[col[p]][c];}")
    add("hspmm_cpu", s, "hspmm_out_sparse_cpu", "#define R 64\n#define C 48\n#define N 512\nvoid aten_hspmm_cpu(int row[N],int col[N],float value[N],float dense[64][C],float out[R][C]){for(int r=0;r<R;++r)for(int c=0;c<C;++c)out[r][c]=0;for(int p=0;p<N;++p)for(int c=0;c<C;++c)out[row[p]][c]+=value[p]*dense[col[p]][c];}")
    add("sparse_sum_cpu", s, "_sparse_sum", "#define N 1024\nvoid aten_sparse_sum_cpu(float x[N],float out[1]){float v=0;for(int i=0;i<N;++i)v+=x[i];out[0]=v;}")
    add("sparse_sum_backward_cpu", s, "_sparse_sum_backward_cpu", "#define N 1024\nvoid aten_sparse_sum_backward_cpu(float grad,float out[N]){for(int i=0;i<N;++i)out[i]=grad;}")
    add("binary_search_strided_rightmost_cpu", s, "binary_search_strided_rightmost", "#define N 512\n#define Q 128\nvoid aten_binary_search_strided_rightmost_cpu(int x[N],int q[Q],int out[Q]){for(int i=0;i<Q;++i){int l=0,r=N;while(l<r){int m=(l+r)/2;if(x[m]<=q[i])l=m+1;else r=m;}out[i]=l-1;}}")
    add("sparse_bmm_cpu", s, "bmm_out_sparse_cpu", "#define B 4\n#define M 32\n#define K 40\n#define N 24\nvoid aten_sparse_bmm_cpu(float a[B][M][K],float b[B][K][N],float out[B][M][N]){for(int q=0;q<B;++q)for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=a[q][i][k]*b[q][k][j];out[q][i][j]=v;}}")
    s = "aten/src/ATen/native/sparse/SparseCsrTensorMath.cpp"
    add("sparse_csr_addmm_cpu", s, "addmm_out_sparse_csr_native_cpu", "#define R 64\n#define K 64\n#define C 48\n#define N 512\nvoid aten_sparse_csr_addmm_cpu(int ptr[R+1],int col[N],float val[N],float b[K][C],float out[R][C]){for(int r=0;r<R;++r)for(int c=0;c<C;++c){float v=0;for(int p=ptr[r];p<ptr[r+1];++p)v+=val[p]*b[col[p]][c];out[r][c]=v;}}")
    add("sparse_csr_add_dense_cpu", s, "add_out_dense_sparse_compressed_cpu", "#define R 64\n#define C 64\n#define N 512\nvoid aten_sparse_csr_add_dense_cpu(float out[R][C],int ptr[R+1],int col[N],float val[N]){for(int r=0;r<R;++r)for(int p=ptr[r];p<ptr[r+1];++p)out[r][col[p]]+=val[p];}")
    add("sparse_csr_reduce_dim0_cpu", s, "reduce_sparse_csr_dim0_cpu_template", "#define R 64\n#define C 64\n#define N 512\nvoid aten_sparse_csr_reduce_dim0_cpu(int ptr[R+1],int col[N],float val[N],float out[C]){for(int c=0;c<C;++c)out[c]=0;for(int r=0;r<R;++r)for(int p=ptr[r];p<ptr[r+1];++p)out[col[p]]+=val[p];}")
    add("sparse_csr_reduce_dim1_cpu", s, "reduce_sparse_csr_dim1_cpu_template", "#define R 64\n#define N 512\nvoid aten_sparse_csr_reduce_dim1_cpu(int ptr[R+1],float val[N],float out[R]){for(int r=0;r<R;++r){float v=0;for(int p=ptr[r];p<ptr[r+1];++p)v+=val[p];out[r]=v;}}")
    add("sparse_csr_reduce_all_cpu", s, "reduce_sparse_csr_dim01_cpu_template", "#define N 512\nvoid aten_sparse_csr_reduce_all_cpu(float val[N],float out[1]){float v=0;for(int p=0;p<N;++p)v+=val[p];out[0]=v;}")


def final_families() -> None:
    s = "aten/src/ATen/native/Distributions.cpp"
    add("sample_poisson_transform_cpu", s, "sample_poisson", "#define N 1024\nextern float expf(float);void aten_sample_poisson_transform_cpu(float lambda[N],float uniform[N][32],int out[N]){for(int i=0;i<N;++i){float p=expf(-lambda[i]),s=p;int k=0;while(k<31&&uniform[i][k]>s){++k;p*=lambda[i]/k;s+=p;}out[i]=k;}}")
    add("standard_gamma_grad_cpu", s, "_standard_gamma_grad_cpu", "#define N 1024\nextern float logf(float);void aten_standard_gamma_grad_cpu(float a[N],float x[N],float out[N]){for(int i=0;i<N;++i)out[i]=(x[i]-a[i])/(a[i]+.001f)+logf(x[i]+.001f);}")
    add("dirichlet_grad_cpu", s, "_dirichlet_grad_cpu", "#define N 1024\nextern float logf(float);void aten_dirichlet_grad_cpu(float x[N],float alpha[N],float total[N],float out[N]){for(int i=0;i<N;++i)out[i]=x[i]*(logf(x[i]+.001f)-alpha[i]/total[i]);}")
    add("binomial_transform_cpu", s, "_s_binomial_cpu", "#define N 1024\n#define T 32\nvoid aten_binomial_transform_cpu(int count[N],float p[N],float uniform[N][T],int out[N]){for(int i=0;i<N;++i){int k=0;for(int t=0;t<T&&t<count[i];++t)k+=uniform[i][t]<p[i];out[i]=k;}}")
    add("poisson_transform_cpu", s, "_s_poisson_cpu", "#define N 1024\n#define T 64\nextern float expf(float);void aten_poisson_transform_cpu(float lambda[N],float uniform[N][T],int out[N]){for(int i=0;i<N;++i){float q=1;int k=0;while(k<T&&q>expf(-lambda[i]))q*=uniform[i][k++];out[i]=k-1;}}")
    add("gamma_transform_cpu", s, "_s_gamma_cpu", "#define N 1024\nextern float sqrtf(float);void aten_gamma_transform_cpu(float alpha[N],float normal[N],float uniform[N],float out[N]){for(int i=0;i<N;++i){float d=alpha[i]-.3333333f,c=1/sqrtf(9*d),v=1+c*normal[i];out[i]=d*v*v*v;}}")
    add("dirichlet_transform_cpu", s, "_s_dirichlet_cpu", "#define R 64\n#define C 16\nvoid aten_dirichlet_transform_cpu(float gamma[R][C],float out[R][C]){for(int r=0;r<R;++r){float s=0;for(int c=0;c<C;++c)s+=gamma[r][c];for(int c=0;c<C;++c)out[r][c]=gamma[r][c]/s;}}")

    s = "aten/src/ATen/native/BlasKernel.cpp"
    add("fp16_gemv_f16arith_cpu", s, "fp16_gemv_notrans_fp16_arith", "#define M 64\n#define K 96\nvoid aten_fp16_gemv_f16arith_cpu(float a[M][K],float x[K],float out[M]){for(int i=0;i<M;++i){float v=0;for(int k=0;k<K;++k)v+=a[i][k]*x[k];out[i]=v;}}")
    add("fp16_gemv_f32arith_cpu", s, "fp16_gemv_notrans_fp32_arith", "#define M 64\n#define K 96\nvoid aten_fp16_gemv_f32arith_cpu(float a[M][K],float x[K],float out[M]){for(int i=0;i<M;++i){float v=0;for(int k=0;k<K;++k)v+=a[i][k]*x[k];out[i]=v;}}")
    add("fp16_gemv_notrans_cpu", s, "fp16_gemv_notrans", "#define M 64\n#define K 96\nvoid aten_fp16_gemv_notrans_cpu(float a[M][K],float x[K],float out[M]){for(int i=0;i<M;++i){float v=0;for(int k=0;k<K;++k)v+=a[i][k]*x[k];out[i]=v;}}")
    add("blas_gemv_generic_cpu", s, "gemv", "#define M 64\n#define K 96\nvoid aten_blas_gemv_generic_cpu(float a[M][K],float x[K],float out[M]){for(int i=0;i<M;++i){float v=0;for(int k=0;k<K;++k)v+=a[i][k]*x[k];out[i]=v;}}")
    add("blas_dot_naive_cpu", s, "dot_naive", "#define N 2048\nvoid aten_blas_dot_naive_cpu(float a[N],float b[N],float out[1]){float v=0;for(int i=0;i<N;++i)v+=a[i]*b[i];out[0]=v;}")

    s = "aten/src/ATen/native/EmbeddingBag.cpp"
    add("embedding_bag_max_cpu", s, "embedding_bag_cpu_max_out", "#define B 32\n#define L 16\n#define E 1024\n#define D 64\nvoid aten_embedding_bag_max_cpu(float table[E][D],int index[B][L],float out[B][D]){for(int b=0;b<B;++b)for(int d=0;d<D;++d){float v=table[index[b][0]][d];for(int l=1;l<L;++l)if(table[index[b][l]][d]>v)v=table[index[b][l]][d];out[b][d]=v;}}")
    add("embedding_bag_backward_max_cpu", s, "_embedding_bag_dense_backward_cpu_max", "#define B 32\n#define D 64\n#define E 1024\nvoid aten_embedding_bag_backward_max_cpu(float grad[B][D],int maxidx[B][D],float out[E][D]){for(int p=0;p<E*D;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int d=0;d<D;++d)out[maxidx[b][d]][d]+=grad[b][d];}")
    add("embedding_bag_counts_cpu", s, "compute_counts", "#define N 512\n#define E 1024\nvoid aten_embedding_bag_counts_cpu(int index[N],int out[E]){for(int e=0;e<E;++e)out[e]=0;for(int i=0;i<N;++i)out[index[i]]++;}")
    add("embedding_bag_counts_uniq_cpu", s, "compute_counts_uniq", "#define N 512\nvoid aten_embedding_bag_counts_uniq_cpu(int index[N],int out[N]){for(int i=0;i<N;++i){int n=0;for(int j=0;j<N;++j)n+=index[j]==index[i];out[i]=n;}}")
    add("embedding_bag_backward_sum_cpu", s, "_embedding_bag_dense_backward_cpu_sum_mean", "#define B 32\n#define L 16\n#define E 1024\n#define D 64\nvoid aten_embedding_bag_backward_sum_cpu(float grad[B][D],int index[B][L],float out[E][D]){for(int p=0;p<E*D;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int l=0;l<L;++l)for(int d=0;d<D;++d)out[index[b][l]][d]+=grad[b][d];}")
    add("embedding_bag_per_sample_backward_cpu", s, "_embedding_bag_per_sample_weights_backward_cpu_template", "#define B 32\n#define L 16\n#define E 1024\n#define D 64\nvoid aten_embedding_bag_per_sample_backward_cpu(float grad[B][D],float table[E][D],int index[B][L],float out[B][L]){for(int b=0;b<B;++b)for(int l=0;l<L;++l){float v=0;for(int d=0;d<D;++d)v+=grad[b][d]*table[index[b][l]][d];out[b][l]=v;}}")

    s = "aten/src/ATen/native/Unique.cpp"
    add("unique_bool_cpu", s, "unique_cpu_bool_template", "#define N 1024\nvoid aten_unique_bool_cpu(int x[N],int values[2],int count[2]){count[0]=count[1]=0;for(int i=0;i<N;++i)count[x[i]!=0]++;values[0]=0;values[1]=1;}")
    add("unique_sorted_cpu", s, "unique_cpu_sorted_template", "#define N 1024\nvoid aten_unique_sorted_cpu(int x[N],int out[N],int count[1]){for(int i=0;i<N;++i)for(int j=i+1;j<N;++j)if(x[j]<x[i]){int t=x[i];x[i]=x[j];x[j]=t;}int p=0;for(int i=0;i<N;++i)if(i==0||x[i]!=x[i-1])out[p++]=x[i];count[0]=p;}")
    add("unique_consecutive_cpu", s, "unique_consecutive_cpu_template", "#define N 1024\nvoid aten_unique_consecutive_cpu(int x[N],int out[N],int count[1]){int p=0;for(int i=0;i<N;++i)if(i==0||x[i]!=x[i-1])out[p++]=x[i];count[0]=p;}")
    add("unique_dim_impl_cpu", s, "_unique_dim_cpu_impl", "#define R 128\n#define C 16\nvoid aten_unique_dim_impl_cpu(float x[R][C],int keep[R]){for(int r=0;r<R;++r){int unique=1;for(int q=0;q<r;++q){int same=1;for(int c=0;c<C;++c)same&=x[r][c]==x[q][c];unique&=!same;}keep[r]=unique;}}")
    add("unique_dim_template_cpu", s, "_unique_dim_cpu_template", "#define R 128\n#define C 16\nvoid aten_unique_dim_template_cpu(float x[R][C],int keep[R]){for(int r=0;r<R;++r){int unique=1;for(int q=0;q<r;++q){int same=1;for(int c=0;c<C;++c)same&=x[r][c]==x[q][c];unique&=!same;}keep[r]=unique;}}")

    s = "aten/src/ATen/native/nested/NestedTensorMath.cpp"
    add("nested_pad_cpu", s, "pad_tensor_to_shape", "#define B 8\n#define N 64\n#define P 80\nvoid aten_nested_pad_cpu(float x[B][N],float out[B][P]){for(int b=0;b<B;++b)for(int i=0;i<P;++i)out[b][i]=i<N?x[b][i]:0;}")
    add("nested_from_padded_cpu", s, "nested_from_padded_generic", "#define B 8\n#define P 80\nvoid aten_nested_from_padded_cpu(float x[B][P],int len[B],float out[B][P]){for(int b=0;b<B;++b)for(int i=0;i<P;++i)out[b][i]=i<len[b]?x[b][i]:0;}")
    add("nested_to_padded_cpu", s, "NestedTensor_to_padded_tensor_generic", "#define B 8\n#define P 80\nvoid aten_nested_to_padded_cpu(float x[B][P],int len[B],float pad,float out[B][P]){for(int b=0;b<B;++b)for(int i=0;i<P;++i)out[b][i]=i<len[b]?x[b][i]:pad;}")
    add("nested_sum_dim_cpu", s, "NestedTensor_sum_dim_CPU", "#define B 8\n#define N 64\nvoid aten_nested_sum_dim_cpu(float x[B][N],int len[B],float out[B]){for(int b=0;b<B;++b){float v=0;for(int i=0;i<len[b];++i)v+=x[b][i];out[b]=v;}}")
    add("nested_select_cpu", s, "select_nested", "#define B 8\n#define N 64\nvoid aten_nested_select_cpu(float x[B][N],int idx[B],float out[B]){for(int b=0;b<B;++b)out[b]=x[b][idx[b]];}")
    add("nested_softmax_cpu", s, "softmax_nested", "#define B 8\n#define N 64\nextern float expf(float);void aten_nested_softmax_cpu(float x[B][N],int len[B],float out[B][N]){for(int b=0;b<B;++b){float s=0;for(int i=0;i<len[b];++i){out[b][i]=expf(x[b][i]);s+=out[b][i];}for(int i=0;i<len[b];++i)out[b][i]/=s;}}")
    add("nested_all_cpu", s, "NestedTensor_all", "#define B 8\n#define N 64\nvoid aten_nested_all_cpu(int x[B][N],int len[B],int out[B]){for(int b=0;b<B;++b){int v=1;for(int i=0;i<len[b];++i)v&=x[b][i]!=0;out[b]=v;}}")
    add("nested_squeeze_cpu", s, "squeeze_dim_nested", "#define B 8\n#define N 64\nvoid aten_nested_squeeze_cpu(float x[B][1][N],float out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=x[b][0][i];}")

    s = "aten/src/ATen/native/nested/NestedTensorTransformerFunctions.cpp"
    add("nested_softmax_dropout_cpu", s, "NestedTensor_softmax_dropout", "#define B 8\n#define N 64\nextern float expf(float);void aten_nested_softmax_dropout_cpu(float x[B][N],float mask[B][N],float out[B][N]){for(int b=0;b<B;++b){float s=0;for(int i=0;i<N;++i){out[b][i]=expf(x[b][i])*mask[b][i];s+=out[b][i];}for(int i=0;i<N;++i)out[b][i]/=s;}}")
    add("nested_batch_offsets_cpu", s, "NestedTensor_batch_offsets_from_size_tensor", "#define B 64\nvoid aten_nested_batch_offsets_cpu(int size[B],int out[B+1]){out[0]=0;for(int b=0;b<B;++b)out[b+1]=out[b]+size[b];}")
    add("nested_to_mask_cpu", s, "NestedTensor_to_mask", "#define B 8\n#define N 64\nvoid aten_nested_to_mask_cpu(int len[B],int out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=i<len[b];}")
    add("jagged_to_padded_cpu", s, "_jagged_to_padded_dense_forward_cpu", "#define B 8\n#define N 64\nvoid aten_jagged_to_padded_cpu(float x[B*N],int off[B+1],float out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=off[b]+i<off[b+1]?x[off[b]+i]:0;}")
    add("padded_to_jagged_cpu", s, "_padded_dense_to_jagged_forward_cpu", "#define B 8\n#define N 64\nvoid aten_padded_to_jagged_cpu(float x[B][N],int off[B+1],float out[B*N]){for(int b=0;b<B;++b)for(int i=0;off[b]+i<off[b+1];++i)out[off[b]+i]=x[b][i];}")

    s = "aten/src/ATen/native/sparse/SoftMax.cpp"
    add("sparse_softmax_offsets_cpu", s, "get_offsets", "#define N 512\n#define R 64\nvoid aten_sparse_softmax_offsets_cpu(int row[N],int out[R+1]){int p=0;for(int r=0;r<=R;++r){while(p<N&&row[p]<r)++p;out[r]=p;}}")
    add("sparse_softmax_pools_cpu", s, "get_pools", "#define R 64\nvoid aten_sparse_softmax_pools_cpu(int off[R+1],int size[R]){for(int r=0;r<R;++r)size[r]=off[r+1]-off[r];}")
    add("sparse_coo_softmax_cpu", s, "cpu_sparse_coo_softmax", "#define R 64\n#define K 8\nextern float expf(float);void aten_sparse_coo_softmax_cpu(float x[R][K],float out[R][K]){for(int r=0;r<R;++r){float s=0;for(int k=0;k<K;++k){out[r][k]=expf(x[r][k]);s+=out[r][k];}for(int k=0;k<K;++k)out[r][k]/=s;}}")
    add("sparse_coo_softmax_backward_cpu", s, "cpu_sparse_coo_softmax_backward", "#define R 64\n#define K 8\nvoid aten_sparse_coo_softmax_backward_cpu(float grad[R][K],float y[R][K],float out[R][K]){for(int r=0;r<R;++r){float s=0;for(int k=0;k<K;++k)s+=grad[r][k]*y[r][k];for(int k=0;k<K;++k)out[r][k]=y[r][k]*(grad[r][k]-s);}}")


def last_bodies() -> None:
    s = "aten/src/ATen/native/AdaptiveMaxPooling3d.cpp"
    add("adaptive_max_pool3d_legacy_cpu", s, "adaptive_max_pool3d_out_frame", "#define C 2\n#define ID 8\n#define IH 9\n#define IW 10\n#define OD 3\n#define OH 4\n#define OW 5\nvoid aten_adaptive_max_pool3d_legacy_cpu(float x[C][ID][IH][IW],float out[C][OD][OH][OW],int idx[C][OD][OH][OW]){for(int c=0;c<C;++c)for(int z=0;z<OD;++z)for(int y=0;y<OH;++y)for(int q=0;q<OW;++q){int zs=z*ID/OD,ze=((z+1)*ID+OD-1)/OD,ys=y*IH/OH,ye=((y+1)*IH+OH-1)/OH,xs=q*IW/OW,xe=((q+1)*IW+OW-1)/OW,b=(zs*IH+ys)*IW+xs;float v=x[c][zs][ys][xs];for(int iz=zs;iz<ze;++iz)for(int iy=ys;iy<ye;++iy)for(int ix=xs;ix<xe;++ix)if(x[c][iz][iy][ix]>v){v=x[c][iz][iy][ix];b=(iz*IH+iy)*IW+ix;}out[c][z][y][q]=v;idx[c][z][y][q]=b;}}")
    add("adaptive_max_pool3d_legacy_backward_cpu", s, "adaptive_max_pool3d_backward_out_frame", "#define C 2\n#define ID 8\n#define IH 9\n#define IW 10\n#define OD 3\n#define OH 4\n#define OW 5\nvoid aten_adaptive_max_pool3d_legacy_backward_cpu(float g[C][OD][OH][OW],int idx[C][OD][OH][OW],float out[C][ID][IH][IW]){for(int p=0;p<C*ID*IH*IW;++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int z=0;z<OD;++z)for(int y=0;y<OH;++y)for(int x=0;x<OW;++x){int q=idx[c][z][y][x];out[c][q/(IH*IW)][(q/IW)%IH][q%IW]+=g[c][z][y][x];}}")
    s = "aten/src/ATen/native/BatchLinearAlgebraKernel.cpp"
    add("reflect_conj_tri_cpu", s, "apply_reflect_conj_tri_single", "#define N 64\nvoid aten_reflect_conj_tri_cpu(float re[N][N],float im[N][N]){for(int i=0;i<N;++i)for(int j=i+1;j<N;++j){re[j][i]=re[i][j];im[j][i]=-im[i][j];}}")
    add("eig_complex_vectors_cpu", s, "linalg_eig_make_complex_eigenvectors_cpu_impl", "#define N 64\nvoid aten_eig_complex_vectors_cpu(float real[N][N],float imag[N],float out_re[N][N],float out_im[N][N]){for(int i=0;i<N;++i)for(int j=0;j<N;++j){out_re[i][j]=real[i][j];out_im[i][j]=imag[j]==0?0:(j+1<N?real[i][j+1]:0);}}")
    add("unpack_pivots_cpu", s, "unpack_pivots_cpu_kernel", "#define N 128\nvoid aten_unpack_pivots_cpu(int piv[N],int perm[N]){for(int i=0;i<N;++i)perm[i]=i;for(int i=0;i<N;++i){int j=piv[i]-1,t=perm[i];perm[i]=perm[j];perm[j]=t;}}")
    s = "aten/src/ATen/native/ConvolutionTBC.cpp"
    add("conv_tbc_cpu", s, "conv_tbc", "#define T 32\n#define B 8\n#define I 16\n#define O 24\n#define K 3\nvoid aten_conv_tbc_cpu(float x[T][B][I],float w[K][I][O],float out[T-K+1][B][O]){for(int t=0;t<T-K+1;++t)for(int b=0;b<B;++b)for(int o=0;o<O;++o){float v=0;for(int k=0;k<K;++k)for(int i=0;i<I;++i)v+=x[t+k][b][i]*w[k][i][o];out[t][b][o]=v;}}")
    add("conv_tbc_backward_cpu", s, "conv_tbc_backward", "#define T 30\n#define B 8\n#define I 16\n#define O 24\n#define K 3\nvoid aten_conv_tbc_backward_cpu(float g[T][B][O],float w[K][I][O],float out[T+K-1][B][I]){for(int p=0;p<(T+K-1)*B*I;++p)((float*)out)[p]=0;for(int t=0;t<T;++t)for(int b=0;b<B;++b)for(int o=0;o<O;++o)for(int k=0;k<K;++k)for(int i=0;i<I;++i)out[t+k][b][i]+=g[t][b][o]*w[k][i][o];}")
    s = "aten/src/ATen/native/NaiveConvolutionTranspose3d.cpp"
    add("conv_transpose3d_cpu", s, "slow_conv_transpose3d_out_cpu_template", "#define C 2\n#define O 3\n#define D 6\n#define H 7\n#define W 8\n#define K 3\nvoid aten_conv_transpose3d_cpu(float x[C][D][H][W],float w[C][O][K][K][K],float out[O][D+2][H+2][W+2]){for(int p=0;p<O*(D+2)*(H+2)*(W+2);++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int o=0;o<O;++o)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)out[o][z+kz][y+ky][q+kx]+=x[c][z][y][q]*w[c][o][kz][ky][kx];}")
    add("conv_transpose3d_backward_cpu", s, "slow_conv_transpose3d_backward_out_cpu_template", "#define C 2\n#define O 3\n#define D 6\n#define H 7\n#define W 8\n#define K 3\nvoid aten_conv_transpose3d_backward_cpu(float g[O][D+2][H+2][W+2],float w[C][O][K][K][K],float out[C][D][H][W]){for(int c=0;c<C;++c)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q){float v=0;for(int o=0;o<O;++o)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)v+=g[o][z+kz][y+ky][q+kx]*w[c][o][kz][ky][kx];out[c][z][y][q]=v;}}")
    add("conv_transpose3d_grad_weight_cpu", s, "slow_conv_transpose3d_acc_grad_parameters_cpu", "#define C 2\n#define O 3\n#define D 6\n#define H 7\n#define W 8\n#define K 3\nvoid aten_conv_transpose3d_grad_weight_cpu(float x[C][D][H][W],float g[O][D+2][H+2][W+2],float out[C][O][K][K][K]){for(int p=0;p<C*O*K*K*K;++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int o=0;o<O;++o)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q)out[c][o][kz][ky][kx]+=x[c][z][y][q]*g[o][z+kz][y+ky][q+kx];}")
    add("dilated_convolution_cpu", "aten/src/ATen/native/NaiveDilatedConvolution.cpp", "slow_conv_dilated_all_cpu_template", "#define C 2\n#define O 3\n#define H 16\n#define W 16\n#define K 3\n#define D 2\nvoid aten_dilated_convolution_cpu(float x[C][H][W],float w[O][C][K][K],float out[O][H-2*D][W-2*D]){for(int o=0;o<O;++o)for(int y=0;y<H-2*D;++y)for(int q=0;q<W-2*D;++q){float v=0;for(int c=0;c<C;++c)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)v+=x[c][y+ky*D][q+kx*D]*w[o][c][ky][kx];out[o][y][q]=v;}}")
    s = "aten/src/ATen/native/QuantizedLinear.cpp"
    add("quant_col_offsets_cpu", s, "CalcColOffsetsTranspose", "#define K 64\n#define N 48\nvoid aten_quant_col_offsets_cpu(signed char w[K][N],int zero,int out[N]){for(int n=0;n<N;++n){int v=0;for(int k=0;k<K;++k)v+=(int)w[k][n];out[n]=v-zero*K;}}")
    add("quant_saturation_cpu", s, "HandleWeightsSaturation", "#define N 4096\nvoid aten_quant_saturation_cpu(int x[N],signed char out[N]){for(int i=0;i<N;++i){int v=x[i];if(v<-128)v=-128;if(v>127)v=127;out[i]=(signed char)v;}}")
    add("compressed_block_convert_cpu", "aten/src/ATen/native/TensorConversions.cpp", "_compressed_to_block_compressed_cpu_kernel", "#define R 64\n#define C 64\n#define BR 4\n#define BC 4\nvoid aten_compressed_block_convert_cpu(float x[R][C],float out[R/BR][C/BC][BR][BC]){for(int r=0;r<R;++r)for(int c=0;c<C;++c)out[r/BR][c/BC][r%BR][c%BC]=x[r][c];}")
    s = "aten/src/ATen/native/TensorShape.cpp"
    add("cat_sparse_cpu", s, "cat_sparse_impl", "#define B 4\n#define N 256\nvoid aten_cat_sparse_cpu(int idx[B][N],float val[B][N],int out_idx[B*N],float out_val[B*N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i){out_idx[b*N+i]=idx[b][i];out_val[b*N+i]=val[b][i];}}")
    add("permute_sparse_coo_cpu", s, "permute_sparse_coo", "#define D 3\n#define N 512\nvoid aten_permute_sparse_coo_cpu(int idx[D][N],int perm[D],int out[D][N]){for(int d=0;d<D;++d)for(int n=0;n<N;++n)out[d][n]=idx[perm[d]][n];}")
    add("index_select_sparse_cpu", s, "index_select_sparse_cpu", "#define N 512\n#define K 128\nvoid aten_index_select_sparse_cpu(float val[N],int idx[K],float out[K]){for(int k=0;k<K;++k)out[k]=val[idx[k]];}")
    add("max_unpool_backward_cpu", "aten/src/ATen/native/cpu/MaxUnpoolKernel.cpp", "cpu_max_unpool_backward", "#define N 512\n#define O 2048\nvoid aten_max_unpool_backward_cpu(float grad[O],int index[N],float out[N]){for(int i=0;i<N;++i)out[i]=grad[index[i]];}")
    s = "aten/src/ATen/native/nested/NestedTensorBackward.cpp"
    add("nested_softmax_backward_cpu", s, "nested_softmax_backward", "#define B 8\n#define N 64\nvoid aten_nested_softmax_backward_cpu(float grad[B][N],float y[B][N],float out[B][N]){for(int b=0;b<B;++b){float s=0;for(int i=0;i<N;++i)s+=grad[b][i]*y[b][i];for(int i=0;i<N;++i)out[b][i]=y[b][i]*(grad[b][i]-s);}}")
    add("nested_sum_backward_cpu", s, "_nested_sum_backward_cpu", "#define B 8\n#define N 64\nvoid aten_nested_sum_backward_cpu(float grad[B],float out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=grad[b];}")
    add("nested_clone_cpu", "aten/src/ATen/native/nested/NestedTensorFactories.cpp", "clone_nested", "#define B 8\n#define N 64\nvoid aten_nested_clone_cpu(float x[B][N],float out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=x[b][i];}")
    s = "aten/src/ATen/native/nested/NestedTensorMatmul.cpp"
    add("nested_bmm_cpu", s, "bmm_nested", "#define B 8\n#define M 16\n#define K 24\n#define N 20\nvoid aten_nested_bmm_cpu(float a[B][M][K],float b[B][K][N],float out[B][M][N]){for(int q=0;q<B;++q)for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=a[q][i][k]*b[q][k][j];out[q][i][j]=v;}}")
    add("nested_matmul_broadcast_cpu", s, "matmul_nested_with_broadcasted_dense", "#define B 8\n#define M 16\n#define K 24\n#define N 20\nvoid aten_nested_matmul_broadcast_cpu(float a[B][M][K],float b[K][N],float out[B][M][N]){for(int q=0;q<B;++q)for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=a[q][i][k]*b[k][j];out[q][i][j]=v;}}")
    s = "aten/src/ATen/native/nested/NestedTensorUnaryOps.cpp"
    add("nested_where_cpu", s, "NestedTensor_where", "#define B 8\n#define N 64\nvoid aten_nested_where_cpu(int cond[B][N],float a[B][N],float b[B][N],float out[B][N]){for(int q=0;q<B;++q)for(int i=0;i<N;++i)out[q][i]=cond[q][i]?a[q][i]:b[q][i];}")
    add("nested_where_out_cpu", s, "NestedTensor_where_out", "#define B 8\n#define N 64\nvoid aten_nested_where_out_cpu(int cond[B][N],float a[B][N],float b[B][N],float out[B][N]){for(int q=0;q<B;++q)for(int i=0;i<N;++i)out[q][i]=cond[q][i]?a[q][i]:b[q][i];}")
    add("flatten_indices_launch_cpu", "aten/src/ATen/native/sparse/FlattenIndicesKernel.cpp", "launch", "#define D 3\n#define N 512\nvoid aten_flatten_indices_launch_cpu(int idx[D][N],int size[D],int out[N]){for(int n=0;n<N;++n){int v=0;for(int d=0;d<D;++d)v=v*size[d]+idx[d][n];out[n]=v;}}")
    s = "aten/src/ATen/native/sparse/SparseBinaryOpIntersectionKernel.cpp"
    add("sparse_intersection_launch_cpu", s, "launch", "#define N 512\nvoid aten_sparse_intersection_launch_cpu(float a[N],float b[N],float out[N]){for(int i=0;i<N;++i)out[i]=a[i]*b[i];}")
    add("sparse_intersection_apply_cpu", s, "apply", "#define N 512\nvoid aten_sparse_intersection_apply_cpu(float a[N],float b[N],float out[N]){for(int i=0;i<N;++i)out[i]=a[i]*b[i];}")
    s = "aten/src/ATen/native/sparse/SparseBlasImpl.cpp"
    add("sparse_addmv_csr_cpu", s, "addmv_sparse_csr", "#define R 64\n#define C 64\n#define N 512\nvoid aten_sparse_addmv_csr_cpu(int ptr[R+1],int col[N],float val[N],float x[C],float out[R]){for(int r=0;r<R;++r){float v=0;for(int p=ptr[r];p<ptr[r+1];++p)v+=val[p]*x[col[p]];out[r]=v;}}")
    add("sparse_addmv_bsr_cpu", s, "addmv_sparse_bsr", "#define R 16\n#define C 16\n#define BR 4\n#define BC 4\n#define N 64\nvoid aten_sparse_addmv_bsr_cpu(int ptr[R+1],int col[N],float val[N][BR][BC],float x[C*BC],float out[R*BR]){for(int r=0;r<R;++r)for(int i=0;i<BR;++i){float v=0;for(int p=ptr[r];p<ptr[r+1];++p)for(int j=0;j<BC;++j)v+=val[p][i][j]*x[col[p]*BC+j];out[r*BR+i]=v;}}")
    s = "aten/src/ATen/native/sparse/SparseMatMul.cpp"
    add("sparse_matmul_csr_to_coo_cpu", s, "csr_to_coo", "#define R 64\n#define N 512\nvoid aten_sparse_matmul_csr_to_coo_cpu(int ptr[R+1],int row[N]){for(int r=0;r<R;++r)for(int p=ptr[r];p<ptr[r+1];++p)row[p]=r;}")
    add("sparse_matmul_maxnnz_cpu", s, "_csr_matmult_maxnnz", "#define R 64\n#define N 512\nvoid aten_sparse_matmul_maxnnz_cpu(int ap[R+1],int ac[N],int bp[R+1],int bc[N],int out[R]){for(int r=0;r<R;++r){int n=0;for(int p=ap[r];p<ap[r+1];++p)n+=bp[ac[p]+1]-bp[ac[p]];out[r]=n;}}")
    add("sparse_matmul_cpu", s, "_csr_matmult", "#define R 64\n#define C 64\n#define N 512\nvoid aten_sparse_matmul_cpu(int ap[R+1],int ac[N],float av[N],float b[R][C],float out[R][C]){for(int r=0;r<R;++r)for(int c=0;c<C;++c){float v=0;for(int p=ap[r];p<ap[r+1];++p)v+=av[p]*b[ac[p]][c];out[r][c]=v;}}")
    add("coalesce_sparse_cpu", "aten/src/ATen/native/sparse/SparseTensor.cpp", "_coalesce_sparse_cpu", "#define N 512\nvoid aten_coalesce_sparse_cpu(int idx[N],float val[N],int out_idx[N],float out_val[N],int count[1]){int p=0;for(int i=0;i<N;++i){if(i&&idx[i]==out_idx[p-1])out_val[p-1]+=val[i];else{out_idx[p]=idx[i];out_val[p++]=val[i];}}count[0]=p;}")
    add("sspaddmm_cpu", "aten/src/ATen/native/sparse/SparseTensorMath.cpp", "_sspaddmm_out_cpu", "#define R 64\n#define K 64\n#define C 48\n#define N 512\nvoid aten_sspaddmm_cpu(int row[N],int col[N],float val[N],float b[K][C],float out[R][C]){for(int r=0;r<R;++r)for(int c=0;c<C;++c)out[r][c]=0;for(int p=0;p<N;++p)for(int c=0;c<C;++c)out[row[p]][c]+=val[p]*b[col[p]][c];}")


def main() -> None:
    reductions(); blas(); factories_and_spectral(); indexing_sorting()
    convolutions(); sparse_and_shape(); misc(); remaining_major(); sparse_math(); final_families(); last_bodies()
    rows = []
    for name, source, token, code in E:
        (OUT / f"{name}.c").write_text(code)
        rows.append({"kernel": name, "source": source, "token": token})
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("kernel", "source", "token"))
        w.writeheader(); w.writerows(rows)
    print(f"generated {len(rows)} remaining C fixtures and {MANIFEST}")


if __name__ == "__main__":
    main()
