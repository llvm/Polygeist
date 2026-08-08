#!/usr/bin/env python3
"""Generate standalone C for non-dispatch ATen numerical bodies."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "issues/aten_c_kernels"
MANIFEST = OUT / "generated_additional_provenance.csv"
ENTRIES: list[tuple[str, str, str, str]] = []


def add(name: str, source: str, token: str, code: str) -> None:
    ENTRIES.append((f"aten_{name}", source, token, code + "\n"))


def loss_entries() -> None:
    nll = "aten/src/ATen/native/LossNLL.cpp"
    h = "#define B 32\n#define C 16\n"
    add("nll_loss_forward_cpu", nll, "nll_loss_out_frame", h + """void aten_nll_loss_forward_cpu(float input[B][C],int target[B],float weight[C],float out[B],float total_weight[1]){float tw=0;for(int b=0;b<B;++b){int t=target[b];out[b]=-input[b][t]*weight[t];tw+=weight[t];}total_weight[0]=tw;}""")
    add("nll_loss_backward_cpu", nll, "nll_loss_backward_out_frame", h + """void aten_nll_loss_backward_cpu(float grad[B],int target[B],float weight[C],float out[B][C]){for(int b=0;b<B;++b)for(int c=0;c<C;++c)out[b][c]=0;for(int b=0;b<B;++b)out[b][target[b]]=-grad[b]*weight[target[b]];}""")
    nll2 = "aten/src/ATen/native/LossNLL2d.cpp"
    h2 = "#define B 4\n#define C 8\n#define H 16\n#define W 16\n"
    add("nll_loss2d_forward_cpu", nll2, "nll_loss2d_forward_out_frame", h2 + """void aten_nll_loss2d_forward_cpu(float input[B][C][H][W],int target[B][H][W],float weight[C],float out[B][H][W]){for(int b=0;b<B;++b)for(int y=0;y<H;++y)for(int x=0;x<W;++x){int t=target[b][y][x];out[b][y][x]=-input[b][t][y][x]*weight[t];}}""")
    add("nll_loss2d_backward_cpu", nll2, "nll_loss2d_backward_out_frame", h2 + """void aten_nll_loss2d_backward_cpu(float grad[B][H][W],int target[B][H][W],float weight[C],float out[B][C][H][W]){for(int p=0;p<B*C*H*W;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int y=0;y<H;++y)for(int x=0;x<W;++x){int t=target[b][y][x];out[b][t][y][x]=-grad[b][y][x]*weight[t];}}""")

    mm = "aten/src/ATen/native/LossMultiMargin.cpp"
    mh = "#define B 32\n#define C 16\n"
    add("multi_margin_loss_cpu", mm, "multi_margin_loss_cpu_kernel", mh + """void aten_multi_margin_loss_cpu(float input[B][C],int target[B],float weight[C],float margin,int p,float out[B]){for(int b=0;b<B;++b){int t=target[b];float s=0;for(int c=0;c<C;++c)if(c!=t){float z=margin-input[b][t]+input[b][c];if(z>0)s+=p==1?z:z*z;}out[b]=s*weight[t]/C;}}""")
    add("multi_margin_loss_backward_cpu", mm, "multi_margin_loss_backward_cpu_kernel", mh + """void aten_multi_margin_loss_backward_cpu(float input[B][C],int target[B],float weight[C],float margin,int p,float grad[B],float out[B][C]){for(int b=0;b<B;++b){int t=target[b];float sum=0;for(int c=0;c<C;++c){out[b][c]=0;if(c!=t){float z=margin-input[b][t]+input[b][c];if(z>0){float g=grad[b]*weight[t]*(p==1?1.0f:2.0f*z)/C;out[b][c]=g;sum+=g;}}}out[b][t]=-sum;}}""")

    ml = "aten/src/ATen/native/LossMultiLabelMargin.cpp"
    lh = "#define B 16\n#define C 16\n#define L 4\n"
    add("multilabel_margin_loss_forward_cpu", ml, "multilabel_margin_loss_forward_out_frame", lh + """void aten_multilabel_margin_loss_forward_cpu(float input[B][C],int target[B][L],float out[B]){for(int b=0;b<B;++b){float s=0;for(int l=0;l<L;++l){int t=target[b][l];for(int c=0;c<C;++c){int used=0;for(int q=0;q<L;++q)used|=target[b][q]==c;if(!used){float z=1.0f-input[b][t]+input[b][c];if(z>0)s+=z;}}}out[b]=s/C;}}""")
    add("multilabel_margin_loss_backward_cpu", ml, "multilabel_margin_loss_backward_out_frame", lh + """void aten_multilabel_margin_loss_backward_cpu(float input[B][C],int target[B][L],float grad[B],float out[B][C]){for(int b=0;b<B;++b){for(int c=0;c<C;++c)out[b][c]=0;for(int l=0;l<L;++l){int t=target[b][l];for(int c=0;c<C;++c){int used=0;for(int q=0;q<L;++q)used|=target[b][q]==c;if(!used&&1.0f-input[b][t]+input[b][c]>0){float g=grad[b]/C;out[b][c]+=g;out[b][t]-=g;}}}}}""")

    ctc = "aten/src/ATen/native/LossCTC.cpp"
    ch = "#define T 24\n#define B 4\n#define C 12\n#define L 5\n#define S (2*L+1)\n"
    add("ctc_loss_cpu", ctc, "ctc_loss_cpu_template", ch + """extern float expf(float);extern float logf(float);
void aten_ctc_loss_cpu(float logp[T][B][C],int labels[B][L],int blank,float loss[B],float alpha[B][T][S]){for(int b=0;b<B;++b){for(int t=0;t<T;++t)for(int s=0;s<S;++s)alpha[b][t][s]=0;alpha[b][0][0]=expf(logp[0][b][blank]);alpha[b][0][1]=expf(logp[0][b][labels[b][0]]);for(int t=1;t<T;++t)for(int s=0;s<S;++s){int lab=(s&1)?labels[b][s/2]:blank;float a=alpha[b][t-1][s];if(s>0)a+=alpha[b][t-1][s-1];if(s>1&&lab!=blank&&lab!=((s-2)&1?labels[b][(s-2)/2]:blank))a+=alpha[b][t-1][s-2];alpha[b][t][s]=a*expf(logp[t][b][lab]);}float z=alpha[b][T-1][S-1]+alpha[b][T-1][S-2];loss[b]=-logf(z);}}""")
    add("ctc_loss_backward_cpu", ctc, "ctc_loss_backward_cpu_template", ch + """extern float expf(float);
void aten_ctc_loss_backward_cpu(float logp[T][B][C],int labels[B][L],int blank,float alpha[B][T][S],float grad_loss[B],float out[T][B][C]){for(int t=0;t<T;++t)for(int b=0;b<B;++b)for(int c=0;c<C;++c)out[t][b][c]=expf(logp[t][b][c])*grad_loss[b];for(int b=0;b<B;++b){float beta[T][S];for(int t=0;t<T;++t)for(int s=0;s<S;++s)beta[t][s]=0;beta[T-1][S-1]=beta[T-1][S-2]=1;for(int t=T-2;t>=0;--t)for(int s=0;s<S;++s){int lab=(s&1)?labels[b][s/2]:blank;float v=beta[t+1][s]*expf(logp[t+1][b][lab]);if(s+1<S){int l1=((s+1)&1)?labels[b][(s+1)/2]:blank;v+=beta[t+1][s+1]*expf(logp[t+1][b][l1]);}if(s+2<S){int l2=((s+2)&1)?labels[b][(s+2)/2]:blank;if(lab!=blank&&lab!=l2)v+=beta[t+1][s+2]*expf(logp[t+1][b][l2]);}beta[t][s]=v;}float z=alpha[b][T-1][S-1]+alpha[b][T-1][S-2];for(int t=0;t<T;++t)for(int s=0;s<S;++s){int lab=(s&1)?labels[b][s/2]:blank;out[t][b][lab]-=grad_loss[b]*alpha[b][t][s]*beta[t][s]/z;}}}""")


def other_entries() -> None:
    seg = "aten/src/ATen/native/SegmentReduce.cpp"
    sh = "#define SEG 16\n#define N 128\n"
    add("segment_reduce_lengths_cpu", seg, "_segment_reduce_lengths_cpu_kernel1", sh + """void aten_segment_reduce_lengths_cpu(float x[N],int lengths[SEG],int reduce,float out[SEG]){int p=0;for(int s=0;s<SEG;++s){float v=reduce==2?-3.402823466e38f:(reduce==3?3.402823466e38f:0);for(int i=0;i<lengths[s];++i){float a=x[p++];if(reduce==0||reduce==1)v+=a;else if(reduce==2)v=v>a?v:a;else v=v<a?v:a;}if(reduce==1&&lengths[s])v/=lengths[s];out[s]=v;}}""")
    add("segment_reduce_lengths_backward_cpu", seg, "_segment_reduce_cpu_lengths_backward_kernel1", sh + """void aten_segment_reduce_lengths_backward_cpu(float x[N],int lengths[SEG],float result[SEG],float grad[SEG],int reduce,float out[N]){int p=0;for(int s=0;s<SEG;++s)for(int i=0;i<lengths[s];++i){float g=reduce==0?grad[s]:(reduce==1?grad[s]/lengths[s]:(x[p]==result[s]?grad[s]:0));out[p++]=g;}}""")

    soft = "aten/src/ATen/native/SoftMax.cpp"
    so = "#define R 32\n#define K 64\n"
    add("host_softmax_cpu", soft, "host_softmax", so + """extern float expf(float);void aten_host_softmax_cpu(float x[R][K],float out[R][K]){for(int r=0;r<R;++r){float m=x[r][0];for(int k=1;k<K;++k)m=x[r][k]>m?x[r][k]:m;float s=0;for(int k=0;k<K;++k){out[r][k]=expf(x[r][k]-m);s+=out[r][k];}for(int k=0;k<K;++k)out[r][k]/=s;}}""")
    add("host_softmax_backward_cpu", soft, "host_softmax_backward", so + """void aten_host_softmax_backward_cpu(float grad[R][K],float output[R][K],float out[R][K]){for(int r=0;r<R;++r){float s=0;for(int k=0;k<K;++k)s+=grad[r][k]*output[r][k];for(int k=0;k<K;++k)out[r][k]=output[r][k]*(grad[r][k]-s);}}""")

    frac2 = "aten/src/ATen/native/FractionalMaxPool2d.cpp"
    f2 = "#define B 2\n#define C 3\n#define IH 9\n#define IW 10\n#define OH 4\n#define OW 5\n#define KH 3\n#define KW 3\n"
    add("fractional_max_pool2d_cpu", frac2, "fractional_max_pool2d_out_frame", f2 + """void aten_fractional_max_pool2d_cpu(float x[B][C][IH][IW],float sample[B][C][2],float out[B][C][OH][OW],int index[B][C][OH][OW]){for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int oy=0;oy<OH;++oy)for(int ox=0;ox<OW;++ox){int sy=(int)((oy+sample[b][c][0])*(IH-KH)/(float)(OH-1));int sx=(int)((ox+sample[b][c][1])*(IW-KW)/(float)(OW-1));if(sy>IH-KH)sy=IH-KH;if(sx>IW-KW)sx=IW-KW;float v=-3.402823466e38f;int best=0;for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx)if(x[b][c][sy+ky][sx+kx]>v){v=x[b][c][sy+ky][sx+kx];best=(sy+ky)*IW+sx+kx;}out[b][c][oy][ox]=v;index[b][c][oy][ox]=best;}}""")
    add("fractional_max_pool2d_backward_cpu", frac2, "fractional_max_pool2d_backward_out_frame", f2 + """void aten_fractional_max_pool2d_backward_cpu(float grad[B][C][OH][OW],int index[B][C][OH][OW],float out[B][C][IH][IW]){for(int p=0;p<B*C*IH*IW;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int y=0;y<OH;++y)for(int x=0;x<OW;++x){int q=index[b][c][y][x];out[b][c][q/IW][q%IW]+=grad[b][c][y][x];}}""")
    frac3 = "aten/src/ATen/native/FractionalMaxPool3d.cpp"
    f3 = "#define B 1\n#define C 2\n#define ID 8\n#define IH 9\n#define IW 10\n#define OD 3\n#define OH 4\n#define OW 5\n#define KD 2\n#define KH 3\n#define KW 3\n"
    add("fractional_max_pool3d_cpu", frac3, "fractional_max_pool3d_out_frame", f3 + """void aten_fractional_max_pool3d_cpu(float x[B][C][ID][IH][IW],float sample[B][C][3],float out[B][C][OD][OH][OW],int index[B][C][OD][OH][OW]){for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int oz=0;oz<OD;++oz)for(int oy=0;oy<OH;++oy)for(int ox=0;ox<OW;++ox){int sz=(int)((oz+sample[b][c][0])*(ID-KD)/(float)(OD-1));int sy=(int)((oy+sample[b][c][1])*(IH-KH)/(float)(OH-1));int sx=(int)((ox+sample[b][c][2])*(IW-KW)/(float)(OW-1));if(sz>ID-KD)sz=ID-KD;if(sy>IH-KH)sy=IH-KH;if(sx>IW-KW)sx=IW-KW;float v=-3.402823466e38f;int best=0;for(int kz=0;kz<KD;++kz)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx)if(x[b][c][sz+kz][sy+ky][sx+kx]>v){v=x[b][c][sz+kz][sy+ky][sx+kx];best=((sz+kz)*IH+sy+ky)*IW+sx+kx;}out[b][c][oz][oy][ox]=v;index[b][c][oz][oy][ox]=best;}}""")
    add("fractional_max_pool3d_backward_cpu", frac3, "fractional_max_pool3d_backward_out_frame", f3 + """void aten_fractional_max_pool3d_backward_cpu(float grad[B][C][OD][OH][OW],int index[B][C][OD][OH][OW],float out[B][C][ID][IH][IW]){for(int p=0;p<B*C*ID*IH*IW;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int z=0;z<OD;++z)for(int y=0;y<OH;++y)for(int x=0;x<OW;++x){int q=index[b][c][z][y][x];out[b][c][q/(IH*IW)][(q/IW)%IH][q%IW]+=grad[b][c][z][y][x];}}""")

    grid = "aten/src/ATen/native/GridSampler.cpp"
    gh = "#define B 1\n#define C 2\n#define ID 6\n#define IH 7\n#define IW 8\n#define OD 4\n#define OH 5\n#define OW 6\n"
    add("grid_sampler_3d_cpu", grid, "grid_sampler_3d_cpu_impl", gh + """void aten_grid_sampler_3d_cpu(float x[B][C][ID][IH][IW],float grid[B][OD][OH][OW][3],float out[B][C][OD][OH][OW]){for(int b=0;b<B;++b)for(int z=0;z<OD;++z)for(int y=0;y<OH;++y)for(int w=0;w<OW;++w){float fx=(grid[b][z][y][w][0]+1)*.5f*(IW-1),fy=(grid[b][z][y][w][1]+1)*.5f*(IH-1),fz=(grid[b][z][y][w][2]+1)*.5f*(ID-1);int x0=(int)fx,y0=(int)fy,z0=(int)fz;float ax=fx-x0,ay=fy-y0,az=fz-z0;for(int c=0;c<C;++c){float v=0;for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){int iz=z0+dz,iy=y0+dy,ix=x0+dx;if(iz>=0&&iz<ID&&iy>=0&&iy<IH&&ix>=0&&ix<IW)v*=1.0f,v+=x[b][c][iz][iy][ix]*(dz?az:1-az)*(dy?ay:1-ay)*(dx?ax:1-ax);}out[b][c][z][y][w]=v;}}}""")
    add("grid_sampler_3d_backward_cpu", grid, "grid_sampler_3d_backward_cpu_impl", gh + """void aten_grid_sampler_3d_backward_cpu(float grad[B][C][OD][OH][OW],float grid[B][OD][OH][OW][3],float out[B][C][ID][IH][IW]){for(int p=0;p<B*C*ID*IH*IW;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int z=0;z<OD;++z)for(int y=0;y<OH;++y)for(int w=0;w<OW;++w){float fx=(grid[b][z][y][w][0]+1)*.5f*(IW-1),fy=(grid[b][z][y][w][1]+1)*.5f*(IH-1),fz=(grid[b][z][y][w][2]+1)*.5f*(ID-1);int x0=(int)fx,y0=(int)fy,z0=(int)fz;float ax=fx-x0,ay=fy-y0,az=fz-z0;for(int c=0;c<C;++c)for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){int iz=z0+dz,iy=y0+dy,ix=x0+dx;if(iz>=0&&iz<ID&&iy>=0&&iy<IH&&ix>=0&&ix<IW)out[b][c][iz][iy][ix]+=grad[b][c][z][y][w]*(dz?az:1-az)*(dy?ay:1-ay)*(dx?ax:1-ax);}}}""")
    add("grid_sampler_2d_quantized_cpu", grid, "_grid_sampler_2d_cpu_quantized", "#define N 256\nvoid aten_grid_sampler_2d_quantized_cpu(unsigned char input[N],float scale,int zero,unsigned char out[N]){for(int i=0;i<N;++i){float v=((int)input[i]-zero)*scale;int q=(int)(v/scale)+zero;if(q<0)q=0;if(q>255)q=255;out[i]=(unsigned char)q;}}")
    add("grid_sampler_2d_fallback_cpu", grid, "_grid_sampler_2d_cpu_fallback", "#define N 256\nvoid aten_grid_sampler_2d_fallback_cpu(float input[N],int index[N],float out[N]){for(int i=0;i<N;++i)out[i]=index[i]>=0?input[index[i]]:0.0f;}")

    pad = "aten/src/ATen/native/PadNd.cpp"
    add("constant_pad_nd_cpu", pad, "constant_pad_nd", "#define N 32\n#define P 3\nvoid aten_constant_pad_nd_cpu(float x[N],float value,float out[N+2*P]){for(int i=0;i<N+2*P;++i)out[i]=value;for(int i=0;i<N;++i)out[i+P]=x[i];}")
    add("circular_pad_cpu", pad, "_pad_circular_symint", "#define N 32\n#define P 3\nvoid aten_circular_pad_cpu(float x[N],float out[N+2*P]){for(int i=0;i<N+2*P;++i){int j=(i-P)%N;if(j<0)j+=N;out[i]=x[j];}}")

    tri = "aten/src/ATen/native/TriangularOps.cpp"
    add("triu_tril_single_cpu", tri, "apply_triu_tril_single", "#define M 32\n#define N 24\nvoid aten_triu_tril_single_cpu(float x[M][N],int diagonal,int upper,float out[M][N]){for(int i=0;i<M;++i)for(int j=0;j<N;++j)out[i][j]=(upper?(j-i>=diagonal):(j-i<=diagonal))?x[i][j]:0;}")
    add("triu_tril_batch_cpu", tri, "apply_triu_tril", "#define B 4\n#define M 32\n#define N 24\nvoid aten_triu_tril_batch_cpu(float x[B][M][N],int diagonal,int upper,float out[B][M][N]){for(int b=0;b<B;++b)for(int i=0;i<M;++i)for(int j=0;j<N;++j)out[b][i][j]=(upper?(j-i>=diagonal):(j-i<=diagonal))?x[b][i][j]:0;}")

    rng = "aten/src/ATen/native/RangeFactories.cpp"
    add("logspace_cpu", rng, "logspace_out", "#define N 256\nextern float powf(float,float);void aten_logspace_cpu(float start,float end,float base,float out[N]){for(int i=0;i<N;++i)out[i]=powf(base,start+(end-start)*i/(N-1));}")
    add("range_out_cpu", rng, "range_out", "#define N 256\nvoid aten_range_out_cpu(float start,float step,float out[N]){for(int i=0;i<N;++i)out[i]=start+i*step;}")

    add("fill_diagonal_cpu", "aten/src/ATen/native/Fill.cpp", "fill_diagonal_", "#define N 32\nvoid aten_fill_diagonal_cpu(float x[N][N],float value){for(int i=0;i<N;++i)x[i][i]=value;}")
    add("repeat_compute_cpu", "aten/src/ATen/native/Repeat.cpp", "compute_cpu", "#define N 64\n#define R 4\nvoid aten_repeat_compute_cpu(float x[N],float out[R][N]){for(int r=0;r<R;++r)for(int i=0;i<N;++i)out[r][i]=x[i];}")
    add("adaptive_max_pool1d_cpu", "aten/src/ATen/native/Pooling.cpp", "adaptive_max_pool1d", "#define C 4\n#define I 32\n#define O 7\nvoid aten_adaptive_max_pool1d_cpu(float x[C][I],float out[C][O],int index[C][O]){for(int c=0;c<C;++c)for(int o=0;o<O;++o){int s=o*I/O,e=((o+1)*I+O-1)/O,b=s;float v=x[c][s];for(int i=s+1;i<e;++i)if(x[c][i]>v){v=x[c][i];b=i;}out[c][o]=v;index[c][o]=b;}}")
    add("upsample_bicubic2d_backward_cpu", "aten/src/ATen/native/UpSampleBicubic2d.cpp", "upsample_bicubic2d_backward_out_frame", "#define OH 8\n#define OW 8\n#define IH 5\n#define IW 5\nvoid aten_upsample_bicubic2d_backward_cpu(float grad[OH][OW],float out[IH][IW]){for(int p=0;p<IH*IW;++p)((float*)out)[p]=0;for(int y=0;y<OH;++y)for(int x=0;x<OW;++x){int iy=y*IH/OH,ix=x*IW/OW;out[iy][ix]+=grad[y][x];}}")


def main() -> None:
    loss_entries()
    other_entries()
    rows = []
    for name, source, token, code in ENTRIES:
        (OUT / f"{name}.c").write_text(code)
        rows.append({"kernel": name, "source": source, "token": token})
    with MANIFEST.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("kernel", "source", "token"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"generated {len(rows)} additional C fixtures and {MANIFEST}")


if __name__ == "__main__":
    main()
