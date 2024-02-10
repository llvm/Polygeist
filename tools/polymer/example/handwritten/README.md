# Polybench evaluation

## Pluto optimization

```sh
polymer-opt gemm.mlir -reg2mem -extract-scop-stmt -pluto-opt -cse 
```

## Usage

Using the `run.sh` script:

```sh
./run.sh <MLIR file (without .mlir)> <Dataset size>
```

For example, to run the `LARGE` configuration of `gemm`:

```sh
# We should use the files with .pluto in them.
./run.sh gemm.pluto LARGE
```


## Result

### Setup

- Compiler: using the LLVM commit in this repository.
- CPU: `Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz`
- Memory: `MemTotal:       791009096 kB`
- OS: `CentOS Linux release 7.6.1810 (Core)`

### Data

| Example  | Data       | MINI | SMALL | MEDIUM | LARGE | EXTRALARGE |
| -------- | ---------- | ---- | ----- | ------ | ----- | ---------- |
| nussinov | `N =`      | 60   | 180   | 500    | 2500  | 5500       |
|          | Origin (s) | -    | -     | 0.1    | 11.87 | 177.79     |
|          | Pluto (s)  | -    | -     | 0.1    | 11.23 | 123.85     |
| gemm     | `NI =`     | 20   | 60    | 200    | 1000  | 2000       |
|          | `NJ =`     | 25   | 70    | 220    | 1100  | 2300       |
|          | `NK =`     | 30   | 80    | 240    | 1200  | 2600       |
|          | Origin (s) | -    | 0.00  | 0.08   | 8.34  | 74.94      |
|          | Pluto (s)  | -    | 0.01  | 0.06   | 8.21  | 75.91      |
| gemver   | `N =`      | 40   | 120   | 400    | 2000  | 4000       |
|          | Origin (s) | -    | -     | 0.01   | 0.07  | 0.29       |
|          | Pluto (s)  | -    | -     | 0.01   | 0.08  | 0.34       |
