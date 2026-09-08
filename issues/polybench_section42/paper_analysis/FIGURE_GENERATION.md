# Figure generation

The source SVG figures are generated from the retained five-warmup/five-sample
CSVs by:

```text
python3 issues/polybench_section42/generate_paper_analysis.py
```

The LaTeX-ready PDFs were converted from those SVGs with CairoSVG 2.9.1:

```text
/home/arjaiswal/.local/bin/uv run --with cairosvg cairosvg issues/polybench_section42/paper_analysis/polybench_cpu_runtime.svg -o issues/polybench_section42/paper_analysis/polybench_cpu_runtime.pdf -f pdf
/home/arjaiswal/.local/bin/uv run --with cairosvg cairosvg issues/polybench_section42/paper_analysis/polybench_gpu_runtime.svg -o issues/polybench_section42/paper_analysis/polybench_gpu_runtime.pdf -f pdf
```

The PDFs are presentation derivatives. The CSVs, generator, and SVGs are the
reproducible source artifacts.
