# 2mm repeatability matrix

All complete LARGE/FP64 comparisons pass (`960000/960000`) with `max_abs=0.01` and `max_rel=5.52852566e-08`:

- unchanged resident binary, runtime timing disabled;
- unchanged resident binary, runtime timing enabled;
- fresh-source host-mapped binary, runtime timing disabled;
- stored-semantic-IR host-mapped binary, runtime timing enabled.

Two additional alternating resident repetitions also pass.  The resident compute-device observations span 77.970–78.783 ms.  These are diagnostic correctness runs, not a publication 5+5 result.
