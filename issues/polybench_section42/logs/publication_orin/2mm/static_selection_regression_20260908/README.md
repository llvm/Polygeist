# Non-static selection regression: 2mm

A fresh ordinary LARGE/FP64 `2mm` build confirms that the new missing-symbol fallback does not run for an externally visible kernel.  Direct selection finds three launches and emits nine runtime calls.

Orin correctness remains `PASS values=960000 failures=0 max_abs=0.01 max_rel=5.52852566e-08`.  The complete compressed silicon output, comparison, build log, and executable hash are retained here.  The observed single-run E2E value is diagnostic and is not a publication 5+5 timing.
