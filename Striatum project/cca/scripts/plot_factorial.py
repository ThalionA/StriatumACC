"""DEPRECATED (round 6).

The 8-config factorial (FS x mean-subtraction x z-scoring x holdout) was
retired when the analysis committed to a single configuration: residual CCA,
FS-excluded, z-scored units, held-out CC. The robustness comparisons it
produced were overwhelming rather than informative.

Use instead:  run_stage2.py  +  plot_stage2.py
"""

if __name__ == "__main__":
    print(__doc__)
