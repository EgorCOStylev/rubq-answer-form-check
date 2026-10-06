"""Fixed parameters of the main analysis (section 5-8 of src/analysis_plan.md)."""
BOOT_B = 10000
BOOT_SEED = 20261006
MARGIN = 0.03
SUBSET_SEED = 20261007
N_SUBSETS = 20
K_VALUES = list(range(1, 11))
CV_SEED = 20261008
CV_FOLDS = 5
ECE_BINS = 10
COVERAGES = (0.2, 0.5, 0.8)
CURVE_POINTS = 20
N_SAMPLES = 10
