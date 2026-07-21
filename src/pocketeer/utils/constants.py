"""Constants used throughout the pocketeer package."""

# Tessellation constants
MIN_ATOMS_FOR_TESSELLATION = 4

# Scoring constants
MIN_RADIUS_THRESHOLD = 3.5
MAX_RADIUS_THRESHOLD = 5.0

# Pocket ranking model: a logistic-regression "ligandability" classifier trained
# on the P2Rank chen11 benchmark and evaluated on the coach420 set
# (top-1 ~87%, ROC-AUC ~0.85, vs ~72% for size-based scoring). It scores each
# pocket from six features computed entirely from the Pocket object. Features are
# standardised with SCORE_LR_MEAN/STD, combined with SCORE_LR_COEF + bias, and
# squashed by a logistic into a 0-1 score. No external model file or ML
# dependency is needed at runtime - only these baked-in coefficients.
#
# Feature order: (n_residues, n_spheres, volume_per_sphere, mean_radius,
#                 mean_sasa, fraction_aromatic_residues)
SCORE_LR_MEAN = (18.68027, 85.5873, 30.03137, 3.76683, 9.8632, 0.14239)
SCORE_LR_STD = (12.45253, 81.93378, 9.78326, 0.24232, 2.34571, 0.09618)
SCORE_LR_COEF = (1.78333, -0.68905, -0.83916, 0.50621, 0.23933, 0.31004)
SCORE_LR_BIAS = -0.3192
SCORE_AROMATIC_RESIDUES = frozenset({"PHE", "TRP", "TYR", "HIS"})

# I/O constants
ALPHABET_SIZE = 26

# Volume calculation constants
VOXEL_SIZE = 0.5
