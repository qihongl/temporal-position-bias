"""
TCM Model Configuration
========================
Default parameters and sweep ranges.
"""

# ---- Data generation ----
D_ITEM = 64                # item embedding dimension
D_CONTEXT = 64             # context vector dimension
SEQ_MIN = 60               # min training sequence length
SEQ_MAX = 180              # max training sequence length
SEQ_TEST = 100             # test sequence length (fixed)

# ---- TCM dynamics ----
RHO = 0.95                 # context persistence (0 = no memory, 1 = perfect)
SIGMA_MEASURE = 0.10       # independent measurement noise
SIGMA_RETRIEVAL = 0.04     # retrieval noise (for matrix-memory variant)

# ---- Training ----
EPOCHS = 4000
BATCH_SIZE = 64
LR = 5e-4

# ---- Evaluation ----
TEST_POSITIONS = [0.20, 0.40, 0.60, 0.80]
N_TEST_PER_POS = 500       # evaluation trials per position
EVAL_BATCH = 50

# ---- Sweep defaults ----
SWEEP_RHOS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
SWEEP_SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.25]
SWEEP_EPOCHS = 2000        # fewer epochs for sweeps
SWEEP_BATCH = 32

# ---- Human benchmark (Hu et al.) ----
HUMAN_ERRORS = [13.41, 7.57, -1.91, -7.36]  # at 20%, 40%, 60%, 80%
