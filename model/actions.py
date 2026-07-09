"""
ARC-AGI-3 action / grid constants shared across the model.

The ARC-AGI-3 API is fixed: grids are up to 64x64 cells of integers 0-15,
and the action space is RESET + ACTION1-5 (simple) + ACTION6 (click at
x, y in [0, 63]). See https://docs.arcprize.org/games
"""

GRID_SIZE = 64
"""Max height/width of an ARC-AGI-3 grid; smaller grids are padded to this."""

NUM_COLORS = 16
"""Cell values 0-15."""

PAD_SYMBOL = 16
"""Extra vocabulary entry marking cells outside a smaller-than-64x64 grid."""

NUM_SYMBOLS = NUM_COLORS + 1
"""Cell vocabulary: 16 colors + the padding symbol."""

# Action-type indices (the "simple" head). ACTION6 additionally carries an
# (x, y) coordinate in [0, GRID_SIZE).
RESET = 0
ACTION1 = 1
ACTION2 = 2
ACTION3 = 3
ACTION4 = 4
ACTION5 = 5
ACTION6 = 6
NUM_ACTION_TYPES = 7
