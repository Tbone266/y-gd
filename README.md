# y-gd
Original formulation and experiments for Y-Gradient Descent (Y-GD),  a baseline-relative triadic update rule developed by Todd Clark (2025).
# Y-Gradient Descent (Y-GD)
**Todd Clark (2025)**  
**Original, timestamped formulation of Baseline-Relative Gradient Descent**

This repository contains the founding documentation for **Y-Gradient Descent (Y-GD)**, a
novel optimization method based on *baseline-relative, triadic update dynamics*.

Y-GD introduces:
- a triadic update operator \( M(x,y,z) = z + y(x - z) \)
- a proportional change mediator \( Y_t \)
- a moving baseline \( z_t \) that stabilizes learning
- a multiplicative structure for composing mediators (gradient, momentum, regularization)

The included PDF contains the first formulation of the method, experiments, and analysis.
This repository serves as the **public, timestamped record** establishing authorship and
conceptual priority for the Y-GD framework.

## Contents
- `Y-GD_experiments_2025.pdf` — Original paper with experiments and analysis  
- (Future) minimal PyTorch implementation  
- (Future) demo notebooks

## Citation
If referencing this work:

