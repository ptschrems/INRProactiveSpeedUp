# Proactive Spatial Partitioning for Implicit Neural Representations (INRs)
A theoretic framework for accelerating continuous coordinate network fitting (SIRENs) by proactively predicting quadtree splits from raw spatial features, bypassing unrecoverable compute on complex parent blocks.

## Key Results
+ Proof of Concept: Achieved a 37.44% wall-clock speedup on AMD dGPU hardware over standard brute-force quadtree fitting while maintaining target reconstruction fidelity.
+ Core Theoretical Premise: False Positive partitioning errors ($\alpha$) drive exponential $4^d$ node expansion, whereas False Negative errors ($\beta$) are reduced in the long run due to much lower training cost.

## Implementation & Development
+ Environment: PyTorch on ROCm
+ Development Notes: Algorithmic formulation, mathematical recurrences, and experimental design by Patrick Schrems. LLM assistance utilized during code implementations and debugging.
