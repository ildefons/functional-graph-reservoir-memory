# TCR noise-aware linear memory study (manuscript Table 7)

Originally run as a separate pilot, subsequently included in Natural Computing manuscript v3.3.58, Table 7.

## Protocol
- N=60, s0=0.8, 20 seeds (numpy default_rng seed 6000+index).
- One fixed score/sign realization per seed, no diagonal, same fixed b across temperatures and baselines. b components Uniform(-1,1) and b normalized to Euclidean norm one; independent scalar input variance 1.
- Positive temperature row softmax plus global RMS normalization, zero temperature one signed incoming edge per row.
- Baselines: one Gaussian recurrent realization with diagonal zero and global Frobenius normalization, and one signed simple cycle, each per seed. Baselines are NOT coupled in random entries to temperature family.
- Only Schur-stable W are evaluated. At tau 0.53 one of 20 seeds was unstable, so the N=19 population there is not paired with every arm. All other parameter combinations stable.
- x_t=W x_(t-1)+b u_t + xi_t, xi_t ~ iid N(0, epsilon I); u_t iid mean-zero, var=1. State noise injected at EVERY timestep.
- Signal covariance S solves S=W S W^T + bb^T; process-noise propagation U solves U=W U W^T + I. Total state covariance C=S+epsilon U.
- m(k)=(W^k b)^T C^{-1}(W^k b) for k>=0. Population total M=tr(C^{-1} S), evaluated with Cholesky linear solves; no finite-sample regression or SVD cutoff. This is the summed scalar-input linear reconstruction score, NOT Fisher memory or the observed nonlinear tanh experiment.
- Epsilon is an absolute variance with ||b||=1, so changing input-vector norm changes the effective noise level. The results are conditional on this explicit normalization.
- Noise levels: 1e-6, 1e-4, 1e-2, 1e-1.
- Source script and raw table included. Study uses the frozen original numerical protocol; no rerun or parameter change is implied by its inclusion in v3.3.58.

## Interpretation
The simple cycle is strongest at every tested noise level, and the higher-temperature family has modest advantages at low noise; at noise epsilon=0.01 family variants are nearly tied. Increasing algebraic rank does not ensure a large gain in effective memory. These comparisons cannot be directly interpreted as the experimental s0=0.8 nonlinear memory numbers in the manuscript.

## Caveats
- Pilot scale N60, only 20 draws and one input strength; extension to N150 and observation-noise alternative could matter.
- No-neuron-update nonlinearity, no finite-data effects. A state-noise model is an assumption, not a calibration of noise in the existing data.
- Some stable matrices have high spectral radius; unpaired omission only for one instability at tau=.53.
