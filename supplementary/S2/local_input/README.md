# TCR local-controllability input placement: fresh-seed confirmation

Experiment incorporated into Natural Computing manuscript (Table 4). Generated 2026-10-20.

## Hypothesis / prespecified rule
For the zero-temperature functional graph with signal-flow edges f(i)->i, d_j is one plus the maximum shortest-path distance among vertices reachable from j. Select the (smallest-index) j maximizing d_j, using graph topology alone; no memory scores in encoder selection. Compare with exhaustive best among all n single-neuron candidates (oracle), best of 13 random single-neuron locations, best of 13 random unit-norm dense encoders, and best node on longest cycle.

## Setup
Fresh Gaussian scores, fixed independent Bernoulli signs, zero diagonal; each row softmax allocation at tau>0; at zero a single winner per row; global RMS normalization W=s0*M/sqrt(sum(M^2)/n). Sizes n=150 with 12 seeds and n=500 with 10 seeds for each s0=0.6, 0.8. Tau=0,0.08,0.53,3.5. Input signal variance 1, norm(b)=1, process-noise covariance eps*I for eps=1e-5,1e-3,1e-2. Only stable matrices rho(W)<1 retained; exclusions tabulated in CSV. The same seed is used across temperatures, with independent seeds between s0 arms and network sizes.

MC calculation: R=sum_{k>=0} W^k(W^T)^k solves R=W R W^T+I. Define K_H=[b,Wb,...,W^(H-1)b], Y=R^(-1/2)K_H. Noise-aware total population memory is sum lambda/(lambda+eps), where lambda are eigenvalues of Y^T Y, equivalent to tr[(G+eps R)^(-1)G] for G=K K^T, when H covers history. H=140 for n150 and H=160 for n500. The manuscript gives an analytic omitted-tail bound at the zero-temperature endpoint (the only temperature reported in Table 3). That bound does not certify the finite-history approximations at positive temperatures in the supplementary exploratory records, especially near criticality.

## Results summary
See summary.csv (paired descriptive unadjusted 95% t confidence intervals), results.csv (per-seed records), rank_audit.csv (290 exact-vs-computed local controllability rank comparisons). Exact rank audit was on 10 seeds for each n=6,9,14; all 290 comparisons matched. This is not an algebraic proof. For tau=.53, s0=.8, 1/12 n150 and 1/10 n500 networks unstable (three rows per excluded seed).

At eps=1e-3 and n=500,s0=.8: tau0 rule=16.15, exhaustive single-node oracle=17.86, best random13=11.61, best dense13=12.14. At tau0.08 rule=14.58, oracle=18.25, random13=13.94. At tau.53 rule=10.27, oracle=15.48, random13=11.98. So a strong zero-T result but no robust positive-T transfer advantage over random-node search. For s0=.6 even at tau0 rule does not decisively outperform best-of-13 random node choices. All graph-max-d nodes were uniquely determined and cyclic in the tested generated graphs.

Table 4 reports only the zero-temperature endpoint; no claim of finite-noise universal optimality or positive-temperature design rule.

## Reproduce
python tcr_local_rule_confirm.py
python report.py

The current script writes into its own directory. Run it on a copy to preserve the supplied results. Requires NumPy and SciPy, plus pandas for reporting.
