# CovIterSolvers.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HoYUN-stat.github.io/CovIterSolvers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HoYUN-stat.github.io/CovIterSolvers.jl/dev)
[![Build Status](https://github.com/HoYUN-stat/CovIterSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HoYUN-stat/CovIterSolvers.jl/actions/workflows/CI.yml)

A Julia package for efficient covariance smoothing and functional principal component analysis (FPCA) using Krylov subspace methods.

## Overview

For large datasets, traditional covariance smoothing using direct factorization can be computationally expensive. 
`CovIterSolvers.jl` provides a high-performance pipeline to address this challenge by leveraging iterative methods.

The key features include:

1.  Generation of multiple sample paths from Gaussian processes using functionality from `BlockArrays.jl`.
2.  Covariance smoothing using B-splines or Reproducing Kernel Hilbert Space (RKHS) methods, accelerated by solvers from `Krylov.jl`.
3.  Functional Principal Component Analysis (FPCA) performed with a Lanczos tridiagonalization variant.

## Installation
It can be installed directly from GitHub:

```julia
import Pkg
Pkg.add(url="https://github.com/HoYUN-stat/CovIterSolvers.jl")
```
## Reference
If you use this package, please cite the paper: 

Yun, H. and Panaretos, V. M. (2025). Fast and Cheap Covariance Smoothing. arXiv preprint [arXiv:2501.08265](https://arxiv.org/abs/2501.08265).

#### BibTeX

```
@article{yun2025fast,
  title={Fast and Cheap Covariance Smoothing},
  author={Yun, Ho and Panaretos, Victor M},
  journal={arXiv preprint arXiv:2501.08265},
  doi  ={10.48550/arXiv.2501.08265},
  year={2025},
}
```

## Quick Start

Here is a runnable example demonstrating how to perform covariance smoothing using the B-spline method with a Krylov solver.

```julia
using CovIterSolvers
using LinearAlgebra
using Random: seed!
using Krylov
import Krylov: statistics, solution
```

**Generate some sample data**
```julia
n = 10              # Number of sample paths
BS = rand(10:12, n) # Number of observations per path
loc = loc_grid(BS; seed=1)
y = sample_gp(BrownianMotion(), loc; jitter=1e-4, seed=1)
Y = y ⊙ y # Create the cross-product observations

println("Sample data generated with $(sum(BS)) total observations.")
```

**Define the B-spline basis**
```julia
order = 4   # Cubic B-splines
m = 10      # Number of internal knots
p = m + order - 2
knots = range(0, 1; length=m)
myspline = BSplineMethod(order, knots)
```

**Set up and run the solver**
```julia
Φ = mean_fwd(loc, myspline)
Φ2 = CovFwdTensor(Φ)
A_spline = zero_block_diag([p])
kc_spline = KrylovConstructor(Y, A_spline)
workspace = LsqrWorkspace(kc_spline)
lsqr!(workspace, Φ2, Y; history=true)
A_spline = solution(workspace)
stats = statistics(workspace)

println("LSQR solver finished after $(stats.niter) iterations.")
```

**Evaluate the smoothed covariance**
```julia
myeval = range(0, 1, length=100)
E_spline = eval_fwd(myeval, myspline)
Σ_spline = eval_covariance(E_spline, A_spline)

println("Smoothed covariance matrix of size $(size(Σ_spline)) computed successfully.")
```
