module CovIterSolvers

using LinearAlgebra
using Random: seed! # For Reproducibility
using Distributions: MvNormal # To sample Gaussian processes
using SpecialFunctions: gamma, besselk # For Matern kernel
using LaTeXStrings # For docstrings
using BlockArrays
using Krylov
import Krylov: FloatOrComplex
using SparseArrays
using BSplineKit

include("blockdiag.jl")
export BlockDiagonal, zero_block_diag, undef_block_diag,
    rand_block_diag, block_outer, ⊙, blocksizes
include("blocktensor.jl")
export AbstractBlockTensor, CovFwdTensor, AdjointCovFwdTensor

include("estimationmethod.jl")
export AbstractEstimateMethod, BSplineMethod, RBFKernelMethod,
    GaussianKernel, LaplacianKernel, MaternKernel, CustomKernel

include("blockforward.jl")
export loc_grid, mean_fwd

include("blockgaussproc.jl")
export AbstractBlockGP, BrownianMotion, BrownianBridge,
    IntegratedBM, OrnsteinUhlenbeck, CustomGP,
    covariancekernel, sample_gp

include("blockevaluation.jl")
export eval_fwd, eval_covariance

include("conjlanczos.jl")
export conj_lanczos

include("blockfpca.jl")
export fpca

end