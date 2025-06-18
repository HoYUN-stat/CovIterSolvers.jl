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
include("blocktensor.jl")
include("estimationmethod.jl")
include("blockforward.jl")
include("blockgaussproc.jl")
include("blockevaluation.jl")
include("conjlanczos.jl")
include("blockfpca.jl")

export BlockDiagonal, zero_block_diag, undef_block_diag, rand_block_diag, block_outer, ⊙, blocksizes
export AbstractBlockTensor, CovFwdTensor, AdjointCovFwdTensor
export AbstractEstimateMethod, BSplineMethod, RBFKernelMethod, GaussianKernel, LaplacianKernel, MaternKernel, CustomKernel
export loc_grid, mean_fwd
export AbstractBlockGP, BrownianMotion, BrownianBridge, IntegratedBM, OrnsteinUhlenbeck, CustomGP, covariancekernel, sample_gp
export eval_fwd, eval_covariance
export conj_lanczos
export fpca

end