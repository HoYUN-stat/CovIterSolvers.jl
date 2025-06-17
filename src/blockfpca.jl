
"""
    fpca(k, E, A, myspline)
    fpca(k, E, A, K; itmax=50)

Performs Functional Principal Component Analysis (FPCA) to find the evaluation of 
leading `k` eigenfunctions and their corresponding eigenvalues.

This function solves the generalized eigenvalue problem `A*v = λ*G*v` or `A*v = λ*K*v`
using one of two methods:

1.  **Direct B-spline Method**: When provided with a `BSplineMethod`, it solves the problem
    directly using a Cholesky and eigenvalue decomposition. This is suitable for smaller,
    well-behaved systems where the Galerkin matrix `G` can be formed.
2.  **Iterative Krylov Method**: When provided with a covariance matrix `K`, it uses the
    Conjugate Lanczos algorithm (`conj_lanczos`) to iteratively find the eigenvalues.
    This is suitable for large, sparse, or matrix-free problems.

# Arguments
- `k::Int`: The number of principal components to compute.
- `E::BlockMatrix`: The evaluation matrix that maps coefficients to function values.
- `A::BlockDiagonal`: A block-diagonal matrix representing the prior covariance of the
  coefficients (the `A` matrix in the eigenproblem).
- `myspline::BSplineMethod`: A B-spline method object defining the basis.
- `K::BlockMatrix`: A Gram matrix representing the inner product for the eigenproblem.

# Keyword Arguments
- `itmax::Int=50`: (Iterative method only) The maximum number of iterations for the
  Lanczos algorithm.

# Returns
- `Tuple{Vector, Matrix}`: A tuple containing:
    1.  A vector of the `k` largest eigenvalues (PC variances).
    2.  A matrix whose columns are the corresponding `k` eigenfunctions (PC vectors).
"""
function fpca end

function fpca(k::Int64, E::BlockMatrix{T}, A::BlockDiagonal{T},
    myspline::BSplineMethod{T})::Tuple{Vector{Float64},Matrix{Float64}} where {T}
    order = myspline.order        # Order of the B-spline (e.g., 4 for cubic B-splines)
    knots = myspline.knots        # Knot vector for the B-spline
    m = length(myspline.knots)    # Number of (internal) knots
    p = m + order - 2             # Total number of B-spline basis functions
    basis = BSplineBasis(BSplineOrder(order), knots)

    A_block = A.blocks[1]  # A has only one block
    E_block = E.blocks[1, 1]  # E has only one block

    # G is the Gram matrix of the B-spline basis functions
    G = Matrix{T}(galerkin_matrix(basis))
    B = similar(G)
    V = similar(G)
    L = cholesky(Symmetric(G)).L
    B = L' * A_block * L # size p × p matrix
    mul!(V, A_block, L)      # use V as a workspace
    mul!(B, transpose(L), V) # B = L' * A_block * L

    (λ, U) = eigen(Symmetric(B))
    V = L' \ U
    V_view = @view V[:, end:-1:end-k+1]

    pc_vals = λ[end:-1:end-k+1]  # Get the largest k eigenvalues
    pc_vecs = E_block * V_view  # Get the corresponding eigenvectors

    return pc_vals, pc_vecs
end

function fpca(k::Int64, E::BlockMatrix{T}, A::BlockDiagonal{T},
    K::BlockMatrix{T}; itmax::Int=50)::Tuple{Vector{Float64},Matrix{Float64}} where {T}

    BS = blocksizes(A, 1)  # Block sizes    
    b0 = BlockVector(rand(T, sum(BS)), BS) # Initial vector

    Tri, U = conj_lanczos(b0, A, K; itmax=itmax)
    # Tri, U = conj_lanczos(b0, D, C; itmax=itmax, reortho_level=:full)

    (λ, S) = eigen(Tri)
    pc_vals = λ[end:-1:end-k+1]  # Get the largest k eigenvalues
    S_view = @view S[:, end:-1:end-k+1]
    pc_vecs = E * U * S_view
    return pc_vals, pc_vecs
end
