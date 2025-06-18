
"""
    eval_fwd(eval_points, myspline)
    eval_fwd(g, myspline)
    eval_fwd(eval_points, loc, kernel)
    eval_fwd(g, loc, kernel)

Construct a forward mapping matrix `E` by evaluating a basis or kernel at specific points.

This function has two main modes of operation:
1.  **B-spline Basis Evaluation**: When given a `BSplineMethod`, it computes the matrix of
    B-spline basis functions evaluated at each point in `eval_points`.
2.  **RBF Kernel Evaluation**: When given an `RBFKernelMethod`, it computes the Gram matrix
    of kernel evaluations between each point in `eval_points` and each point in `loc`.

# Arguments
- `eval_points::AbstractVector`: A vector of points at which to evaluate the basis/kernel.
- `g::Int`: An integer to create a regular grid of `g` evaluation points from 0 to 1.
- `myspline::BSplineMethod`: A B-spline object containing the order and knots.
- `loc::BlockVector`: A block vector of source locations for the RBF kernel.
- `kernel::RBFKernelMethod`: An RBF kernel object.

# Returns
- `BlockMatrix`: A sparse block matrix representing the evaluation of the basis or kernel.

# Examples
```julia-repl
julia> myeval = range(0, 1; length=10);

julia> knots = 0.0:0.2:1.0;

julia> myspline = BSplineMethod(4, knots);

julia> basis = BSplineBasis(BSplineOrder(order), knots);

julia> E1 = eval_fwd(myeval, myspline);

julia> E2 = eval_fwd(10, myspline);

julia> E1 == E2
true

julia> E1[1, 1] ≈ 1.0
true
```
"""
function eval_fwd end

# B-spline methods
function eval_fwd(eval::AbstractVector{T}, myspline::BSplineMethod{T})::BlockMatrix{T} where {T}
    # BS = blocksizes(loc, 1)        # Get the sizes of the blocks in loc
    order = myspline.order        # Order of the B-spline (e.g., 4 for cubic B-splines)
    knots = myspline.knots        # Knot vector for the B-spline
    m = length(myspline.knots)    # Number of (internal) knots
    p = m + order - 2             # Total number of B-spline basis functions
    basis = BSplineBasis(BSplineOrder(order), knots)
    total_rows = length(eval) # Use length(loc) directly

    # Preallocate lists with a reasonable guess for size (e.g., order * total_rows)
    I_indices = Vector{Int}(undef, 0)
    J_indices = Vector{Int}(undef, 0)
    V_values = Vector{T}(undef, 0)
    sizehint!(I_indices, order * total_rows)
    sizehint!(J_indices, order * total_rows)
    sizehint!(V_values, order * total_rows)

    for j in 1:total_rows
        eval_point = eval[j]
        i, bs = evaluate_all(basis, eval_point) # i is first_active_basis_idx

        for k in 1:order
            col_idx = i - k + 1 # Corrected access based on BSplineKit's reverse order
            val = bs[k]

            if abs(val) > eps(T) # Only push non-zero values
                push!(I_indices, j)
                push!(J_indices, col_idx)
                push!(V_values, val)
            end
        end
    end
    Φ_sparse = sparse(I_indices, J_indices, V_values, total_rows, p)
    return BlockMatrix(Φ_sparse, [total_rows], [p])
end


function eval_fwd(g::Int64, myspline::BSplineMethod{T})::BlockMatrix{T} where {T}
    eval_grid = range(start=0.0, stop=1.0, length=g)  # Regular grid
    return eval_fwd(eval_grid, myspline)
end



function eval_fwd(eval::AbstractVector{T}, loc::BlockVector{T}, kernel::RBFKernelMethod{T})::BlockMatrix{T} where {T}
    total_rows = length(eval)  # Use length(eval) directly
    BS2 = blocksizes(loc, 1)        # Get the sizes of the blocks in loc
    loc_vec = Vector(loc)  # Convert BlockVector to Vector for easier indexing 
    total_cols = length(loc_vec) # Use length(loc) directly

    # Preallocate lists with a reasonable guess for size (e.g., order * total_rows)
    # The actual number of non-zeros is at most order * total_rows
    I_indices = Vector{Int}(undef, 0)
    J_indices = Vector{Int}(undef, 0)
    V_values = Vector{T}(undef, 0)
    sizehint!(I_indices, total_rows * total_cols)
    sizehint!(J_indices, total_rows * total_cols)
    sizehint!(V_values, total_rows * total_cols)

    for i in 1:total_rows
        x = eval[i]
        for j in 1:total_cols
            y = loc_vec[j]
            dist = abs(x - y)  # Compute the distance between x and y
            if kernel.trunc_dist !== nothing && dist > kernel.trunc_dist
                continue  # Skip if distance exceeds truncation distance
            end
            val = compute_kernel(dist, kernel)

            if abs(val) > eps(T) # Only push non-zero values
                push!(I_indices, i)
                push!(J_indices, j)
                push!(V_values, val)
            end
        end
    end

    # Construct the sparse matrix in one go
    K_sparse = sparse(I_indices, J_indices, V_values, total_rows, total_cols)

    return BlockMatrix(K_sparse, [total_rows], BS2)
end

function eval_fwd(g::Int64, loc::BlockVector{T}, kernel::RBFKernelMethod{T})::BlockMatrix{T} where {T}
    eval_grid = range(start=0.0, stop=1.0, length=g)  # Regular grid
    return eval_fwd(eval_grid, loc, kernel)
end



"""
    eval_covariance(E, A)

Computes the covariance matrix `Σ = E * A * E'`.

# Arguments
- `E::BlockMatrix{T}`: The forward evaluation matrix.
- `A::BlockDiagonal{T}`: A block-diagonal matrix, typically representing the prior
  covariance of the coefficients.

# Returns
- `Matrix{T}`: The resulting dense covariance matrix `Σ`.

# Examples
```julia-repl
julia> E = BlockMatrix(ones(3, 2), [3], [2]);

julia> A = BlockDiagonal([[1.0 0.5; 0.5 1.0]]);

julia> Σ = eval_covariance(E, A)
3×3 Matrix{Float64}:
 3.0  3.0  3.0
 3.0  3.0  3.0
 3.0  3.0  3.0
```
"""
function eval_covariance(E::BlockMatrix{T}, A::BlockDiagonal{T})::Matrix{T} where {T}
    g = blocksizes(E, 1)
    L = BlockOuter(E)
    C = undef_block_diag(T, g)
    mul!(C, L, A)
    Σ = C.blocks[1]
    return Σ
end
