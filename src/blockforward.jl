"""
    loc_grid([T=Float64], block_sizes; seed=nothing)

Generate a `BlockVector` of random locations on the interval [0, 1).

This is useful for creating sample points for functional data, where each block
represents the observation locations for a single function. Locations within each
block are sorted for real-valued types.

# Arguments
- `T::Type{<:Number}`: The element type of the locations. Defaults to `Float64`.
- `block_sizes::Vector{Int}`: A vector specifying the number of locations for each block.

# Keyword Arguments
- `seed::Union{Nothing,Int}`: An optional seed for reproducibility.

# Examples
```julia-repl
julia> loc = loc_grid([3, 2]; seed=42);

julia> blocklength(loc)
2

julia> blocksizes(loc, 1)
[3, 2]

julia> loc.blocks[1]
3-element Vector{Float64}:
 0.4503389405961936
 0.47740714343281776
 0.6293451231426089
```
"""
function loc_grid(T::Type{<:Number}, BS::Vector{Int}; seed::Union{Nothing,Int}=nothing)::BlockVector{T}
    t = BlockVector{T}(undef_blocks, BS)
    if seed !== nothing
        seed!(seed)
    end
    for i in 1:blocklength(t)
        t.blocks[i] = rand(T, BS[i])
        if T <: Real
            # Sort the block if T is a real number
            sort!(t.blocks[i])
        end
    end
    return t
end

function loc_grid(BS::Vector{Int}; seed::Union{Nothing,Int}=nothing)::BlockVector{Float64}
    y = BlockVector{Float64}(undef_blocks, BS)
    if seed !== nothing
        seed!(seed)
    end
    for i in 1:blocklength(y)
        y.blocks[i] = rand(BS[i])
        sort!(y.blocks[i])
    end
    return y
end


## Generate Forward Matrix##
#1. B-Spline-based method
"""
    mean_fwd(loc, myspline)

Construct a forward mapping matrix `Φ` for a B-spline basis.

Each row of `Φ` corresponds to a location in `loc`, and each column corresponds
to a B-spline basis function. The entry `Φ[i, j]` is the value of the j-th
basis function evaluated at the i-th location.

# Arguments
- `loc::BlockVector{T}`: A block vector of locations.
- `myspline::BSplineMethod{T}`: A `BSplineMethod` object containing the B-spline
  order and knot vector.

# Returns
- `BlockMatrix{T}`: A sparse block matrix representing the B-spline evaluation.

# Examples
```julia-repl
julia> knots = 0.0:0.2:1.0; # Simplified knots for a clear example

julia> myspline = BSplineMethod(4, knots);

julia> loc = loc_grid([2, 1]; seed=1);

julia> Φ = mean_fwd(loc, myspline);

julia> size(Φ)
(3, 8)

julia> Φ[1, 3] ≈ 0.1565989814693208
true
```
"""
function mean_fwd(loc::BlockVector{T}, myspline::BSplineMethod{T})::BlockMatrix{T} where {T}
    BS = blocksizes(loc, 1)        # Get the sizes of the blocks in loc
    order = myspline.order        # Order of the B-spline (e.g., 4 for cubic B-splines)
    knots = myspline.knots        # Knot vector for the B-spline
    m = length(myspline.knots)    # Number of (internal) knots
    p = m + order - 2             # Total number of B-spline basis functions
    basis = BSplineBasis(BSplineOrder(order), knots)
    total_rows = length(loc) # Use length(loc) directly

    # Preallocate lists with a reasonable guess for size (e.g., order * total_rows)
    I_indices = Vector{Int}(undef, 0)
    J_indices = Vector{Int}(undef, 0)
    V_values = Vector{T}(undef, 0)
    sizehint!(I_indices, order * total_rows)
    sizehint!(J_indices, order * total_rows)
    sizehint!(V_values, order * total_rows)

    for j in 1:total_rows
        loc_point = loc[j]
        i, bs = evaluate_all(basis, loc_point) # i is first_active_basis_idx

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
    return BlockMatrix(Φ_sparse, BS, [p])
end



#2. RKHS-based method
"""
    compute_kernel(dist, kernel)

Compute the value of an RBF kernel for a given distance.

This function uses multiple dispatch to select the correct kernel formula based on
the type of the `kernel` object.

# Arguments
- `dist::Real`: The distance between two points.
- `kernel::RBFKernelMethod`: The kernel object containing its parameters.

# Returns
- The scalar value of the kernel evaluation.

# Examples
```julia-repl
julia> dist = 0.4;

julia> kernel = GaussianKernel(1.0);

julia> compute_kernel(dist, kernel) ≈ 0.8521437889662113
true
```
"""
function compute_kernel(dist::T, kernel::GaussianKernel{T}) where {T<:Real}
    return exp(-kernel.γ * dist^2)
end

function compute_kernel(dist::T, kernel::LaplacianKernel{T}) where {T<:Real}
    return exp(-kernel.γ * dist)
end

function compute_kernel(dist::T, kernel::MaternKernel{T}) where {T<:Real}
    # Using SpecialFunctions.gamma and besselk for the Matérn formula
    ν, γ = kernel.ν, kernel.γ
    if iszero(dist)
        return one(T)
    end
    term1 = 2^(1 - ν) / gamma(ν)
    arg = γ * dist
    term2 = arg^ν
    term3 = besselk(ν, arg)
    return term1 * term2 * term3
end

function compute_kernel(dist::T, kernel::CustomKernel{T}) where {T<:Real}
    return kernel.f(dist)
end


"""
    mean_fwd(loc, kernel)

Construct a Gram matrix `K` from an RBF kernel and a set of locations.

The entry `K[i, j]` is the value of the kernel evaluated at the distance
between the i-th and j-th locations, i.e., `K[i, j] = kernel(||loc[i] - loc[j]||)`.

# Arguments
- `loc::BlockVector{T}`: A block vector of evaluation locations.
- `kernel::RBFKernelMethod{T}`: An RBF kernel object. If `kernel.trunc_dist` is set,
  the resulting matrix will be sparse.

# Examples
```julia-repl
julia> loc = loc_grid([2, 1]; seed=1);

julia> kernel = GaussianKernel(50.0); # A kernel with high decay

julia> K = mean_fwd(loc, kernel);

julia> size(K)
(3, 3)

julia> K[1, 2] ≈ 0.022251307501408822
true
```
"""
function mean_fwd(loc::BlockVector{T}, kernel::RBFKernelMethod{T})::BlockMatrix{T} where {T}
    BS = blocksizes(loc, 1)        # Get the sizes of the blocks in loc
    loc_vec = Vector(loc)  # Convert BlockVector to Vector for easier indexing 
    total_rows = length(loc_vec) # Use length(loc) directly

    # Preallocate lists with a reasonable guess for size (e.g., order * total_rows)
    # The actual number of non-zeros is at most order * total_rows
    I_indices = Vector{Int}(undef, 0)
    J_indices = Vector{Int}(undef, 0)
    V_values = Vector{T}(undef, 0)
    sizehint!(I_indices, total_rows^2)  # Adjusted size hint for kernel Gram matrix
    sizehint!(J_indices, total_rows^2)
    sizehint!(V_values, total_rows^2)

    for i in 1:total_rows
        x = loc_vec[i]
        for j in 1:total_rows
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
    K_sparse = sparse(I_indices, J_indices, V_values, total_rows, total_rows)

    return BlockMatrix(K_sparse, BS, BS)
end
