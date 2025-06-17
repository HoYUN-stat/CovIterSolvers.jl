"""
    AbstractEstimateMethod{T}

Supertype for all estimation methods in this package.

# Subtypes
- `BSplineMethod{T}`: Methods based on B-splines.
- `RBFKernelMethod{T}`: Methods based on Radial Basis Functions (RBF) kernels.
"""
abstract type AbstractEstimateMethod{T} end

"""
    RBFKernelMethod{T} <: AbstractEstimateMethod{T}

Abstract supertype for methods based on a Radial Basis Function (RBF) kernel.

All RBF kernels define a function `K(x, y)` that depends only on the distance
between `x` and `y`, i.e., `K(x, y) = f(||x - y||)`.

See also [`GaussianKernel`](@ref), [`LaplacianKernel`](@ref), [`MaternKernel`](@ref), [`CustomKernel`](@ref) for concrete implementations.
"""
abstract type RBFKernelMethod{T} <: AbstractEstimateMethod{T} end

# --------------------------------------------------------------------
#                      CONCRETE SUBTYPES
# --------------------------------------------------------------------

"""
    BSplineMethod(order, knots)

An estimation method that uses a B-spline basis.

# Arguments
- `order::Int`: The order of the B-spline (e.g., 4 for cubic B-splines).
- `knots::AbstractVector`: The knot vector defining the B-spline basis.

# Examples
```jldoctest
julia> knots = 0.0:0.1:1.0;

julia> method = BSplineMethod(4, knots);

julia> method.order
4
```
"""
struct BSplineMethod{T} <: AbstractEstimateMethod{T}
    order::Int          # Order of the B-spline
    knots::AbstractVector{T}  # Knot vector for the B-spline
end

"""
    GaussianKernel(γ, [trunc_dist=nothing])

A Gaussian Radial Basis Function (RBF) kernel defined by `K(x, y) = exp(-γ * ||x - y||^2)`.

# Arguments
- `γ::Real`: The positive kernel parameter.
- `trunc_dist::Union{Nothing, Real}`: An optional truncation distance. If provided,
  the kernel will return zero for `||x - y|| > trunc_dist`.

# Examples
```jldoctest
julia> gk = GaussianKernel(0.5)
GaussianKernel{Float64}(0.5, nothing)

julia> gk_trunc = GaussianKernel(0.5, 10.0)
GaussianKernel{Float64}(0.5, 10.0)
```
"""
struct GaussianKernel{T<:Real} <: RBFKernelMethod{T}
    γ::T  # Kernel Parameter
    trunc_dist::Union{Nothing,T}  # Optional truncation distance for sparsity. If `nothing`, no truncation is applied.
end

function GaussianKernel(γ::T) where {T<:Real}
    # Calls the inner constructor, defaulting trunc_dist to nothing
    GaussianKernel{T}(γ, nothing)
end


"""
    LaplacianKernel(γ, [trunc_dist=nothing])

A Laplacian Radial Basis Function (RBF) kernel defined by `K(x, y) = exp(-γ * ||x - y||)`.

# Arguments
- `γ::Real`: The positive kernel parameter.
- `trunc_dist::Union{Nothing, Real}`: An optional truncation distance. If provided,
  the kernel will return zero for `||x - y|| > trunc_dist`.

# Examples
```jldoctest
julia> lk = LaplacianKernel(0.5)
LaplacianKernel{Float64}(0.5, nothing)

julia> lk_trunc = LaplacianKernel(0.5, 10.0)
LaplacianKernel{Float64}(0.5, 10.0)
```
"""
struct LaplacianKernel{T<:Real} <: RBFKernelMethod{T}
    γ::T  # Kernel Parameter
    trunc_dist::Union{Nothing,T}  # Optional truncation distance for sparsity. If `nothing`, no truncation is applied.
end

function LaplacianKernel(γ::T) where {T<:Real}
    # Calls the inner constructor, defaulting trunc_dist to nothing
    LaplacianKernel{T}(γ, nothing)
end


"""
    MaternKernel(ν, γ, [trunc_dist=nothing])

A Matern Radial Basis Function (RBF) kernel defined by
`K(x, y) = (2^(1-ν) / Γ(ν)) * (γ * ||x - y||)^ν * K_{ν}(γ * ||x - y||)`,
where `K_{ν}` is the modified Bessel function of the second kind.

# Arguments
- `ν::Real`: The smoothness parameter (ν = 1/2: Laplacian, ν → ∞: Gaussian).
- `γ::Real`: The positive kernel parameter.
- `trunc_dist::Union{Nothing, Real}`: An optional truncation distance. If provided,
  the kernel will return zero for `||x - y|| > trunc_dist`.

# Examples
```jldoctest
julia> mk = MaternKernel(1.5, 0.5)
MaternKernel{Float64}(1.5, 0.5, nothing)

julia> mk_trunc = MaternKernel(1.5, 0.5, 10.0)
MaternKernel{Float64}(1.5, 0.5, 10.0)
```
"""
struct MaternKernel{T<:Real} <: RBFKernelMethod{T}
    ν::T  # Smoothness Parameter (ν = 1/2: Laplacian, ν → ∞: Gaussian)
    γ::T  # Kernel Parameter
    trunc_dist::Union{Nothing,T} # Optional truncation distance for sparsity. If `nothing`, no truncation is applied.
end

function MaternKernel(ν::T, γ::T) where {T<:Real}
    # Calls the inner constructor, defaulting trunc_dist to nothing
    MaternKernel{T}(ν, γ, nothing)
end


"""
    CustomKernel(f, [trunc_dist=nothing])

A custom Radial Basis Function (RBF) kernel defined by a user-provided function `f`.

The kernel is defined as `K(x, y) = f(||x - y||)`.

# Arguments
- `f::Function`: A function that takes a distance of type `T` and returns a real number.
- `trunc_dist::Union{Nothing, T}`: An optional truncation distance. If provided,
  the kernel will return zero for `||x - y|| > trunc_dist`.

# Examples
```jldoctest
julia> f(x) = exp(-x^2);  # Example custom kernel function

julia> ck = CustomKernel(f)
CustomKernel{Float64,typeof(f)}(f, nothing)

julia> ck_trunc = CustomKernel(f, 10.0)
CustomKernel{Float64,typeof(f)}(f, 10.0)
```
"""
struct CustomKernel{T<:Real,F<:Function} <: RBFKernelMethod{T}
    f::F                         # Custom kernel function, K(x, y) = f(distance_of_type_T)
    trunc_dist::Union{Nothing,T} # Optional truncation distance of type T
end

function CustomKernel(f::F, trunc_dist::T) where {T<:Real,F<:Function}
    # Calls the default new{T_param, F_param}(f, trunc_dist)
    CustomKernel{T,F}(f, trunc_dist)
end

function CustomKernel(f::F) where {F<:Function}
    # Calls the inner constructor, defaulting trunc_dist to nothing
    CustomKernel{Float64,F}(f, nothing)
end


