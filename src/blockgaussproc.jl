"""
    AbstractBlockGP{T}

Abstract supertype for all Gaussian Process (GP), characterized by its covariance function.
"""
abstract type AbstractBlockGP{T} end

"""
    BrownianMotion([T=Float64])

A standard Brownian Motion / Wiener Process.

The covariance function is `Σ(s, t) = min(s, t)`.

# Examples
```jldoctest
julia> bm = BrownianMotion();

julia> covariancekernel(bm, 0.2, 0.5)
0.2
```
"""
struct BrownianMotion{T} <: AbstractBlockGP{T} end
BrownianMotion() = BrownianMotion{Float64}()

"""
    BrownianBridge([T=Float64])

A standard Brownian Bridge process, which is a Brownian Motion conditioned
to be zero at `t=0` and `t=1`.

The covariance function is `Σ(s, t) = min(s, t) - s * t`.

# Examples
```jldoctest
julia> bb = BrownianBridge();

julia> covariancekernel(bb, 0.2, 0.5)
0.1
```
"""
struct BrownianBridge{T} <: AbstractBlockGP{T} end
BrownianBridge() = BrownianBridge{Float64}()

"""
    IntegratedBM([T=Float64])

An Integrated Brownian Motion process.

The covariance function is `Σ(s, t) = max(s,t) * min(s,t)^2 / 2 - min(s,t)^3 / 6`.

# Examples
```jldoctest
julia> ibm = IntegratedBM();

julia> covariancekernel(ibm, 0.2, 0.5)
0.008666666666666668
```
"""
struct IntegratedBM{T} <: AbstractBlockGP{T} end
IntegratedBM() = IntegratedBM{Float64}()

"""
    OrnsteinUhlenbeck(θ, σ)

An Ornstein-Uhlenbeck process, which models a mean-reverting stochastic process.

The covariance function is `Σ(s, t) = (σ²/2θ) * (exp(-θ|s-t|) - exp(-θ(s+t)))`.

# Arguments
- `θ::Real`: The mean reversion rate (must be positive).
- `σ::Real`: The volatility parameter (must be non-negative).

# Examples
```jldoctest
julia> ou = OrnsteinUhlenbeck(1.0, 0.5);

julia> ou.θ
1.0
```
"""
struct OrnsteinUhlenbeck{T} <: AbstractBlockGP{T}
    θ::T  # Mean reversion rate
    σ::T  # Volatility

    # Inner constructor
    function OrnsteinUhlenbeck(θ::T, σ::T) where {T}
        @assert θ > 0 "Mean reversion rate θ must be positive"
        @assert σ >= 0 "Volatility σ must be non-negative"
        new{T}(θ, σ)
    end
end

"""
    CustomGP([T=Float64], Σ)

A Gaussian Process defined by a user-supplied covariance function `Σ(s, t)`.

# Arguments
- `T::Type`: The element type of the process. Defaults to `Float64`.
- `Σ::Function`: A two-argument function `(s, t) -> value` that defines the covariance.

# Examples
```jldoctest
julia> my_cov(s, t) = exp(-abs(s - t)); # Exponential covariance

julia> gp = CustomGP(my_cov);

julia> covariancekernel(gp, 0.2, 0.5) ≈ exp(-0.3)
true
```
"""
struct CustomGP{T,F<:Function} <: AbstractBlockGP{T}
    Σ::F  # Custom covariance function Σ(s, t)
end
CustomGP(T::Type, Σ::F) where {F<:Function} = CustomGP{T,F}(Σ)
CustomGP(Σ::F) where {F<:Function} = CustomGP{Float64,F}(Σ)

# --------------------------------------------------------------------
#                      COVARIANCE KERNELS
# --------------------------------------------------------------------
"""
    covariancekernel(process, s, t)
    covariancekernel(process, s)
    covariancekernel(process, loc1, loc2)
    covariancekernel(process, loc)

Compute the covariance for a given Gaussian Process.

This function has several methods:
1.  `covariancekernel(process, s, t)`: Computes the scalar covariance between two points `s` and `t`.
2. `covariancekernel(process, s)`: Computes the covariance of a single point `s` with itself.
3.  `covariancekernel(process, loc1, loc2)`: Computes the cross-covariance matrix between two vectors of locations.
4.  `covariancekernel(process, loc)`: Computes the full covariance matrix (Gram matrix) for a single vector of locations.

# Arguments
- `process::AbstractBlockGP`: The Gaussian Process object defining the covariance structure.
- `s, t::Number`: Scalar locations.
- `loc1, loc2, loc::AbstractVector`: Vectors of locations.

# Examples
```jldoctest
julia> bm = BrownianMotion();

julia> covariancekernel(bm, 0.3, 0.8) # scalar version
0.3

julia> loc = [0.1, 0.5, 0.9];

julia> K = covariancekernel(bm, loc) # matrix version
3×3 Matrix{Float64}:
 0.1  0.1  0.1
 0.1  0.5  0.5
 0.1  0.5  0.9
```

See also [`sample_gp`](@ref) for generating sample paths from the GP.
"""
function covariancekernel end

covariancekernel(process::AbstractBlockGP{T}, s::T) where {T} = covariancekernel(process, s, s)
covariancekernel(process::BrownianMotion{T}, s::T, t::T) where {T} = min(s, t)
covariancekernel(process::BrownianBridge{T}, s::T, t::T) where {T} = min(s, t) - s * t
covariancekernel(process::IntegratedBM{T}, s::T, t::T) where {T} = max(s, t) * min(s, t)^2 / 2 - min(s, t)^3 / 6

function covariancekernel(process::OrnsteinUhlenbeck{T}, s::T, t::T) where {T}
    θ, σ = process.θ, process.σ
    return σ^2 / (2 * θ) * (exp(-θ * abs(s - t)) - exp(-θ * (s + t)))
end

covariancekernel(process::CustomGP{T}, s::T, t::T) where {T} = process.Σ(s, t)

function covariancekernel(process::R, loc1::AbstractVector{T}, loc2::AbstractVector{T}) where {T,R<:AbstractBlockGP{T}}
    n, m = length(loc1), length(loc2)
    Σ = Matrix{T}(undef, n, m)
    for i in 1:n, j in 1:m
        Σ[i, j] = covariancekernel(process, loc1[i], loc2[j])
    end
    return Σ
end

covariancekernel(process::R, loc::AbstractVector{T}) where {T,R<:AbstractBlockGP{T}} = covariancekernel(process, loc, loc)





"""
    sample_gp([μ], process, loc; jitter=1e-6, seed=nothing)

Generate one or more sample paths from a specified Gaussian Process.

# Arguments
- `μ::Function`: (Optional) A mean function `t -> value`. Defaults to a zero mean.
- `process::AbstractBlockGP`: The Gaussian Process object defining the covariance.
- `loc::AbstractVector` or `AbstractBlockVector`: A vector or block vector of locations.
  If a `BlockVector` is provided, a separate sample path is generated for each block.

# Keyword Arguments
- `jitter::Float64`: A small value added to the diagonal of the covariance matrix for
  numerical stability. Default is `1e-6`.
- `seed::Union{Nothing,Int}`: An integer seed for the random number generator to ensure
  reproducibility.

# Returns
- `Vector` or `BlockVector`: The generated sample path(s), matching the type of `loc`.

# Examples
```jldoctest
julia> loc = [0.1, 0.5, 0.9];

julia> bm = BrownianMotion();

julia> y = sample_gp(bm, loc; seed=123);

julia> length(y)
3
```
"""
function sample_gp end

function sample_gp(μ::Function, process::AbstractBlockGP{T}, loc::AbstractVector{T};
    jitter::Float64=1e-6, seed::Union{Nothing,Int}=nothing)::Vector{T} where {T}
    μ_vec = μ.(loc)
    Σ = covariancekernel(process, loc)
    Σ += jitter * I
    d = MvNormal(μ_vec, Σ)
    if seed !== nothing
        seed!(seed)
    end
    y = rand(d)
    return y
end


function sample_gp(process::AbstractBlockGP{T}, loc::AbstractVector{T};
    jitter::Float64=1e-6, seed::Union{Nothing,Int}=nothing)::Vector{T} where {T}
    Σ = covariancekernel(process, loc)
    Σ += jitter * I
    d = MvNormal(Σ)
    if seed !== nothing
        seed!(seed)
    end
    y = rand(d)
    return y
end

function sample_gp(μ::Function, process::AbstractBlockGP{T},
    loc::AbstractBlockVector{T}; jitter::Float64=1e-6,
    seed::Union{Nothing,Int}=nothing)::BlockVector{T} where {T}
    y = similar(loc)
    if seed !== nothing
        seed!(seed)
    end
    for i in 1:blocklength(loc)
        loc_block = loc.blocks[i]
        μ_block = μ.(loc_block)
        Σ_block = covariancekernel(process, loc_block)
        Σ_block += jitter * I
        d = MvNormal(μ_block, Σ_block)
        y.blocks[i] = rand(d)
    end
    return y
end

function sample_gp(process::AbstractBlockGP{T},
    loc::AbstractBlockVector{T}; jitter::Float64=1e-6,
    seed::Union{Nothing,Int}=nothing)::BlockVector{T} where {T}
    y = similar(loc)
    if seed !== nothing
        seed!(seed)
    end
    for i in 1:blocklength(loc)
        loc_block = loc.blocks[i]
        Σ_block = covariancekernel(process, loc_block)
        Σ_block += jitter * I
        d = MvNormal(Σ_block)
        y.blocks[i] = rand(d)
    end
    return y
end


