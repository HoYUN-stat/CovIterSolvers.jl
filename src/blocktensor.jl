"""
    AbstractBlockTensor{T}

Element-free tensor type for block-wise operations.

See also [`BlockOuter`](@ref), [`CovFwdTensor`](@ref), [`AdjointBlockOuter`](@ref), and [`AdjointCovFwdTensor`](@ref).
"""
abstract type AbstractBlockTensor{T} end


# --------------------------------------------------------------------
#                      CONCRETE BLOCK TENSORS
# --------------------------------------------------------------------
"""
    BlockOuter(E, [workspace])
    E ⊙ E

Represents a linear operator `L` that performs a block-wise outer product.

The operator `L` is defined by its action on a `BlockDiagonal` matrix `A`:
`B = L(A)`, where the j-th block of `B` is computed as
`B_j = ∑ᵢ Eⱼᵢ Aᵢ Eᵢⱼ'`.

The infix operator `⊙` is an alias for the constructor `BlockOuter(E)`. It requires
that both operands are the same object (e.g., `E ⊙ E`).

# Fields
- `E::AbstractBlockMatrix`: The block matrix that defines the operator.
- `workspace::Matrix`: Pre-allocated matrix to be used for intermediate calculations.

# Examples
```julia-repl
julia> E = BlockMatrix(rand(3, 2), [2, 1], [1, 1]);

julia> L = E ⊙ E; # Element-free operator

julia> L_adj = L'; # Take the adjoint

julia> L_adj isa AdjointBlockOuter
true
```

See also [`AdjointBlockOuter`](@ref).
"""
struct BlockOuter{T,R<:AbstractBlockMatrix{T}} <: AbstractBlockTensor{T}
    E::R
    workspace::Matrix{T}

    # Inner Constructor
    function BlockOuter(E_input::R) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(E_input) == T "Element type mismatch: F has eltype $(eltype(E_input)), but expected $T."
        workspace = Matrix{T}(undef, maximum(blocksizes(E_input, 1)), maximum(blocksizes(E_input, 2)))
        return new{T,R}(E_input, workspace)
    end
    function BlockOuter(E_input::R, workspace::Matrix{T}) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(E_input) == T "Element type mismatch: F has eltype $(eltype(E_input)), but expected $T."
        @assert size(workspace, 1) >= maximum(blocksizes(E_input, 1)) "workspace has insufficient rows for F."
        @assert size(workspace, 2) >= maximum(blocksizes(E_input, 2)) "workspace has insufficient columns for F."

        return new{T,R}(E_input, workspace)
    end
end
Base.eltype(::Type{<:BlockOuter{T}}) where {T} = T
Base.size(L::BlockOuter) = (sum(blocksizes(L.E, 1) .^ 2), sum(blocksizes(L.E, 2) .^ 2))
Base.size(L::BlockOuter, d::Int) = size(L)[d]

"""
    E ⊙ E

The infix operator `⊙` is an alias for the constructor `BlockOuter(E)`. It requires
that both operands are the same object (e.g., `E ⊙ E`).

See also [`BlockOuter`](@ref).
"""
function ⊙(E1::AbstractBlockMatrix, E2::AbstractBlockMatrix)
    @assert E1 === E2 "The ⊙ operator is only defined for the same matrix, e.g., F ⊙ F."
    return BlockOuter(E1)
end


"""
    CovFwdTensor(F, [workspace])

Represents a covariance-based forward operator `L` for B-splines or RKHS.

Its action `L = O * (F ⊙ F) * O'` depends on the block structure of `F``.

# Fields
- `F::AbstractBlockMatrix`: The block matrix that defines the core of the operator.
- `workspace::Matrix`: Pre-allocated matrix to be used for intermediate calculations.

# Examples
```julia-repl
julia> F = BlockMatrix(rand(3, 2), [2, 1], [1, 1]);

julia> L = CovFwdTensor(F);

julia> L_adj = adjoint(L); # Or L'

julia> L_adj isa AdjointCovFwdTensor
true
```

See also [`AdjointCovFwdTensor`](@ref).
"""
struct CovFwdTensor{T,R<:AbstractBlockMatrix{T}} <: AbstractBlockTensor{T}
    F::R
    workspace::Matrix{T}

    # Inner Constructor
    function CovFwdTensor(F_input::R) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(F_input) == T "Element type mismatch: F has eltype $(eltype(F_input)), but expected $T."
        workspace = Matrix{T}(undef, maximum(blocksizes(F_input, 1)), maximum(blocksizes(F_input, 2)))
        return new{T,R}(F_input, workspace)
    end
    function CovFwdTensor(F_input::R, workspace::Matrix{T}) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(F_input) == T "Element type mismatch: F has eltype $(eltype(F_input)), but expected $T."
        @assert size(workspace, 1) >= maximum(blocksizes(F_input, 1)) "workspace has insufficient rows for F."
        @assert size(workspace, 2) >= maximum(blocksizes(F_input, 2)) "workspace has insufficient columns for F."

        return new{T,R}(F_input, workspace)
    end
end
Base.eltype(::Type{<:CovFwdTensor{T}}) where {T} = T
Base.size(L::CovFwdTensor) = (sum(blocksizes(L.F, 1) .^ 2), sum(blocksizes(L.F, 2) .^ 2))
Base.size(L::CovFwdTensor, d::Int) = size(L)[d]


"""
    AdjointBlockOuter(E, [workspace])

Represents the adjoint of the `BlockOuter` operator.

This type is typically not constructed directly, but rather by taking the adjoint of a
`BlockOuter` object (e.g., `L'`).

The operator `L'` is defined by its action on a `BlockDiagonal` matrix `A`:
`A = L'(B)`, where the i-th block of `A` is computed as
`A_i = ∑ⱼ Eⱼᵢ' Bⱼ Eᵢⱼ`.

# Fields
- `F::AbstractBlockMatrix`: The block matrix that defines the original operator.
- `workspace::Matrix`: Pre-allocated matrix to be used for intermediate calculations.

# Examples
```julia-repl
julia> E = BlockMatrix(rand(3, 2), [2, 1], [1, 1]);

julia> L_adj = (E ⊙ E)';

julia> L_adj isa AdjointBlockOuter
true
```

See also [`BlockOuter`](@ref).
"""
struct AdjointBlockOuter{T,R<:AbstractBlockMatrix{T}} <: AbstractBlockTensor{T}
    E::R
    workspace::Matrix{T}

    # Inner Constructor
    function AdjointBlockOuter(E_input::R) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(E_input) == T "Element type mismatch: F has eltype $(eltype(E_input)), but expected $T."
        workspace = Matrix{T}(undef, maximum(blocksizes(E_input, 1)), maximum(blocksizes(E_input, 2)))
        return new{T,R}(E_input, workspace)
    end
    function AdjointBlockOuter(E_input::R, workspace::Matrix{T}) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(E_input) == T "Element type mismatch: F has eltype $(eltype(E_input)), but expected $T."
        @assert size(workspace, 1) >= maximum(blocksizes(E_input, 1)) "workspace has insufficient rows for F."
        @assert size(workspace, 2) >= maximum(blocksizes(E_input, 2)) "workspace has insufficient columns for F."
        return new{T,R}(E_input, workspace)
    end
end
Base.eltype(::Type{<:AdjointBlockOuter{T}}) where {T} = T
Base.size(L::AdjointBlockOuter) = (sum(blocksizes(L.E, 2) .^ 2), sum(blocksizes(L.E, 1) .^ 2))
Base.size(L::AdjointBlockOuter, d::Int) = size(L)[d]

Base.adjoint(L::BlockOuter) = AdjointBlockOuter(L.E, L.workspace)
Base.adjoint(L::AdjointBlockOuter) = BlockOuter(L.E, L.workspace)


"""
    AdjointCovFwdTensor(F, [workspace])

Represents the adjoint of the `CovFwdTensor` operator.

# Fields
- `F::AbstractBlockMatrix`: The block matrix that defines the core of the operator.
- `workspace::Matrix`: Pre-allocated matrix to be used for intermediate calculations.

# Examples
```julia-repl
julia> F = BlockMatrix(rand(3, 2), [2, 1], [1, 1]);

julia> L = CovFwdTensor(F);

julia> L_adj = L';

julia> L_adj isa AdjointCovFwdTensor
true

julia> L_adj' === L
true
```

See also [`CovFwdTensor`](@ref).
"""
struct AdjointCovFwdTensor{T,R<:AbstractBlockMatrix{T}} <: AbstractBlockTensor{T}
    F::R
    workspace::Matrix{T}

    # Inner Constructor
    function AdjointCovFwdTensor(F_input::R) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(F_input) == T "Element type mismatch: F has eltype $(eltype(F_input)), but expected $T."
        workspace = Matrix{T}(undef, maximum(blocksizes(F_input, 1)), maximum(blocksizes(F_input, 2)))
        return new{T,R}(F_input, workspace)
    end
    function AdjointCovFwdTensor(F_input::R, workspace::Matrix{T}) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(F_input) == T "Element type mismatch: F has eltype $(eltype(F_input)), but expected $T."
        @assert size(workspace, 1) >= maximum(blocksizes(F_input, 1)) "workspace has insufficient rows for F."
        @assert size(workspace, 2) >= maximum(blocksizes(F_input, 2)) "workspace has insufficient columns for F."

        return new{T,R}(F_input, workspace)
    end
end
Base.eltype(::Type{<:AdjointCovFwdTensor{T}}) where {T} = T
Base.size(L::AdjointCovFwdTensor) = (sum(blocksizes(L.F, 2) .^ 2), sum(blocksizes(L.F, 1) .^ 2))
Base.size(L::AdjointCovFwdTensor, d::Int) = size(L)[d]
Base.adjoint(L::CovFwdTensor) = AdjointCovFwdTensor(L.F, L.workspace)
Base.adjoint(L::AdjointCovFwdTensor) = CovFwdTensor(L.F, L.workspace)


# --------------------------------------------------------------------
#                      SHOW METHODS
# --------------------------------------------------------------------
function Base.show(io::IO, mime::MIME"text/plain", L::BlockOuter)
    println(io, "Block Outer Product Tensor: L = E ⊙ E")
    println(io, " Action: B = L(A), where B[j] = ∑_i E[j, i] A[i] E[i, j]'")
    print(io, "E: ")
    show(io, mime, L.E)
end

function Base.show(io::IO, mime::MIME"text/plain", L::AdjointBlockOuter)
    println(io, "Adjoint Block Outer Product Tensor: L = (E ⊙ E)'")
    println(io, " Action: A = L(B), where A = ∑_j E[j]' B[j] E[j]")
    print(io, "E: ")
    show(io, mime, L.E)
end

function Base.show(io::IO, mime::MIME"text/plain", L::CovFwdTensor)
    if blocksize(L.F, 2) == 1
        println(io, "Covariance Forward Tensor (BSpline): L = O * (F ⊙ F)")
        println(io, " Action: B = L(A), where B[j] = O[j](F[j] A F[j]')")
    else
        println(io, "Covariance Forward Tensor (RKHS): L = O * (F ⊙ F) * O'")
        println(io, " Action: B = L(A), where B[j] = O[j](∑_i F[j, i] O[i](A[i]) F[i, j]')")
    end
    print(io, "F: ")
    show(io, mime, L.F)
end

function Base.show(io::IO, mime::MIME"text/plain", L::AdjointCovFwdTensor)
    if blocksize(L.F, 2) == 1
        println(io, "Adjoint Covariance Forward Tensor: L = (F ⊙ F)' * O  (BSpline)")
        println(io, "Action: A = L(B), where A[i] = (∑_j F[j, i]' O[j](B[j]) F[j, i])")
        println(io, "F: ")
        show(io, mime, L.F)
    else
        println(io, "Adjoint Covariance Forward Tensor: L = O * (F ⊙ F)' * O'  (RKHS(Redundant due to symmetry))")
        println(io, "Action: A = L(B), where A[i] = O[i](∑_j F[j, i]' O[j](B[j]) F[j, i])")
        println(io, "F: ")
        show(io, mime, L.F)
    end
    print(io, "F: ")
    show(io, mime, L.F)
end

function Base.show(io::IO, mime::MIME"text/latex", L::BlockOuter)
    print(io, L"Block Outer Product Tensor: $\mathcal{L} = E \odot E$")
    print(io, raw"$$ B = \mathcal{L}(A) \implies B_j = \sum_{i} E_{j,i} A_i E_{i,j}^{\top} $$")
    print(io, "E: ")
    show(io, mime, L.E)
end

function Base.show(io::IO, mime::MIME"text/latex", L::AdjointBlockOuter)
    print(io, L"Adjoint Block Outer Product Tensor: $\mathcal{L} = (E \odot E)^{\top}$")
    print(io, raw"$$ A = \mathcal{L}(B) \implies A_i = \sum_{j} E_{j,i}^{\top} B_j E_{i,j} $$")
    print(io, "E: ")
    show(io, mime, L.E)
end

function Base.show(io::IO, mime::MIME"text/latex", L::CovFwdTensor)
    if blocksize(L.F, 2) == 1
        print(io, L"Covariance Forward Tensor (BSpline): $\mathcal{L} = O \circ (F \odot F)$")
        print(io, raw"$$ B = \mathcal{L}(A) \implies B_j = O_j(F_j A F_j^{\top}) $$")
    else
        print(io, L"Covariance Forward Tensor (RKHS): $\mathcal{L} = O \circ (F \odot F) \circ O^{\top}$")
        print(io, raw"$$ B_j = O_j\left(\sum_{i} F_{j,i} O_i(A_i) F_{i,j}^{\top}\right) $$")
    end
    print(io, "F: ")
    show(io, mime, L.F)
end

function Base.show(io::IO, mime::MIME"text/latex", L::AdjointCovFwdTensor)
    if blocksize(L.F, 2) == 1
        print(io, L"Adjoint Covariance Forward Tensor: $\mathcal{L} = (F \odot F)^{\top} \circ O$ (BSpline)")
        print(io, raw"$$ A = \sum_{j} F_{j}^{\top} O_j(B_j) F_{j} $$")
    else
        print(io, L"Adjoint Covariance Forward Tensor: $\mathcal{L} = O \circ (F \odot F)^{\top} \circ O^{\top}$ (RKHS(Redundant due to symmetry))")
        print(io, raw"$$ A_i = O_i\left(\sum_{j} F_{j,i}^{\top} O_j(B_j) F_{j,i}\right) $$")
    end
    print(io, "F: ")
    show(io, mime, L.F)
end

# --------------------------------------------------------------------
#                      IN-PLACE SANDWICH PRODUCT
# --------------------------------------------------------------------
function sandwich!(B::AbstractMatrix{T}, F::AbstractMatrix{T}, A::AbstractMatrix{T},
    workspace::AbstractMatrix{T}) where {T}
    n, m = size(F)

    @assert size(A) == (m, m) "Dimension mismatch for A"
    @assert size(B) == (n, n) "Dimension mismatch for B"
    temp_view = view(workspace, 1:n, 1:m)
    mul!(temp_view, F, A) # Compute F * A and store in temp_view
    mul!(B, temp_view, F', one(T), one(T)) # Compute B = F * A * F' + B * β
    return B
end


function adjoint_sandwich!(A::AbstractMatrix{T}, F::AbstractMatrix{T}, B::AbstractMatrix{T},
    workspace::AbstractMatrix{T}) where {T}
    n, m = size(F)

    @assert size(A) == (m, m) "Dimension mismatch for A"
    @assert size(B) == (n, n) "Dimension mismatch for B"
    temp_view = view(workspace, 1:n, 1:m)
    mul!(temp_view, B, F) # Compute B * F and store in temp_view
    mul!(A, F', temp_view, one(T), one(T)) # Compute A = F' * B * F + A * β
    return A
end


function blockdiagelim!(B_block::AbstractMatrix{T}) where {T}
    # Remove diagonal entries of B_block
    for i in 1:size(B_block, 1)
        B_block[i, i] = zero(T)
    end
    return B_block
end

# --------------------------------------------------------------------
#                      IN-PLACE MULTIPLICATION
# --------------------------------------------------------------------
#1. Block Outer Product
function LinearAlgebra.mul!(D::BlockDiagonal{T}, L::BlockOuter{T}, C::BlockDiagonal{T}) where {T}
    A = C.blocks
    B = D.blocks
    E = L.E
    workspace = L.workspace
    # Dimension check
    # @assert blocksizes(D, 1) == blocksizes(D, 2) == blocksizes(F, 1)
    # @assert blocksizes(C, 1) == blocksizes(C, 2) == blocksizes(F, 2)

    n1 = length(A)
    n2 = length(B)

    for j in 1:n2
        @inbounds B_block = B[j]
        B_block .= zero(T)
        for i in 1:n1
            @inbounds A_block = A[i]
            @inbounds E_block = view(E, Block(j, i))
            # Compute the sandwich product B_block += F_block * A_block * F_block'
            sandwich!(B_block, E_block, A_block, workspace)
        end
    end
    return D
end

#2. Covariance Forward Tensor
function LinearAlgebra.mul!(D::BlockDiagonal{T}, L::CovFwdTensor{T}, C::BlockDiagonal{T}) where {T}
    A = C.blocks
    B = D.blocks
    F = L.F
    workspace = L.workspace
    # Dimension check
    # @assert blocksizes(D, 1) == blocksizes(D, 2) == blocksizes(F, 1)
    # @assert blocksizes(C, 1) == blocksizes(C, 2) == blocksizes(F, 2)

    n1 = length(A)
    n2 = length(B)

    # assert that n1 is either 1 or equal to n2
    @assert n1 == 1 || n1 == n2 "Dimension mismatch: n1 should be 1 (Bspline) or equal to n2 (RKHS), got n1=$n1 and n2=$n2"

    if n1 == 1 # n1 == 1 (Bspline)
        A_block = A[1]
        for j in 1:n2
            @inbounds B_block = B[j]
            # Make this block zero matrix
            B_block .= zero(T)
            @inbounds F_block = view(F, Block(j, 1))
            # Compute the sandwich product B_block += F_block * A_block * F_block'
            sandwich!(B_block, F_block, A_block, workspace)
            #Remove Diagonal Entries of B_block
            blockdiagelim!(B_block)
        end
    else    # n1 == n2 (RKHS)
        for i in 1:n1
            @inbounds A_block = A[i]
            blockdiagelim!(A_block)  # Remove diagonal entries of A_block
        end
        for j in 1:n2
            @inbounds B_block = B[j]
            # Make this block zero matrix
            B_block .= zero(T)
            for i in 1:n1
                @inbounds A_block = A[i]
                @inbounds F_block = view(F, Block(j, i))
                # Compute the sandwich product B_block += F_block * A_block * F_block'
                sandwich!(B_block, F_block, A_block, workspace)
            end
            blockdiagelim!(B_block)  # Remove diagonal entries of B_block
        end
    end
    return D
end


#3. Adjoint of Block Outer Product
function LinearAlgebra.mul!(C::BlockDiagonal{T}, adjL::AdjointBlockOuter{T}, D::BlockDiagonal{T}) where {T}
    A = C.blocks
    B = D.blocks
    E = adjL.E
    workspace = adjL.workspace
    # Dimension check
    # @assert blocksizes(D, 1) == blocksizes(D, 2) == blocksizes(F, 1)
    # @assert blocksizes(C, 1) == blocksizes(C, 2) == blocksizes(F, 2)

    n1 = length(A)
    n2 = length(B)

    for i in 1:n1
        @inbounds A_block = A[i]
        # Make this block zero matrix
        A_block .= zero(T)
        for j in 1:n2
            @inbounds B_block = B[j]
            @inbounds E_block = view(E, Block(j, i))
            # Compute the sandwich product B_block += F_block * A_block * F_block'
            adjoint_sandwich!(A_block, E_block, B_block, workspace)
        end
    end
    return C
end

#4. Adjoint of Covariance Forward Tensor
function LinearAlgebra.mul!(C::BlockDiagonal{T}, adjL::AdjointCovFwdTensor{T}, D::BlockDiagonal{T}) where {T}
    A = C.blocks
    B = D.blocks
    F = adjL.F
    workspace = adjL.workspace
    # Dimension check
    # @assert blocksizes(D, 1) == blocksizes(D, 2) == blocksizes(F, 1)
    # @assert blocksizes(C, 1) == blocksizes(C, 2) == blocksizes(F, 2)

    n1 = length(A)
    n2 = length(B)

    # assert that n1 is either 1 or equal to n2
    @assert n1 == 1 || n1 == n2 "Dimension mismatch: n1 should be 1 (Bspline) or equal to n2 (RKHS), got n1=$n1 and n2=$n2"

    if n1 == 1 # n1 == 1 (Bspline)
        @inbounds A_block = A[1]
        # Make this block zero matrix
        A_block .= zero(T)
        for j in 1:n2
            @inbounds B_block = B[j]
            blockdiagelim!(B_block)  # Remove diagonal entries of B_block
            @inbounds F_block = view(F, Block(j, 1))
            # Compute the sandwich product A_block += F_block' * A_block * F_block
            adjoint_sandwich!(A_block, F_block, B_block, workspace)
        end
    else    # n1 == n2 (RKHS)
        for j in 1:n2
            @inbounds B_block = B[j]
            blockdiagelim!(B_block)  # Remove diagonal entries of B_block
        end
        for i in 1:n1
            @inbounds A_block = A[i]
            # Make this block zero matrix
            A_block .= zero(T)
            for j in 1:n2
                @inbounds B_block = B[j]
                @inbounds F_block = view(F, Block(j, i))
                # Compute the sandwich product B_block += F_block * A_block * F_block'
                adjoint_sandwich!(A_block, F_block, B_block, workspace)
            end
            blockdiagelim!(A_block)  # Remove diagonal entries of A_block
        end
    end
    return C
end


