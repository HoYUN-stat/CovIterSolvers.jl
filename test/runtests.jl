using Test
using Documenter
using CovIterSolvers

@testset "Doctests" begin
    doctest(CovIterSolvers; manual=true)
end
