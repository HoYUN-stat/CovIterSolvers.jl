using Documenter
using CovIterSolvers

DocMeta.setdocmeta!(CovIterSolvers, :DocTestSetup, quote
        using CovIterSolvers
    end; recursive=true)

makedocs(;
    modules=[CovIterSolvers],
    authors="Ho Yun <zrose0921@gmail.com>",
    repo="github.com/HoYUN-stat/CovIterSolvers.jl/blob/{commit}/{path}#{line}",
    sitename="CovIterSolvers.jl",
    format=Documenter.HTML(; canonical="https://HoYUN-stat.github.io/CovIterSolvers.jl"),
    pages=[
        "Home" => "index.md"
    ]
)

deploydocs(; repo="github.com/HoYUN-stat/CovIterSolvers.jl")