using Documenter
using CovIterSolvers

DocMeta.setdocmeta!(CovIterSolvers, :DocTestSetup, quote
        using CovIterSolvers
    end; recursive=true)

makedocs(;
    modules=[CovIterSolvers],
    authors="Ho Yun <zrose0921@gmail.com>",
    sitename="CovIterSolvers.jl",
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md"
    ]
)
