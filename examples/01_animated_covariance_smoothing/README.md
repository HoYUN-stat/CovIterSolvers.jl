# Example: Animated Covariance Smoothing

This example demonstrates how to use `CovIterSolvers.jl` to perform covariance smoothing with cubic B-splines.

It visualizes the convergence of the solution over each iteration of the `lsqr` solver by generating an animation.

## How to Run This Example

1.  Navigate to this directory in your terminal.
2.  Launch Julia and activate the local environment:
    ```sh
    julia --project=.
    ```
3.  From the Julia REPL, run the script:
    ```julia
    include("run_animation.jl")
    ```
This will generate the output video `animated_smoothing.gif` in this directory.

## Result

Here is the final animation showing the true covariance matrix (left) and the smoothed estimate converging over the solver iterations (right).

![Covariance Smoothing Animation](animated_smoothing.gif)

The animation demonstrates that full iteration is often unnecessary. 

In the next example `02_lcurve`, we illustrate how to draw the L-curve to perform early termination.