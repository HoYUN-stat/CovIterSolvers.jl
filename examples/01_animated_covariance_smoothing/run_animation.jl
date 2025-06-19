# Load the necessary packages
using CovIterSolvers
# using CairoMakie #v0.10.12
using GLMakie
using Printf
using Dates
using LinearAlgebra
using Random: seed!
using Statistics: median
using Krylov
import Krylov: statistics, solution

# Set the global parameters
σ::Float64 = 1e-2           #Standard deviation σ
jitter::Float64 = σ^2       #Perturbation level σ^2
g::Int64 = 500              #Resolution to plot the covariance

n::Int64 = 100              #Number of random functions
seed!(1234)                 #Seed for reproducibility
BS = fill(20, n)            #Fixed number of locations for simplicity
r = median(BS)              #Median number of locations

# --------------------------------------------------------------------
#                      Data Generation
# --------------------------------------------------------------------
# Generate random locations
loc = loc_grid(BS, seed=345);
# Generate Gaussian process sample paths
process = IntegratedBM()
@time y = sample_gp(process, loc; jitter=jitter, seed=42);
@time Y = y ⊙ y;

myeval = range(start=0.0, step=1.0 / g, length=g)
Σ_true = covariancekernel(process, myeval)


# --------------------------------------------------------------------
#          Step 1: Get Solution History via Callback
# --------------------------------------------------------------------
# Preallocate memory for the history of the solution blocks.
order = 4;  # Cubic B-splines
m = 10;  # Number of (internal) knots
p = m + order - 2
knots = range(0, 1; length=m)
myspline = BSplineMethod(order, knots)
Φ = mean_fwd(loc, myspline);
# Φ.blocks[2, 1]
Φ2 = CovFwdTensor(Φ)
E_spline = eval_fwd(myeval, myspline)
E_spline_mat = Matrix(E_spline)


Σ_blocks = Vector{Matrix{Float64}}()

# --- Define the Callback Function ---
function store_solution_callback(workspace)
    k = workspace_spline.stats.niter
    A_block = workspace_spline.x.blocks[1]
    Σ_block = E_spline_mat * A_block * E_spline_mat'
    push!(Σ_blocks, Σ_block)

    return false  # We don't want to add new stopping conditions
end

println("Running LSQR with callback to capture solution history efficiently...")

# Set up the workspace
A_zero = zero_block_diag([p]);
kc_spline = KrylovConstructor(Y, A_zero);
workspace_spline = LsqrWorkspace(kc_spline);

# Call lsqr!
lsqr!(workspace_spline, Φ2, Y; history=true, callback=store_solution_callback);
stats_spline = statistics(workspace_spline)
n_iterations = stats_spline.niter

# --------------------------------------------------------------------
#          Step 2: Create the Makie Animation
# --------------------------------------------------------------------
# Determine color limits based on true and final matrices for a stable colorbar
vmin = min(minimum(Σ_true))
vmax = max(maximum(Σ_true))
clims = (vmin, vmax)

# Set up the figure and observables for animation
fig = Figure(size=(1600, 800), fontsize=30)
iter_obs = Observable(1) # Observable for the current iteration

# The title will update automatically as iter_obs changes
title_obs = lift(i -> "Smoothed Covariance (Iteration: $i / $n_iterations)", iter_obs)

# The smoothed covariance matrix will update automatically.
Σ_spline_obs = lift(i -> Σ_blocks[i], iter_obs)

# --- Create an observable for the animated residual text ---
residual_text_obs = lift(iter_obs) do i
    res_val = stats_spline.residuals[i]
    ares_val = stats_spline.Aresiduals[i]
    # --- Format to 2 decimal places ---
    @sprintf("Residual: %.2e\nA-Residual: %.2e", res_val, ares_val)
end

# Left plot: True Covariance (static)
ax1 = Axis(fig[1, 1], xlabel="Time", ylabel="Time", title="True Covariance")
heatmap!(ax1, myeval, myeval, Σ_true, colormap=:viridis, colorrange=clims)

# Right plot: Smoothed Covariance (animated)
ax2 = Axis(fig[1, 2], xlabel="Time", ylabel="Time", title=title_obs)
p_anim = heatmap!(ax2, myeval, myeval, Σ_spline_obs, colormap=:viridis, colorrange=clims)
Colorbar(fig[1, 3], p_anim)

# --- Add the animated text to the bottom-left of the right plot (ax2) ---
text!(ax2,
    0.05, 0.05,                   # Position: 5% from left, 5% from bottom
    text=residual_text_obs,     # The observable string
    align=(:left, :bottom),     # Anchor the text at its bottom-left
    space=:relative,            # Use coordinates relative to the axis frame
    fontsize=22,                # Choose a suitable font size
    color=:white                # Choose a color with good contrast against the heatmap
)

# Record the animation, updating the iteration observable on each frame
framerate = 10
compression_level = 30

# --- MODIFICATION: Define a clean, fixed output path ---
output_filename = joinpath(@__DIR__, "animated_smoothing.gif")

record(fig, output_filename, 1:n_iterations; framerate=framerate) do i
    iter_obs[] = i # This update triggers the title and heatmap to change for each frame
end

println("\nAnimation saved to $output_filename")