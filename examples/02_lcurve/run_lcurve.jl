# Load the necessary packages
using CovIterSolvers
using CairoMakie #v0.10.12
using Random: seed!
using LinearAlgebra
using BSplineKit: BSplineBasis, BSplineOrder, galerkin_matrix # For solution norm callback
using Krylov
import Krylov: statistics, solution

# Set the global parameters
σ::Float64 = 1e-2           #Standard deviation σ
jitter::Float64 = σ^2       #Perturbation level σ^2
g::Int64 = 500              #Resolution to plot the covariance
n::Int64 = 40               #Number of random functions
seed!(1234)                 #Seed for reproducibility
r = 50                      #Number of locations for each function
BS = fill(50, n)

# --------------------------------------------------------------------
#                      Data Generation
# --------------------------------------------------------------------
# Generate random locations
loc = loc_grid(BS, seed=345);
# Generate Custom Gaussian process
my_cov(s, t) = 2 * sin(π * s / 2) * sin(π * t / 2)
process = CustomGP(my_cov)
y = sample_gp(process, loc; jitter=jitter, seed=42);
Y = y ⊙ y;

myeval = range(start=0.0, step=1.0 / g, length=g);
Σ_true = covariancekernel(process, myeval);

# --------------------------------------------------------------------
#                Cubic B-Spline Covariance Smoothing
# --------------------------------------------------------------------
order = 4;  # Cubic B-splines
m = 10;  # Number of (internal) knots
p = m + order - 2
knots = range(0, 1; length=m)
myspline = BSplineMethod(order, knots)

Φ = mean_fwd(loc, myspline);
Φ2 = CovFwdTensor(Φ)

#Preallocate BlockDiagonal matrices for the covariance smoothing
A_spline = zero_block_diag([p]);

# --- Define the Callback Function for Solution norm ---
sol_norms = Vector{Float64}()
G = galerkin_matrix(BSplineBasis(BSplineOrder(order), knots))

function store_sol_norm_callback(workspace)
    k = workspace_spline.stats.niter
    A_block = workspace_spline.x.blocks[1]
    sol_norm = norm(G * A_block)
    push!(sol_norms, sol_norm)

    return false  # We don't want to add new stopping conditions
end

kc_spline = KrylovConstructor(Y, A_spline);
workspace_spline = LsqrWorkspace(kc_spline);

lsqr!(workspace_spline, Φ2, Y; history=true, callback=store_sol_norm_callback, itmax=50);
A_spline = solution(workspace_spline)
stats_spline = statistics(workspace_spline)
stats_spline.Aresiduals
sol_norms

function find_l_curve_corner(x_coords, y_coords)
    log_x = log10.(x_coords)
    log_y = log10.(y_coords)

    # Line endpoints
    p1 = (log_x[1], log_y[1])
    p_end = (log_x[end], log_y[end])

    # Vector for the line segment connecting start and end
    line_vec = (p_end[1] - p1[1], p_end[2] - p1[2])
    line_vec_norm_sq = line_vec[1]^2 + line_vec[2]^2

    # Find the perpendicular distance for each point to the line
    distances = zeros(length(log_x))
    for i in 2:(length(log_x)-1)
        point_vec = (log_x[i] - p1[1], log_y[i] - p1[2])
        cross_product = abs(point_vec[1] * line_vec[2] - point_vec[2] * line_vec[1])
        distances[i] = cross_product / sqrt(line_vec_norm_sq)
    end
    _, corner_idx = findmax(distances)

    return corner_idx
end


aresiduals = stats_spline.Aresiduals[2:end]
solution_norms = sol_norms

corner_idx = find_l_curve_corner(aresiduals, solution_norms)
corner_x = aresiduals[corner_idx]
corner_y = solution_norms[corner_idx]

fig = Figure(size=(1200, 800), fontsize=30)
ax = Axis(fig[1, 1],
    xlabel="A-residuals",
    ylabel="Solution norms",
    xscale=log10,
    yscale=log10
)

scatterlines!(ax, aresiduals, solution_norms, color=:blue, label="L-curve", markersize=12)
scatter!(ax, corner_x, corner_y,
    color=:red,
    markersize=25,
    marker=:cross,
    label="Corner (iter count = $corner_idx)"
)

axislegend(ax, position=:lb)
fig

# Save the figure
output_filename = "spline_lcurve.png"
CairoMakie.save(output_filename, fig)