# PREAMBLE
# PKG_SETUP
# ## Setup

using IterativeLQR
const iLQR = IterativeLQR
using LinearAlgebra
using Plots
using JLD2

# ## mountain car
include("model.jl")
nx = mountain_car.nx
nu = mountain_car.nu
nθ = 1

# ## horizon
T = 201

# ## model
function f1(x, u, w)
    h = u[2:2]
    [
        dynamics(mountain_car, h.^2.0, x[1:nx], u[1:nu], w);
        h
    ]
end

function ft(x, u, w)
    h = x[3:3]
    [
        dynamics(mountain_car, h.^2.0, x[1:nx], u[1:nu], w);
        h
    ]
end

dyn = [
         iLQR.Dynamics(f1, nx, nu + nθ),
        [iLQR.Dynamics(ft, nx + nθ, nu) for t = 2:(T-1)]...
      ]

# ## initialization
x1 = [-π / 6.0; 0.0]
h0 = 1.0
ū = [t == 1 ? [1.0 * randn(nu); h0] : 1.0 * randn(nu) for t = 1:T-1]
x̄ = rollout(dyn, x1, ū)

visualize_mountain_car!(vis, mountain_car, x̄; mesh=true, color="metal", Δt=0.05 * h0, xl=xl[1], xu=xu[1])

# ## objective
function o1(x, u, w)
    h = u[2]^2.0
    return 1.0 * h + 1.0e-4 * u[1]^2.0 + 1.0e-4 * u[2]^2.0
end

function ot(x, u, w)
    h = x[3]^2.0
    return 1.0 * h + 1.0e-4 * u[1]^2.0
end

function oT(x, u, w)
    h = x[3]^2.0
    return 1.0 * h
end

c1 = iLQR.Cost(o1, nx, nu + nθ)
ct = iLQR.Cost(ot, nx + nθ, nu)
cT = iLQR.Cost(oT, nx + nθ, 0)
obj = [c1, [ct for t = 2:T-1]..., cT]

# ## constraints
ul = -1.0 * ones(nu)
uu = 1.0 * ones(nu)
xl = [-1.2; -0.07]
xu = [0.7; 0.07]
xTl = [0.6; -0.07]
xTu = [1.0; 0.07]
hl = 0.1
hu = 2.0

function con1(x, u, w)
    [
        ul[1] - u[1]; 
        u[1] - uu[1]; 
        hl - u[2]^2.0; 
        u[2]^2.0 - hu;
    ]
end

function cont(x, u, w)
    [
        xl - x[1:2]; 
        x[1:2] - xu;
        ul - u;
        u - uu;
        hl - x[3]^2.0; 
        x[3]^2.0 - hu;
    ]
end

function conT(x, u, w)
    [
        10.0 * (xTl - x[1:2]);
        10.0 * (x[1:2] - xTu);
        hl - x[3]^2.0;
        x[3]^2.0 - hu;
    ]
end

cons = [iLQR.Constraint(con1, nx, nu + nθ, idx_ineq=collect(1:4)),
    [iLQR.Constraint(cont, nx + nθ, nu, idx_ineq=collect(1:8)) for t = 2:T-1]...,
    iLQR.Constraint(conT, nx + nθ, nu, idx_ineq=collect(1:6))]

# ## problem
p = iLQR.problem_data(dyn, obj, cons)

ū = [t == 1 ? [1.0 * randn(nu); h0] : 1.0 * randn(nu) for t = 1:T-1]
x̄ = rollout(dyn, x1, ū)

iLQR.initialize_controls!(p, ū)
iLQR.initialize_states!(p, x̄)

# ## solve
@time iLQR.solve!(p,
    linesearch=:armijo,
    α_min=1.0e-8,
    obj_tol=1.0e-3,
    grad_tol=1.0e-3,
    con_tol=0.001,
    max_iter=250,
    max_al_iter=10,
    verbose=false)

# ## solution
x_sol, u_sol = iLQR.get_trajectory(p)
h_sol = [u_sol[1][nu .+ (1:nθ)].^2.0, [x_sol[t][nx .+ (1:nθ)].^2.0 for t = 2:T]...]

@show x_sol[1]
@show x_sol[T]
@show h_sol[1]

@show iLQR.eval_obj(obj, x_sol, [u_sol..., 0], p.m_data.w)
@show p.s_data.iter[1]
@show norm(max.(0.0, conT(p.m_data.x[T], zeros(0), zeros(0))), Inf)


x̄ = [x[1:nx] for x in x_sol]
ū = [u[1:nu] for u in u_sol]

rank(p.p_data.Quu[1])
rank(p.p_data.Qxx[2])
cond(p.p_data.Quu[1])
cond(p.p_data.Quu[3])

# @save joinpath(@__DIR__, "fixed_time.jld2") x̄ ū
# ## state
plot(hcat([x[1:nx] for x in x_sol]...)', label="", color=:orange, width=2.0)
# ## control
plot(hcat([u[1:nu] for u in u_sol]..., u_sol[end][1:nu])', linetype=:steppost)
# ## visualization
include("visuals.jl")
vis = Visualizer()
open(vis)
# x_sol = x̄
# h_sol = h0
x_vis = [[x_sol[1] for t = 1:50]..., x_sol..., [x_sol[end] for t = 1:50]...]
visualize_mountain_car!(vis, mountain_car, x_vis; mesh=true, color="metal", Δt=0.05 * h_sol[1], xl=xl[1], xu=xu[1])
# ## ghost
t = [T, 180, 170, 160, 150, 130, 110, 80, 50, 1]
ghost_mountain_car!(vis, mountain_car, x_sol; timestep=t, mesh=true, color="transparent", xl=xl[1], xu=xu[1])
# ## PGFPlots
using PGFPlots
const PGF = PGFPlots
tt = [(j - 1) * h_sol for j = 1:T] ./ (h * (T - 1))
p_free = PGF.Plots.Linear(tt, vcat([u[1:nu] for u in u_sol]..., u_sol[end][1:nu]), legendentry="free time", mark="none",style="const plot, color=cyan, line width=2pt, solid")
a1 = Axis([p_fixed; p_free],
    hideAxis=false,
    ylabel="control",
    xlabel="time",
    legendPos="south east")
dir = joinpath("/home/taylor/Research/parameter_optimization_manuscript/figures")
PGF.save(joinpath(dir, "mountain_car_control.tikz"), a1)