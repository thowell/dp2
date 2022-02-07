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

# ## horizon 
T = 201
h = 1.0

# ## model
dyn = [iLQR.Dynamics((x, u, w) -> dynamics(mountain_car, h, x, u, w), nx, nu) for t = 1:T-1]
        
# ## initialization
x1 = [-π / 6.0; 0.0]
ū = [1.0 * randn(nu) for t = 1:T-1]
x̄ = rollout(dyn, x1, ū)

# ## objective 
function o1(x, u, w) 
    1.0e-4 * u[1]^2.0
end

function ot(x, u, w) 
    1.0e-4 * u[1]^2.0
end

function oT(x, u, w) 
   return 0.0
end

c1 = iLQR.Cost(o1, nx, nu)
ct = iLQR.Cost(ot, nx, nu)
cT = iLQR.Cost(oT, nx, 0)
obj = [c1, [ct for t = 2:T-1]..., cT]

# ## constraints
ul = -1.0 * ones(nu) 
uu = 1.0 * ones(nu)
xl = [-1.2; -0.07] 
xu = [0.6; 0.07]
xTl = [0.6; -0.07]
xTu = [1.0; 0.07]

function cont(x, u, w) 
    [
        xl - x; 
        x - xu;
        ul - u; 
        u - uu;
    ]
end

function conT(x, u, w) 
    [
       xTl - x; 
       x - xTu;
    ]
end

cons = [
        [iLQR.Constraint(cont, nx, nu, idx_ineq=collect(1:6)) for t = 1:T-1]..., 
        iLQR.Constraint(conT, nx, nu, idx_ineq=collect(1:4))
       ]

# ## problem 
p = iLQR.problem_data(dyn, obj, cons) 
iLQR.initialize_controls!(p, ū)
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
@show x_sol[1]
@show x_sol[T]

@show iLQR.eval_obj(obj, x_sol, [u_sol..., 0], p.m_data.w)
@show p.s_data.iter[1]
@show norm(max.(0.0, conT(p.m_data.x[T], zeros(0), zeros(0))), Inf)

# ## state
plot(hcat([x[1:nx] for x in x_sol]...)', label="", color=:orange, width=2.0)

# ## control
plot(hcat([u[1:nu] for u in u_sol]..., u_sol[end][1:nu])', linetype=:steppost)

# ## visualization 
include("visuals.jl")
vis = Visualizer()
open(vis) 

# x_sol = x̄
x_vis = [[x_sol[1] for t = 1:50]..., x_sol..., [x_sol[end] for t = 1:50]...]
visualize_mountain_car!(vis, mountain_car, x_vis; mesh=true, color="metal", Δt=0.05 * h_sol, xl=xl[1], xu=xu[1])

# ## save solution 
x = x_sol 
u = u_sol
@save joinpath(@__DIR__, "fixed_time_ilqr.jld2") x u

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