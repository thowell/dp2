# PREAMBLE

# PKG_SETUP

# ## Setup

using DirectTrajectoryOptimization
const DTO = DirectTrajectoryOptimization
using LinearAlgebra
using Plots

# ## mountain car 
include("model.jl") 

nx = mountain_car.nx
nu = mountain_car.nu
nw = mountain_car.nw

# ## horizon 
T = 201

# ## model
h = 1.0
dyn = [DTO.Dynamics(
        (y, x, u, w) -> dynamics(mountain_car, h, y, x, u, w), 
        nx, nx, nu) for t = 1:T-1] 

# ## initialization
x1 = [-π / 6.0; 0.0]
xT = [0.5; 0.0]

# ## objective 
function ot(x, u, w) 
    0.0 * dot(x - xT, x - xT) + 1.0e-3 * dot(u, u)
end
function oT(x, u, w) 
    0.0 * dot(x - xT, x - xT)
end
ct = DTO.Cost(ot, nx, nu, nw)
cT = DTO.Cost(oT, nx, 0, nw)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
ul = -1.0 * ones(nu) 
uu = 1.0 * ones(nu)
xl = [-1.2; -0.07] 
xu = [0.6; 0.07]
xTl = [0.5; -0.07]
xTu = [Inf; 0.07]
bnd1 = DTO.Bound(nx, nu, xl=x1, xu=x1, ul=ul, uu=uu)
bndt = DTO.Bound(nx, nu, xl=xl, xu=xu, ul=ul, uu=uu)
bndT = DTO.Bound(nx, 0,  xl=xTl, xu=xTu)
bnds = [bnd1, [bndt for t = 2:T-1]..., bndT]

cons = [DTO.Constraint() for t = 1:T]

# ## problem 
p = DTO.ProblemData(obj, dyn, cons, bnds, 
    options=DTO.Options(tol=1.0e-3, constr_viol_tol=1.0e-3))

# ## initialize
DTO.initialize_controls!(p, [1.0 * randn(nu) for t = 1:T-1])
DTO.initialize_states!(p, [x1 for t = 1:T])

# ## solve
@time DTO.solve!(p)

# ## solution
x_sol, u_sol = DTO.get_trajectory(p)
@show x_sol[1]
@show x_sol[T]

# ## state
plot(hcat(x_sol...)', label = "", color = :orange, width=2.0)

# ## control
plot(hcat(u_sol..., u_sol[end])', linetype = :steppost)

# ## visualization 
include("visuals.jl")
vis = Visualizer()
open(vis)
x_vis = [[x_sol[1] for t = 1:50]..., x_sol..., [x_sol[end] for t = 1:50]...]
visualize_mountain_car!(vis, mountain_car, x_vis; Δt=0.05 * h, mesh=true, z_shift=0.0, xl=xl[1], xu=xu[1])

# ## PGFPlots 
using PGFPlots
const PGF = PGFPlots
tt = [(j - 1) * h for j = 1:T] ./ (h * (T - 1))
p_fixed = PGF.Plots.Linear(tt, vcat([u_sol..., u_sol[end]]...), legendentry="fixed time", mark="none",style="const plot, color=orange, line width=2pt, solid")

