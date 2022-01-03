# PREAMBLE

# PKG_SETUP

# ## Setup

using DirectTrajectoryOptimization
using LinearAlgebra
using Plots

# ## mountain car 
include("model.jl") 

nx = mountain_car.nx
nu = mountain_car.nu
nw = mountain_car.nw

# ## horizon 
T = 101

# ## model
h = 1.0
dyn = [DirectTrajectoryOptimization.Dynamics(
        (y, x, u, w) -> dynamics(mountain_car, h, y, x, u, w), 
        nx, nx, nu) for t = 1:T-1] 
model = DirectTrajectoryOptimization.DynamicsModel(dyn)

# ## initialization
x1 = [0.0; 0.0] 
xT = [0.5; 0.0]

# ## objective 
function ot(x, u, w) 
    0.0 * dot(x - xT, x - xT) + 1.0e-3 * dot(u, u)
end
function oT(x, u, w) 
    0.0 * dot(x - xT, x - xT)
end
ct = DirectTrajectoryOptimization.Cost(ot, nx, nu, nw, [t for t = 1:T-1])
cT = DirectTrajectoryOptimization.Cost(oT, nx, 0, nw, [T])
obj = [ct, cT]

# ## constraints
ul = -1.0 * ones(nu) 
uu = 1.0 * ones(nu)
xl = [-1.2; -0.07] 
xu = [0.5; 0.07]
xTl = [0.5; -0.07]
xTu = [Inf; 0.07]
bnd1 = Bound(nx, nu, [1], xl=x1, xu=x1, ul=ul, uu=uu)
bndt = Bound(nx, nu, [t for t = 2:T-1], xl=xl, xu=xu, ul=ul, uu=uu)
bndT = Bound(nx, 0, [T], xl=xTl, xu=xTu)

cons = ConstraintSet([bnd1, bndt, bndT])

# ## problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt, 
    options=Options(
        tol=1.0e-3,
        constr_viol_tol=1.0e-3))

# ## initialize
u_guess = [1.0 * randn(nu) for t = 1:T-1]
z0 = zeros(s.p.num_var)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    z0[idx] = x1
end
for (t, idx) in enumerate(s.p.trajopt.model.idx.u)
    z0[idx] = u_guess[t]
end
initialize!(s, z0)

# ## solve
@time DirectTrajectoryOptimization.solve!(s)

# ## solution
@show trajopt.x[1]
@show trajopt.x[T]

# ## state
plot(hcat(trajopt.x...)', label = "", color = :orange, width=2.0)

# ## control
plot(hcat(trajopt.u[1:end-1]..., trajopt.u[end-1])', linetype = :steppost)

# ## visualization 
vis = Visualizer()
open(vis) 
visualize!(vis, mountain_car, trajopt.x; Δt=h)