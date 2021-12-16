# PREAMBLE

# PKG_SETUP

# ## Setup

using DirectTrajectoryOptimization
using LinearAlgebra
using Plots

# ## satellite 
include("model.jl") 

nx = satellite.nx
nu = satellite.nu
nw = satellite.nw

# ## horizon 
T = 21

# ## model
h = 0.05
dyn = [DirectTrajectoryOptimization.Dynamics(
        (y, x, u, w) -> dynamics(satellite, h, y, x, u, w), 
        nx, nx, nu) for t = 1:T-1] 
model = DirectTrajectoryOptimization.DynamicsModel(dyn)

# ## initialization
q1 = [1.0; 0.0; 0.0; 0.0] 
qT = [0.0; 1.0; 0.0; 0.0]
x1 = [q1; q1]
xT = [qT; qT] 

# ## objective 
ot = (x, u, w) -> 1.0 * dot(x - xT, x - xT) + 1.0 * dot(u, u)
oT = (x, u, w) -> 1.0 * dot(x - xT, x - xT)
ct = DirectTrajectoryOptimization.Cost(ot, nx, nu, nw, [t for t = 1:T-1])
cT = DirectTrajectoryOptimization.Cost(oT, nx, 0, nw, [T])
obj = [ct, cT]

# ## constraints
ul = -100.0 * ones(nu) 
uu = 100.0 * ones(nu)
bnd1 = Bound(nx, nu, [1], xl=x1, xu=x1, ul=ul, uu=uu)
bndt = Bound(nx, nu, [t for t = 2:T-1], ul=ul, uu=uu)
bndT = Bound(nx, 0, [T], xl=xT, xu=xT)

cons = ConstraintSet([bnd1, bndt, bndT])

# ## problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt, 
    options=Options(
        tol=1.0e-2,
        constr_viol_tol=1.0e-2,
    ))

# ## initialize
u_guess = [0.001 * randn(nu) for t = 1:T-1]
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
q_sol = [trajopt.x[1][1:4], [x[4 .+ (1:4)] for x in trajopt.x]...]

# ## state
plot(hcat(trajopt.x...)', label = "", color = :orange, width=2.0)

# ## control
plot(hcat(trajopt.u[1:end-1]..., trajopt.u[end-1])', linetype = :steppost)

# ## visualization 
vis = Visualizer()
open(vis) 
visualize!(vis, satellite, q_sol; Δt=h)