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
nθ = 1

# ## horizon 
T = 301

# ## model
function f1(y, x, u, w)
    h = u[2:2]
    [
        dynamics(mountain_car, h, y[1:nx], x[1:nx], u[1:nu], w);
        y[nx .+ (1:nθ)] - h
    ]
end

function ft(y, x, u, w)
    h = x[3:3]
    [
        dynamics(mountain_car, h, y[1:nx], x[1:nx], u[1:nu], w);
        y[nx .+ (1:nθ)] - h
    ]
end

dyn = [
         DirectTrajectoryOptimization.Dynamics(f1, nx + nθ, nx, nu + nθ), 
        [DirectTrajectoryOptimization.Dynamics(ft, nx + nθ, nx + nθ, nu) for t = 2:(T-1)]...
      ]
        
model = DirectTrajectoryOptimization.DynamicsModel(dyn)

# ## initialization
x1 = [-π / 6.0; 0.0] 

# ## objective 
function o1(x, u, w) 
    h = u[2]
    return h 
end
function ot(x, u, w) 
    h = x[3] 
    return h
end
function oT(x, u, w) 
    return 0.0
end

c1 = DirectTrajectoryOptimization.Cost(o1, nx, nu + nθ, nw, [1])
ct = DirectTrajectoryOptimization.Cost(ot, nx + nθ, nu, nw, [t for t = 2:T-1])
cT = DirectTrajectoryOptimization.Cost(oT, nx + nθ, 0, nw, [T])
obj = [c1, ct, cT]

# ## constraints
ul = -1.0 * ones(nu) 
uu = 1.0 * ones(nu)
xl = [-1.2; -0.07] 
xu = [0.6; 0.07]
xTl = [0.5; -0.07]
xTu = [Inf; 0.07]
h0 = 0.25
hl = 0.0
hu = 1.0
bnd1 = Bound(nx, nu + nθ, [1], xl=x1, xu=x1, ul=[ul; hl], uu=[uu; hu])
bndt = Bound(nx + nθ, nu, [t for t = 2:T-1], xl=[xl; hl], xu=[xu; hu], ul=ul, uu=uu)
bndT = Bound(nx + nθ, 0, [T], xl=[xTl; hl], xu=[xTu; hu])

cons = ConstraintSet([bnd1, bndt, bndT])

# ## problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt, 
    options=Options(
        tol=1.0e-2,
        constr_viol_tol=1.0e-2))

# ## initialize
u_guess = [[1.0 * randn(nu); h0], [1.0 * randn(nu) for t = 2:T-1]...]
z0 = zeros(s.p.num_var)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    if t > 1
        z0[idx] = [x1; h0] 
    else
        z0[idx] = x1 
    end
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
plot(hcat([x[1:nx] for x in trajopt.x]...)', label="", color=:orange, width=2.0)

# ## control
plot(hcat([u[1:nu] for u in trajopt.u[1:end-1]]..., trajopt.u[end-1][1:nu])', linetype=:steppost)

# ## visualization 
vis = Visualizer()
open(vis) 
visualize_mountain_car!(vis, mountain_car, trajopt.x; mesh=true, Δt=0.01)
