# PREAMBLE

# PKG_SETUP

# ## Setup

using IterativeLQR 
const iLQR = IterativeLQR
using LinearAlgebra
using Plots

# ## mountain car 
include("model.jl") 

nx = mountain_car.nx
nu = mountain_car.nu
nw = mountain_car.nw
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
ū = [t == 1 ? [1.0 * randn(nu); h0] : 1.0 * randn(nu) for t = 1:T-1]
w = [zeros(nw) for t = 1:T] 
x̄ = rollout(dyn, x1, ū, w)

# ## objective 
function o1(x, u, w) 
    h = u[2]^2.0
    return h + 0.0e-5 * dot(x, x) + 1.0e-5 * u[1]^2.0
end

function ot(x, u, w) 
    h = x[3]^2.0
    return h + 0.0e-5 * dot(x[1:2], x[1:2]) + 1.0e-5 * u[1]^2.0
end

function oT(x, u, w) 
    h = x[3]^2.0
    return h + 0.0e-5 * dot(x[1:2], x[1:2])
end

c1 = iLQR.Cost(o1, nx, nu + nθ, nw)
ct = iLQR.Cost(ot, nx + nθ, nu, nw)
cT = iLQR.Cost(oT, nx + nθ, 0, nw)
obj = [c1, [ct for t = 2:T-1]..., cT]

# ## constraints
ul = -1.0 * ones(nu) 
uu = 1.0 * ones(nu)
xl = [-1.2; -0.07] 
xu = [0.6; 0.07]
xTl = [0.6; -0.07]
xTu = [0.6; 0.07]
hl = 0.0
hu = 2.0^(0.5)

function con1(x, u, w) 
    [
        u[2] - 1.0;
        # [ul; hl] - u; 
        # u - [uu; hu]; 
    ]
end

function cont(x, u, w) 
    [
        x[3] - 1.0;
        # [xl; hl] - x; 
        # x - [xu; hu];
        # ul - u; 
        # u - uu;
    ]
end

function conT(x, u, w) 
    [
        x[3] - 1.0;
        # hl - x[3]; 
        # x[3] - hu;
        x[1] - 0.6;
    ]
end

cons = [iLQR.Constraint(con1, nx, nu + nθ),# idx_ineq=collect(1:4)), 
        [iLQR.Constraint(cont, nx + nθ, nu) for t = 2:T-1]..., 
            # idx_ineq=collect(1:8)) for t = 2:T-1]..., 
        iLQR.Constraint(conT, nx + nθ, nu)]#, idx_ineq=collect(1:2))]

# ## problem 
p = iLQR.problem_data(dyn, obj, cons) 
iLQR.initialize_controls!(p, ū)
iLQR.initialize_states!(p, x̄)

# m_data = p.m_data
# iLQR.reset!(m_data.model_deriv)
# iLQR.reset!(m_data.obj_deriv) 
# s_data = p.s_data
# iLQR.reset!(p.s_data)

# plot(hcat([x[1:2] for x in p.m_data.x]...)')
# plot(hcat([u[1:1] for u in p.m_data.u[1:end-1]]...)')

# iLQR.objective!(p.s_data, p.m_data, mode=:nominal)
# iLQR.derivatives!(p.m_data, mode=:nominal)
# iLQR.backward_pass!(p.p_data, p.m_data, mode=:nominal)

# ## solve
@time iLQR.solve!(p,
    max_iter=500,
    max_al_iter=5,
    verbose=true)

# ## solution
x_sol, u_sol = iLQR.get_trajectory(p)
@show x_sol[1]
@show x_sol[T]
@show h_sol = u_sol[1][end]^2.0
@show h_sol 

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