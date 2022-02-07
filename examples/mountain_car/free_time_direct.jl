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
nθ = 1

# ## horizon 
T = 201

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
         DTO.Dynamics(f1, nx + nθ, nx, nu + nθ), 
        [DTO.Dynamics(ft, nx + nθ, nx + nθ, nu) for t = 2:(T-1)]...
      ]
        
# ## initialization
x1 = [-π / 6.0; 0.0] 

# ## objective 
function o1(x, u, w) 
    h = u[2]
    return h + 1.0e-4 * dot(u, u)
end

function ot(x, u, w) 
    h = x[3] 
    return h + 1.0e-4 * dot(u, u)
end

function oT(x, u, w) 
    return 0.0
end

c1 = DTO.Cost(o1, nx, nu + nθ)
ct = DTO.Cost(ot, nx + nθ, nu)
cT = DTO.Cost(oT, nx + nθ, 0)
obj = [c1, [ct for t = 2:T-1]..., cT]

# ## constraints
ul = -1.0 * ones(nu) 
uu = 1.0 * ones(nu)
xl = [-1.2; -0.07] 
xu = [0.6; 0.07]
xTl = [0.6; -0.07]
xTu = [1.0; 0.07]
h0 = 1.0
hl = 0.1
hu = 2.0
bnd1 = DTO.Bound(nx, nu + nθ, xl=x1, xu=x1, ul=[ul; hl], uu=[uu; hu])
bndt = DTO.Bound(nx + nθ, nu, xl=[xl; hl], xu=[xu; hu], ul=ul, uu=uu)
bndT = DTO.Bound(nx + nθ, 0, xl=[xTl; hl], xu=[xTu; hu])
bnds = [bnd1, [bndt for t = 2:T-1]..., bndT]

cons = [DTO.Constraint() for t = 1:T]

# ## problem 
p = DTO.solver(dyn, obj, cons, bnds, 
    options=DTO.Options(tol=1.0e-2, constr_viol_tol=1.0e-2))

# ## initialize
u_guess = [t == 1 ? [1.0 * randn(nu); h0] : 1.0 * randn(nu) for t = 1:T-1]
x_guess = [t == 1 ? x1 : [x1; h0] for t = 1:T]
DTO.initialize_controls!(p, u_guess)
DTO.initialize_states!(p, x_guess)

# ## solve
@time DTO.solve!(p)

# ## solution
x_sol, u_sol = DTO.get_trajectory(p)
@show x_sol[1]
@show x_sol[T]
@show h_sol = u_sol[1][end]
@show h_sol  
@show DTO.eval_obj(obj, x_sol, [u_sol..., 0], p.nlp.trajopt.w)


# ## state
plot(hcat([x[1:nx] for x in x_sol]...)', label="", color=:orange, width=2.0)

# ## control
plot(hcat([u[1:nu] for u in u_sol]..., u_sol[end][1:nu])', linetype=:steppost)

# ## visualization 
include("visuals.jl")
vis = Visualizer()
open(vis) 
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