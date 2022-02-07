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
nθ = 1

# ## horizon 
T = 201

# ## model
dyn = [DTO.Dynamics((y, x, u, w) -> dynamics(mountain_car, u[nu .+ (1:nθ)], y[1:nx], x[1:nx], u[1:nu], w), nx, nx, nu + nθ) for t = 1:(T-1)]
        
# ## initialization
x1 = [-π / 6.0; 0.0] 

# ## objective 
function o1(x, u, w) 
    h = u[2]
    return h + 1.0e-4 * dot(u, u)
end

function ot(x, u, w) 
    h = u[2]
    return h + 1.0e-4 * dot(u, u)
end

function oT(x, u, w) 
    return 0.0
end

c1 = DTO.Cost(o1, nx, nu + nθ)
ct = DTO.Cost(ot, nx, nu + nθ)
cT = DTO.Cost(oT, nx, 0)
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
bndt = DTO.Bound(nx, nu + nθ, xl=xl, xu=xu, ul=[ul; hl], uu=[uu; hu])
bndT = DTO.Bound(nx, 0, xl=xTl, xu=xTu)
bnds = [bnd1, [bndt for t = 2:T-1]..., bndT]

cons = [DTO.Constraint() for t = 1:T]

# ## general constraint
nz = T * nx + (T - 1) * (nu + nθ)
x_idx = DTO.x_indices(dyn)
u_idx = DTO.u_indices(dyn)

function timestep_constraint(z, w) 
    c = []
    θ = z[u_idx[1]][nu .+ (1:nθ)]
    for t = 2:T-1 
        push!(c, z[u_idx[t]][nu .+ (1:nθ)] - θ)
    end
    return vcat(c...)
end
gen_con = DTO.GeneralConstraint(timestep_constraint, nz)

# ## problem 
p = DTO.solver(dyn, obj, cons, bnds, 
    general_constraint=gen_con,
    options=DTO.Options(tol=1.0e-2, constr_viol_tol=1.0e-2))

# ## initialize
u_guess = [[1.0 * randn(nu); h0] for t = 1:T-1]
x_guess = [x1 for t = 1:T]
DTO.initialize_controls!(p, u_guess)
DTO.initialize_states!(p, x_guess)

# ## solve
@time DTO.solve!(p)
push!(c, z[x_idx[T][nx .+ (1:nθ)]] - θ)

# ## solution
x_sol, u_sol = DTO.get_trajectory(p)
@show x_sol[1]
@show x_sol[T]
@show h_sol = [u[nu .+ (1:nθ)] for u in u_sol]
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