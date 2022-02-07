# PREAMBLE

# PKG_SETUP

# ## Setup

using IterativeLQR
const iLQR = IterativeLQR
using LinearAlgebra
using Plots

# ## satellite 
include("model_mrp.jl") 

nx = satellite.nx
nu = satellite.nu
nw = satellite.nw

# ## horizon 
T = 41

# ## Reference trajectory
# ##https://en.wikipedia.org/wiki/Viviani%27s_curve
# ##https://mathworld.wolfram.com/VivianisCurve.html
t = range(-2.0 * π, stop=2.0 * π, length=T)
a = 0.05 # 0.375 for vis
rx = a * (1.0 .+ cos.(t))
ry = a * sin.(t)
rz = 2.0 * a * sin.(0.5 .* t)

a_vis = 0.375
rx_vis = a_vis * (1.0 .+ cos.(t))
ry_vis = a_vis * sin.(t)
rz_vis = 2.0 * a_vis * sin.(0.5 .* t)

# plot(rx, rz, aspect_ratio = :equal)
# plot(rx, ry, aspect_ratio = :equal)
# plot(ry, rz, aspect_ratio = :equal)
# plot(rx, ry, rz, aspect_ratio = :equal)

ref = [Vector(RotZ(0.0 * π) * [rx[t]; ry[t]; rz[t]]) for t = 1:T]
ref_vis = [Vector(RotZ(0.0 * π) * [rx_vis[t]; ry_vis[t]; rz_vis[t]]) for t = 1:T]

# ## model
h = 0.1
dyn = [iLQR.Dynamics(
        (x, u, w) -> dynamics(satellite, h, satellite.J, x, u, w), 
        nx, nu, nw) for t = 1:T-1] 

# ## initialization
mrp1 = MRP(RotX(0.0 * π))
x1 = [mrp1.x, mrp1.y, mrp1.z, 0.0, 0.0, 0.0]
mrpT = MRP(RotX(0.0 * π))
xT = [mrpT.x, mrpT.y, mrpT.z, 0.0, 0.0, 0.0]

# ## rollout 
ū = [1.0e-3 * randn(nu) for t = 1:T-1]
x̄ = rollout(dyn, x1, ū)
# visualize_satellite!(vis, satellite, x̄; dim=satellite.dim, Δt=h, body_scale=0.75, orientation=:mrp)

## tracking objective
obj = iLQR.Cost{Float64}[]

for t = 1:T-1 
    function ot(x, u, w) 
        r = x[1:3]
        ω = x[3 .+ (1:3)]
        k = kinematics(satellite, r)
        0.5 * transpose(ω) * satellite.J * ω + 0.5 * dot(u, u) + 0.5 * dot(k - ref[t], k - ref[t])
    end
    push!(obj, iLQR.Cost(ot, nx, nu, nw))
end

function oT(x, u, w)
    r = x[1:3]
    ω = x[3 .+ (1:3)]
    k = kinematics(satellite, r)
    0.5 * transpose(ω) * satellite.J * ω + 0.5 * dot(k - ref[T], k - ref[T])
end

push!(obj, iLQR.Cost(oT, nx, 0, nw))

# ## constraints
ul = -10.0 * ones(nu) 
uu = 10.0 * ones(nu)

cons = iLQR.Constraint{Float64}[]
for t = 1:T-1
    function constraint(x, u, w) 
        [
            ul - u;
            u - uu;
            t in [6, 11, 16, 21, 26, 31, 36] ? kinematics(satellite, x[1:3]) - ref[t] : 0.0;
        ]
    end
    push!(cons, iLQR.Constraint(constraint, nx, nu, idx_ineq=collect(1:(2nu))))
end

function goal(x, u, w)
    # k = kinematics(satellite, x[1:3])
    # k - ref[T]
    x - xT
end
push!(cons, iLQR.Constraint(goal, nx, 0))

# ## problem 
p = iLQR.problem_data(dyn, obj, cons)
    
# ## initialize
iLQR.initialize_controls!(p, ū)
iLQR.initialize_states!(p, x̄)

# ## solve
@time iLQR.solve!(p, 
    linesearch=:armijo,
    α_min=1.0e-5,
    obj_tol=1.0e-2,
    grad_tol=1.0e-2,
    con_tol=0.001,
    max_iter=25,
    max_al_iter=10,
    ρ_init=1.0,
    ρ_scale=10.0,
    verbose=false)

# ## solution
x_sol, u_sol = iLQR.get_trajectory(p)
@show x_sol[1]
@show x_sol[T]

@show iLQR.eval_obj(obj, x_sol, [u_sol..., 0], p.m_data.w)
@show p.s_data.iter[1]
@show norm(goal(p.m_data.x[T], zeros(0), zeros(0)), Inf)

# ## state
plot(hcat(x_sol...)', label = "", color = :orange, width=2.0)

# ## control
plot(hcat(u_sol..., u_sol[end])', linetype = :steppost)

# ## 
p_sol = [kinematics(satellite, x[1:3]) for x in x_sol]
plot(hcat(ref...)', color=:black, width=2.0)
plot!(hcat(p_sol...)', color=:red, width=1.0)

# ## visualization 
include("visuals.jl")
vis = Visualizer()
open(vis) 
visualize_satellite!(vis, satellite, x_sol; dim=satellite.dim, Δt=h, body_scale=0.75, orientation=:mrp)

function kinematics_vis(model::Satellite, q)
	p = [0.75, 0.0, 0.0]
	# k = rotation_matrix(q) * p
    k = Rotations.MRP(q[1:3]...) * p
	return k
end
points = Vector{Point{3,Float64}}()
for t = 1:T 
    _p = kinematics_vis(satellite, cayley(x_sol[t][3 .+ (1:3)]))
	# push!(points, Point(_p...))
    push!(points, Point(ref_vis[t]...))
end

line_mat = LineBasicMaterial(color=RGBA(0.0, 1.0, 0.0, 1.0), linewidth=25)
setobject!(vis, MeshCat.Line(points, line_mat))

# ## ghost 
timestep = [t for t = 1:2:T]#[37, 27, 17, 5]#, 10, 15, 20, 25, 30, 35, 40]
ghost(vis, satellite, q_sol, dim=satellite.dim, timestep=timestep, transparency=[0.1 for t = 1:length(timestep)], body_scale=0.75)
