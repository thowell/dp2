# PREAMBLE

# PKG_SETUP

# ## Setup

using DirectTrajectoryOptimization
const DTO = DirectTrajectoryOptimization
using LinearAlgebra
using Plots

# ## satellite 
include("model.jl") 

nx = satellite.nx
nu = satellite.nu
nw = satellite.nw

# ## horizon 
T = 40

# ## Reference trajectory
# ##https://en.wikipedia.org/wiki/Viviani%27s_curve
# ##https://mathworld.wolfram.com/VivianisCurve.html
t = range(-2.0 * π, stop=2.0 * π, length=T)
a = 0.05
rx = a * (1.0 .+ cos.(t))
ry = a * sin.(t)
rz = 2.0 * a * sin.(0.5 .* t)

# plot(rx, rz, aspect_ratio = :equal)
# plot(rx, ry, aspect_ratio = :equal)
# plot(ry, rz, aspect_ratio = :equal)
# plot(rx, ry, rz, aspect_ratio = :equal)

ref = [RotZ(0.0 * π) * [rx[t]; ry[t]; rz[t]] for t = 1:T]

# ## model
h = 0.1
dyn = [DTO.Dynamics(
        (y, x, u, w) -> dynamics(satellite, h, satellite.J, y, x, u, w), 
        nx, nx, nu) for t = 1:T-1] 

# ## initialization
q1 = [1.0; 0.0; 0.0; 0.0] 
qT = [1.0; 0.0; 0.0; 0.0]
x1 = [q1; q1] # ω1 = 0
xT = [qT; qT] # ωT = 0

# # ## objective 
# function ot(x, u, w) 
#     q1 = x[1:4] 
#     q2 = x[4 .+ (1:4)]
#     ω = angular_velocity(h, q1, q2)
#     1.0 * transpose(ω) * satellite.J * ω + 1.0 * dot(u, u)
# end

# function oT(x, u, w)
#     q1 = x[1:4] 
#     q2 = x[4 .+ (1:4)]
#     ω = angular_velocity(h, q1, q2)
#     1.0 * transpose(ω) * satellite.J * ω
# end

# ct = DTO.Cost(ot, nx, nu, nw, [t for t = 1:T-1])
# cT = DTO.Cost(oT, nx, 0, nw, [T])
# obj = [ct, cT]

## tracking objective
obj = DTO.Cost{Float64}[]

for t = 1:T-1 
    function ot(x, u, w) 
        q1 = x[1:4] 
        q2 = x[4 .+ (1:4)]
        ω = angular_velocity(h, q1, q2)
        k = kinematics(satellite, q2)
        1.0e-3 * transpose(ω) * satellite.J * ω + 1.0e-3 * dot(u, u) + 10000.0 * dot(k - ref[t], k - ref[t])
    end
    push!(obj, DTO.Cost(ot, nx, nu, nw))
end

function oT(x, u, w)
    q1 = x[1:4] 
    q2 = x[4 .+ (1:4)]
    ω = angular_velocity(h, q1, q2)
    k = kinematics(satellite, q2)
    1.0e-3 * transpose(ω) * satellite.J * ω + 10000.0 * dot(k - ref[T], k - ref[T])
end

push!(obj, DTO.Cost(oT, nx, 0, nw))

# ## constraints
ul = -10.0 * ones(nu) 
uu = 10.0 * ones(nu)
bnd1 = DTO.Bound(nx, nu, xl=x1, xu=x1, ul=ul, uu=uu)
bndt = DTO.Bound(nx, nu, ul=ul, uu=uu)
bndT = DTO.Bound(nx, 0, xl=xT, xu=xT)
bnds = [bnd1, [bndt for t = 2:T-1]..., bndT]

cons = [DTO.Constraint() for t = 1:T]

# ## problem 
p = DTO.ProblemData(obj, dyn, cons, bnds, options=Options(
        tol=1.0e-3,
        constr_viol_tol=1.0e-3))
    
# ## initialize
DTO.initialize_controls!(p, [0.1 * randn(nu) for t = 1:T-1])
DTO.initialize_states!(p, [x1 for t = 1:T])

# ## solve
@time DTO.solve!(p)

# ## solution
x_sol, u_sol = DTO.get_trajectory(p)
@show x_sol[1]
@show x_sol[T]
q_sol = [x_sol[1][1:4], [x[4 .+ (1:4)] for x in x_sol]...]
q_vis = [[q_sol[1] for t = 1:10]..., q_sol..., [q_sol[end] for t = 1:10]...]

@show DTO.eval_obj(obj, x_sol, [u_sol..., 0], p.nlp.trajopt.w)

# ## state
plot(hcat(x_sol...)', label = "", color = :orange, width=2.0)

# ## control
plot(hcat(u_sol..., u_sol[end])', linetype = :steppost)

# ## visualization 
include("visuals.jl")
vis = Visualizer()
open(vis) 
visualize_satellite!(vis, satellite, q_vis; dim=satellite.dim, Δt=h, body_scale=0.75)

function kinematics_vis(model::Satellite, q)
	p = [0.75, 0.0, 0.0]
	k = rotation_matrix(q) * p
	return k
end
points = Vector{Point{3,Float64}}()
for t = 1:T 
    _p = kinematics_vis(satellite, x_sol[t][4 .+ (1:4)])
	push!(points, Point(_p...))
end

line_mat = LineBasicMaterial(color=RGBA(0.0, 1.0, 0.0, 1.0), linewidth=25)
setobject!(vis, MeshCat.Line(points, line_mat))

# ## ghost 
timestep = [t for t = 1:2:T]#[37, 27, 17, 5]#, 10, 15, 20, 25, 30, 35, 40]
ghost(vis, satellite, q_sol, dim=satellite.dim, timestep=timestep, transparency=[0.1 for t = 1:length(timestep)], body_scale=0.75)
