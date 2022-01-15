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
nθ = 3

# ## horizon 
T = 40

# ## Reference trajectory
# ##https://en.wikipedia.org/wiki/Viviani%27s_curve
# ##https://mathworld.wolfram.com/VivianisCurve.html
t = range(-2.0 * π, stop = 2.0 * π, length = T)
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
dyn = DTO.Dynamics{Float64}[]
dyn = [DTO.Dynamics(
        (t == 1 ? (y, x, u, w) -> [dynamics(satellite, h, Diagonal(u[nu .+ (1:nθ)]), y[1:nx], x[1:nx], u[1:nu], w); y[nx .+ (1:nθ)] -  u[nu .+ (1:nθ)]]
         : (y, x, u, w) -> [dynamics(satellite, h, Diagonal(x[nx .+ (1:nθ)]), y[1:nx], x[1:nx], u[1:nu], w); y[nx .+ (1:nθ)] -  x[nx .+ (1:nθ)]]), 
        nx + nθ, nx + (t == 1 ? 0 : nθ), nu + (t == 1 ? nθ : 0)) for t = 1:T-1] 

# ## initialization
q1 = [1.0; 0.0; 0.0; 0.0] 
qT = [1.0; 0.0; 0.0; 0.0]
x1 = [q1; q1] # ω1 = 0
xT = [qT; qT] # ωT = 0

# # ## objective 

obj = DTO.Cost{Float64}[]

for t = 1:T-1 
    function ot(x, u, w) 
        q1 = x[1:4] 
        q2 = x[4 .+ (1:4)]
        ω = angular_velocity(h, q1, q2)
        k = kinematics(satellite, q2)
        J = (t == 1 ? Diagonal(u[nu .+ (1:nθ)]) : Diagonal(x[nx .+ (1:nθ)]))
        1.0e-3 * transpose(ω) * J * ω + 1.0e-3 * dot(u, u) + 1000.0 * dot(k - ref[t], k - ref[t]) + 1.0e-5 * dot(x[1:nx] - xT, x[1:nx] - xT)
    end
    push!(obj, DTO.Cost(ot, nx + (t == 1 ? 0 : nθ), nu + (t == 1 ? nθ : 0), nw))
end

function oT(x, u, w)
    q1 = x[1:4] 
    q2 = x[4 .+ (1:4)]
    ω = angular_velocity(h, q1, q2)
    k = kinematics(satellite, q2)
    J = Diagonal(x[nx .+ (1:nθ)])
    1.0e-3 * transpose(ω) * J * ω + 1000.0 * dot(k - ref[T], k - ref[T]) + 1.0e-5 * dot(x[1:nx] - xT, x[1:nx] - xT)
end

push!(obj, DTO.Cost(oT, nx + nθ, 0, nw))

# ## constraints
ul = -10.0 * ones(nu) 
uu = 10.0 * ones(nu)
Jl = 0.1 * diag(satellite.J) 
Ju = 10.0 * diag(satellite.J)
bnd1 = DTO.Bound(nx, nu + nθ, xl=x1, xu=x1, ul=[ul; Jl], uu=[uu; Ju])
bndt = DTO.Bound(nx + nθ, nu, xl=[-Inf * ones(nx); Jl], xu=[Inf * ones(nx); Ju], ul=ul, uu=uu)
bndT = DTO.Bound(nx + nθ, 0, xl=[xT; Jl] , xu=[xT; Ju])
bnds = [bnd1, [bndt for t = 2:T-1]..., bndT]

cons = [DTO.Constraint() for t = 1:T]

# ## problem 
p = ProblemData(obj, dyn, cons, bnds, options=Options(
        tol=1.0e-3,
        constr_viol_tol=1.0e-3))

# ## initialize
J_init = copy(diag(satellite.J))
x_guess = [t == 1 ? x1 : [x1; J_init] for t = 1:T]
u_guess = [[0.1 * randn(nu); J_init], [0.1 * randn(nu) for t = 2:T-1]...]
DTO.initialize_states!(p, x_guess)
DTO.initialize_controls!(p, u_guess)

# ## solve
@time DTO.solve!(p)

# ## solution
x_sol, u_sol = DTO.get_trajectory(p)
@show θ = u_sol[1][nu .+ (1:nθ)]
@show x_sol[1]
@show x_sol[T]
q_sol = [x_sol[1][1:4], [x[4 .+ (1:4)] for x in x_sol]...]

@show DTO.eval_obj(obj, x_sol, p.nlp.trajopt.u, p.nlp.trajopt.w)

# ## state
plot(hcat([x[1:nx] for x in x_sol]...)', label="", color=:orange, width=2.0)

# ## control
plot(hcat([u[1:nu] for u in u_sol]..., u_sol[end][1:nu])', linetype=:steppost)

# ## visualization 
include("visuals.jl")
vis = Visualizer()
open(vis) 
q_vis = [[q_sol[1] for t = 1:10]..., q_sol..., [q_sol[end] for t = 1:10]...]

# dimensions for visualization
A = satellite.m / 12 * [1.0 1.0 0.0; 
                        0.0 1.0 1.0; 
                        1.0 0.0 1.0]
b = θ
sol = (A' * A + 1.15 * I) \ (A' * b) # set regularization to ensure physical solution (i.e., positive dimensions)
height, depth, width = sqrt.(sol)

visualize_satellite!(vis, satellite, q_vis; Δt=h, dim=[width, depth, height], body_scale=0.75, arrow_scale=0.25)

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

line_mat = LineBasicMaterial(color=color=RGBA(0.0, 1.0, 0.0, 1.0), linewidth=25)
setobject!(vis, MeshCat.Line(points, line_mat))

# ## ghost 
timestep = [t for t = 1:2:T]#[37, 27, 17, 5]#, 10, 15, 20, 25, 30, 35, 40]
ghost(vis, satellite, q_sol, dim=[width, depth, height], timestep=timestep, transparency=[0.1 for t = 1:length(timestep)], body_scale=0.75, arrow_scale=0.25)

