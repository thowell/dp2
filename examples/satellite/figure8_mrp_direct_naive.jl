# PREAMBLE

# PKG_SETUP

# ## Setup

using DirectTrajectoryOptimization
const DTO = DirectTrajectoryOptimization
using LinearAlgebra
using Plots

# ## satellite 
include("model_mrp.jl") 

nx = satellite.nx
nu = satellite.nu
nθ = 3

# ## horizon 
T = 41

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
        (y, x, u, w) -> dynamics(satellite, h, Diagonal(u[nu .+ (1:nθ)]), y[1:nx], x[1:nx], u[1:nu], w), 
        t == T-1 ? nx + nθ : nx, nx, nu + nθ) for t = 1:T-1] 

# ## initialization
mrp1 = MRP(RotX(0.0 * π))
x1 = [mrp1.x, mrp1.y, mrp1.z, 0.0, 0.0, 0.0]
mrpT = MRP(RotX(0.0 * π))
xT = [mrpT.x, mrpT.y, mrpT.z, 0.0, 0.0, 0.0]

## tracking objective
obj = DTO.Cost{Float64}[]

for t = 1:T-1 
    function ot(x, u, w) 
        r = x[1:3]
        ω = x[3 .+ (1:3)]
        k = kinematics(satellite, r)
        J = Diagonal(u[nu .+ (1:nθ)])
        0.5 * transpose(ω) * J * ω + 0.5 * dot(u[1:nu], u[1:nu]) + 0.5 * dot(k - ref[t], k - ref[t])
    end
    push!(obj, DTO.Cost(ot, nx, nu + nθ))
end

function oT(x, u, w)
    r = x[1:3]
    ω = x[3 .+ (1:3)]
    k = kinematics(satellite, r)
    J = Diagonal(x[nx .+ (1:nθ)])
    0.5 * transpose(ω) * J * ω + 0.5 * dot(k - ref[T], k - ref[T])
end

push!(obj, DTO.Cost(oT, nx + nθ, 0))

# ## constraints
ul = -10.0 * ones(nu) 
uu = 10.0 * ones(nu)
Jl = 0.1 * diag(satellite.J) 
Ju = 10.0 * diag(satellite.J)
bnd1 = DTO.Bound(nx, nu + nθ, xl=x1, xu=x1, ul=[ul; Jl], uu=[uu; Ju])
bndt = DTO.Bound(nx, nu + nθ, ul=[ul; Jl], uu=[uu; Ju])
bndT = DTO.Bound(nx + nθ, 0, xl=[xT; Jl], xu=[xT; Ju])
bnds = [bnd1, [bndt for t = 2:T-1]..., bndT]

cons = DTO.Constraint{Float64}[]
for t = 1:T
    if t in [6, 11, 16, 21, 26, 31, 36]
        function constraint(x, u, w) 
            [
                kinematics(satellite, x[1:3]) - ref[t];
            ]
        end
        push!(cons, DTO.Constraint(constraint, nx, nu + nθ))
    else 
        push!(cons, DTO.Constraint())
    end
end

# ## general constraint
nz = T * nx + nθ + (T - 1) * (nu + nθ)
x_idx = DTO.x_indices(dyn)
u_idx = DTO.u_indices(dyn)

function inertia_constraint(z, w) 
    c = []
    θ = z[u_idx[1]][nu .+ (1:nθ)]
    for t = 2:T-1 
        push!(c, z[u_idx[t]][nu .+ (1:nθ)] - θ)
    end
    push!(c, z[x_idx[T][nx .+ (1:nθ)]] - θ)
    return vcat(c...)
end
gen_con = DTO.GeneralConstraint(inertia_constraint, nz)

# ## problem 
p = DTO.solver(dyn, obj, cons, bnds, 
        general_constraint=gen_con,
        options=Options(
            tol=1.0e-2,
            constr_viol_tol=1.0e-2))

# ## initialize
J_init = copy(diag(satellite.J))
DTO.initialize_controls!(p, [[1.0e-3 * randn(nu); J_init] for t = 1:T-1])
DTO.initialize_states!(p, [t == T ? [x1; J_init] : x1 for t = 1:T])

# ## solve
@time DTO.solve!(p)

# ## solution
x_sol, u_sol = DTO.get_trajectory(p)
θ_sol = u_sol[1][nu .+ (1:nθ)]
@show x_sol[1]
@show x_sol[T]

u_sol[T-1][nu .+ (1:nθ)]
x_sol[T][nx .+ (1:nθ)]

@show DTO.eval_obj(obj, x_sol, [u_sol..., 0], p.nlp.trajopt.w)

# ## state
plot(hcat([x[1:nx] for x in x_sol]...)', label = "", color = :orange, width=2.0)

# ## control
plot(hcat([u[1:nu] for u in u_sol]..., u_sol[end][1:nu])', linetype = :steppost)

# ## geometry 
# dimensions for visualization
A = satellite.m / 12 * [1.0 1.0 0.0; 
                        0.0 1.0 1.0; 
                        1.0 0.0 1.0]
b = θ_sol
sol = (A' * A + 15.0 * I) \ (A' * b) # set regularization to ensure physical solution (i.e., positive dimensions)
height, depth, width = sqrt.(sol)

# ## visualization 
include("visuals.jl")
vis = Visualizer()
open(vis) 
visualize_satellite!(vis, satellite, x_sol; dim=[width, depth, height], Δt=h, body_scale=0.75, orientation=:mrp)

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
