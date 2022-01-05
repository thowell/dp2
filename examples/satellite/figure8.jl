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
T = 40

# ## Reference trajectory
# ##https://en.wikipedia.org/wiki/Viviani%27s_curve
# ##https://mathworld.wolfram.com/VivianisCurve.html
t = range(-2.0 * π, stop = 2.0 * π, length = T)
a = 0.05
xf = a * (1.0 .+ cos.(t))
yf = a * sin.(t)
zf = 2.0 * a * sin.(0.5 .* t)

# plot(xf, zf, aspect_ratio = :equal)
# plot(xf, yf, aspect_ratio = :equal)
# plot(yf, zf, aspect_ratio = :equal)
# plot(xf, yf, zf, aspect_ratio = :equal)

pf = [RotZ(0.0 * π) * [xf[t]; yf[t]; zf[t]] for t = 1:T]

# ## model
h = 0.1
dyn = [DirectTrajectoryOptimization.Dynamics(
        (y, x, u, w) -> dynamics(satellite, h, satellite.J, y, x, u, w), 
        nx, nx, nu) for t = 1:T-1] 
model = DirectTrajectoryOptimization.DynamicsModel(dyn)

# ## initialization
q1 = [1.0; 0.0; 0.0; 0.0] 
qT = [1.0; 0.0; 0.0; 0.0]
# _qT = UnitQuaternion(RotX(1.0) * RotY(1.0) * RotZ(0.0))
# qT = [_qT.w; _qT.x; _qT.y; _qT.z]
x1 = [q1; q1]
xT = [qT; qT] 

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

# ct = DirectTrajectoryOptimization.Cost(ot, nx, nu, nw, [t for t = 1:T-1])
# cT = DirectTrajectoryOptimization.Cost(oT, nx, 0, nw, [T])
# obj = [ct, cT]

## tracking objective
obj = DirectTrajectoryOptimization.Cost{Float64}[]

for t = 1:T-1 
    function ot(x, u, w) 
        q1 = x[1:4] 
        q2 = x[4 .+ (1:4)]
        ω = angular_velocity(h, q1, q2)
        k = kinematics(satellite, q2)
        1.0e-3 * transpose(ω) * satellite.J * ω + 1.0e-3 * dot(u, u) + 10000.0 * dot(k - pf[t], k - pf[t])
    end
    push!(obj, DirectTrajectoryOptimization.Cost(ot, nx, nu, nw, [t]))
end

function oT(x, u, w)
    q1 = x[1:4] 
    q2 = x[4 .+ (1:4)]
    ω = angular_velocity(h, q1, q2)
    k = kinematics(satellite, q2)
    1.0e-3 * transpose(ω) * satellite.J * ω + 10000.0 * dot(k - pf[T], k - pf[T])
end

push!(obj, DirectTrajectoryOptimization.Cost(oT, nx, 0, nw, [T]))

# ## constraints
ul = -10.0 * ones(nu) 
uu = 10.0 * ones(nu)
bnd1 = Bound(nx, nu, [1], xl=x1, xu=x1, ul=ul, uu=uu)
bndt = Bound(nx, nu, [t for t = 2:T-1], ul=ul, uu=uu)
bndT = Bound(nx, 0, [T], xl=xT, xu=xT)


# ## figure 8
# stage_cons = DirectTrajectoryOptimization.StageConstraint{Float64}[]
# ϵ = 1.0e-2
# for t = 1:T-1
#     function figure8t(x, u, w) 
#         q2 = x[4 .+ (1:4)]
#         k = kinematics(satellite, q2)
#         e = k - pf[t]

#         [
#             -ϵ .- e; 
#             e .- ϵ;
#         ]
#     end
#     push!(stage_cons, StageConstraint(figure8t, nx, nu, nw, [t], :inequality))
# end

# function figure8T(x, u, w) 
#     q2 = x[4 .+ (1:4)]
#     k = kinematics(satellite, q2)
#     e = k - pf[T]
#     [
#         -ϵ .- e; 
#         e .- ϵ;
#     ]
# end
# push!(stage_cons, StageConstraint(figure8T, nx, nu, nw, [T], :inequality))

cons = ConstraintSet([bnd1, bndt, bndT])

# ## problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt, 
    options=Options(
        tol=1.0e-3,
        constr_viol_tol=1.0e-3,
    ))

# ## initialize
u_guess = [0.1 * randn(nu) for t = 1:T-1]
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
[norm(q) for q in q_sol]

@show DirectTrajectoryOptimization.eval_obj(obj, trajopt.x, trajopt.u, trajopt.w)

# ## state
plot(hcat(trajopt.x...)', label = "", color = :orange, width=2.0)

# ## control
plot(hcat(trajopt.u[1:end-1]..., trajopt.u[end-1])', linetype = :steppost)

# ## visualization 
include("visuals.jl")
vis = Visualizer()
open(vis) 
visualize_satellite!(vis, satellite, q_sol; Δt=h)

function kinematics_vis(model::Satellite, q)
	p = [0.75, 0.0, 0.0]
	k = rotation_matrix(q) * p
	return k
end
points = Vector{Point{3,Float64}}()
for t = 1:T 
    _p = kinematics_vis(satellite, trajopt.x[t][4 .+ (1:4)])
	push!(points, Point(_p...))
end

line_mat = LineBasicMaterial(color=color=RGBA(1.0, 1.0, 1.0, 1.0), linewidth=5)
setobject!(vis[:figure8], MeshCat.Line(points, line_mat))
