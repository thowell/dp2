using LinearAlgebra 
using DirectTrajectoryOptimization
const DTO = DirectTrajectoryOptimization
using RoboDojo 
const RD = RoboDojo 

# ## quadruped 
include("model.jl")
nc = 4
nq = RD.quadruped.nq
nx = 2 * nq
nu = RD.quadruped.nu + nc + 8 + nc + 8 + 1
nw = RD.quadruped.nw

# ## time 
T = 41 
T_fix = 5
h = 0.01

# ## initial configuration
θ1 = pi / 4.0
θ2 = pi / 4.0
θ3 = pi / 3.0

q1 = initial_configuration(RD.quadruped, θ1, θ2, θ3)
q1[2] += 0.0#
# RD.signed_distance(RD.quadruped, q1)[1:4]
vis = Visualizer()
open(vis)
RD.visualize!(vis, RD.quadruped, [q1])

# ## feet positions
pr1 = RD.quadruped_contact_kinematics[1](q1)
pr2 = RD.quadruped_contact_kinematics[2](q1)
pf1 = RD.quadruped_contact_kinematics[3](q1)
pf2 = RD.quadruped_contact_kinematics[4](q1)

strd = 2 * (pr1 - pr2)[1]
qT = Array(perm) * copy(q1)
qT[1] += 0.5 * strd

zh = 0.05

xr1 = [pr1[1] for t = 1:T]
zr1 = [pr1[2] for t = 1:T]
pr1_ref = [[xr1[t]; zr1[t]] for t = 1:T]

xf1 = [pf1[1] for t = 1:T]
zf1 = [pf1[2] for t = 1:T]
pf1_ref = [[xf1[t]; zf1[t]] for t = 1:T]

xr2_el, zr2_el = ellipse_trajectory(pr2[1], pr2[1] + strd, zh, T - T_fix)
xr2 = [[xr2_el[1] for t = 1:T_fix]..., xr2_el...]
zr2 = [[zr2_el[1] for t = 1:T_fix]..., zr2_el...]
pr2_ref = [[xr2[t]; zr2[t]] for t = 1:T]

xf2_el, zf2_el = ellipse_trajectory(pf2[1], pf2[1] + strd, zh, T - T_fix)
xf2 = [[xf2_el[1] for t = 1:T_fix]..., xf2_el...]
zf2 = [[zf2_el[1] for t = 1:T_fix]..., zf2_el...]
pf2_ref = [[xf2[t]; zf2[t]] for t = 1:T]

# tr = range(0, stop = tf, length = T)
# plot(tr, hcat(pr1_ref...)')
# plot!(tr, hcat(pf1_ref...)')

# plot(tr, hcat(pr2_ref...)')
# plot!(tr, hcat(pf2_ref...)')

# ## model
mass_matrix, dynamics_bias = RD.codegen_dynamics(RD.quadruped)
d1 = DTO.Dynamics((y, x, u, w) -> quadruped_dyn1(mass_matrix, dynamics_bias, [h], y, x, u, w), nx + nc + 1 + nx, nx, nu)
dt = DTO.Dynamics((y, x, u, w) -> quadruped_dynt(mass_matrix, dynamics_bias, [h], y, x, u, w), nx + nc + 1 + nx, nx + nc + 1 + nx, nu)
dyn = [d1, [dt for t = 2:T-1]...]

# ## objective
obj = DTO.Cost{Float64}[]

slack_penalty = 1.0e3
function obj1(x, u, w)
    u_ctrl = u[1:RD.quadruped.nu]
    q = x[RD.quadruped.nq .+ (1:RD.quadruped.nq)]

	J = 0.0 
	J += slack_penalty * u[end]
    J += 1.0e-2 * dot(u_ctrl, u_ctrl)
    J += 1.0e-3 * dot(q - qT, q - qT)
    J += 100.0 * dot(RD.quadruped_contact_kinematics[9](q)[2] - qT[2], RD.quadruped_contact_kinematics[9](q)[2] - qT[2])
    J += 100.0 * dot(RD.quadruped_contact_kinematics[10](q)[2] - qT[2], RD.quadruped_contact_kinematics[10](q)[2] - qT[2])
    return J
end
push!(obj, DTO.Cost(obj1, nx, nu, nw))

for t = 2:T-1
    function objt(x, u, w)
        u_ctrl = u[1:RD.quadruped.nu]
        q = x[RD.quadruped.nq .+ (1:RD.quadruped.nq)]

        J = 0.0 
        J += slack_penalty * u[end]
        J += 1.0e-2 * dot(u_ctrl, u_ctrl)
        J += 1.0e-3 * dot(q - qT, q - qT)
        J += 100.0 * dot(RD.quadruped_contact_kinematics[9](q)[2] - qT[2], RD.quadruped_contact_kinematics[9](q)[2] - qT[2])
        J += 100.0 * dot(RD.quadruped_contact_kinematics[10](q)[2] - qT[2], RD.quadruped_contact_kinematics[10](q)[2] - qT[2])
        J += 100.0 * sum((pr2_ref[t] - RD.quadruped_contact_kinematics[2](q)).^2.0)
        J += 100.0 * sum((pf2_ref[t] - RD.quadruped_contact_kinematics[4](q)).^2.0)

        return J
    end
    push!(obj, DTO.Cost(objt, nx + nc + 1 + nx, nu, nw))
end

function objT(x, u, w)
    q = x[RD.quadruped.nq .+ (1:RD.quadruped.nq)]

	J = 0.0 
    J += 1.0e-3 * dot(q - qT, q - qT)
    J += 100.0 * dot(RD.quadruped_contact_kinematics[9](q)[2] - qT[2], RD.quadruped_contact_kinematics[9](q)[2] - qT[2])
    J += 100.0 * dot(RD.quadruped_contact_kinematics[10](q)[2] - qT[2], RD.quadruped_contact_kinematics[10](q)[2] - qT[2])
    J += 100.0 * sum((pr2_ref[T] - RD.quadruped_contact_kinematics[2](q)).^2.0)
    J += 100.0 * sum((pf2_ref[T] - RD.quadruped_contact_kinematics[4](q)).^2.0)

    return J
end
push!(obj, DTO.Cost(objT, nx + nc + 1 + nx, 0, 0))

# control limits
ul = zeros(nu)
ul[1:RD.quadruped.nu] .= -Inf
uu = Inf * ones(nu) 

# initial configuration
# xl1 = [q1; q1]
# xu1 = [q1; q1]
xl1 = [-Inf * ones(RoboDojo.quadruped.nq); q1]
xu1 = [Inf * ones(RoboDojo.quadruped.nq); q1]

# lateral position goal
xlT = [-Inf * ones(RD.quadruped.nq); qT[1]; -Inf * ones(nq - 1 + nc + 1 + nx)]
xuT = [Inf * ones(RD.quadruped.nq); qT[1]; Inf * ones(nq - 1 + nc + 1 + nx)]

bnd1 = DTO.Bound(nx, nu, xl=xl1, xu=xu1, ul=ul, uu=uu)
bndt = DTO.Bound(nx + nc + 1 + nx, nu, ul=ul, uu=uu)
bndT = DTO.Bound(nx + nc + 1 + nx, 0, xl=xlT, xu=xuT)
bnds = [bnd1, [bndt for t = 2:T-1]..., bndT] 

# pinned feet constraints 
function pinned1(x, u, w, t) 
    q = x[1:11]
    [
        pr1_ref[t] - RD.quadruped_contact_kinematics[1](q);
        pf1_ref[t] - RD.quadruped_contact_kinematics[3](q);
    ] 
end 

function pinned2(x, u, w, t) 
    q = x[1:11]
    [
        pr2_ref[t] - RD.quadruped_contact_kinematics[2](q);
        pf2_ref[t] - RD.quadruped_contact_kinematics[4](q);
    ]
end

# loop constraints 
function loop(x, u, w) 
    nq = RD.quadruped.nq
    xT = x[1:nx] 
    x1 = x[nx + nc + 1 .+ (1:nx)] 
    e = x1 - Array(cat(perm, perm, dims = (1,2))) * xT 
    nq = RD.quadruped.nq
    return [e[2:nq]; e[nq .+ (2:nq)]]
end

cons = DTO.Constraint{Float64}[]
function constraints_1(x, u, w) 
    [
     # equality (16)
     pinned1(x, u, w, 1); 
     pinned2(x, u, w, 1);
     contact_constraints_equality(h, x, u, w); 
     # inequality (20)
     contact_constraints_inequality1(h, x, u, w);
    ]
end
push!(cons, DTO.Constraint(constraints_1, nx, nu, nw, idx_ineq=collect(16 .+ (1:20))))

for t = 2:T_fix
    function constraints_t(x, u, w) 
        [
        # equality (16)
        pinned1(x, u, w, t);
        pinned2(x, u, w, t);
        contact_constraints_equality(h, x, u, w); 
        # inequality (24)
        contact_constraints_inequalityt(h, x, u, w);
        ]
    end
    push!(cons, DTO.Constraint(constraints_t, nx + nc + 1 + nx, nu, nw, idx_ineq=collect(16 .+ (1:24))))
end

for t = (T_fix + 1):(T-1) 
    function constraints_t(x, u, w) 
        [
        # equality (12)
        pinned1(x, u, w, t);
        contact_constraints_equality(h, x, u, w); 
        # inequality (24)
        contact_constraints_inequalityt(h, x, u, w);
        ]
    end
    push!(cons, DTO.Constraint(constraints_t, nx + nc + 1 + nx, nu, nw, idx_ineq=collect(12 .+ (1:24))))
end

function constraints_T(x, u, w) 
    [
     # equality (20)
     loop(x, u, w);
     # inequality (8)
     contact_constraints_inequalityT(h, x, u, w);
    ]
end
push!(cons, DTO.Constraint(constraints_T, nx + nc + 1 + nx, nu, nw, idx_ineq=collect(20 .+ (1:8))))

# ## problem 
p = DTO.ProblemData(obj, dyn, cons, bnds, 
    options=DTO.Options(
        max_iter=2000,
        tol=1.0e-2,
        constr_viol_tol=1.0e-2))

# ## initialize
q_interp = DTO.linear_interpolation(q1, qT, T+1)
x_interp = [[q_interp[t]; q_interp[t+1]] for t = 1:T]
u_guess = [max.(0.0, 1.0e-3 * randn(nu)) for t = 1:T-1] # may need to run more than once to get good trajectory
x_guess = [t == 1 ? x_interp[t] : [x_interp[t]; max.(0.0, 1.0e-3 * randn(nc + 1)); x_interp[t-1]] for t = 1:T]

DTO.initialize_controls!(p, u_guess)
DTO.initialize_states!(p, x_guess)

# ## solve
DTO.solve!(p)

# ## solution
x_sol, u_sol = DTO.get_trajectory(p)
@show x_sol[1]
@show x_sol[T]
@show sum([u[end] for u in u_sol])
@show x_sol[T][nx + nc + 1 .+ (1:nx)] - x_sol[1]

# ## check constraints
# maximum([norm(contact_constraints_equality(h, x_sol[t], u_sol[t], zeros(0)), Inf) for t = 1:T-1])
# norm(max.(0.0, contact_constraints_inequality1(h, x_sol[1], u_sol[1], zeros(0))), Inf)
# maximum([norm(max.(0.0, contact_constraints_inequalityt(h, x_sol[t], u_sol[t], zeros(0))), Inf) for t = 2:T-1])
# norm(max.(0.0, contact_constraints_inequalityT(h, x_sol[T], zeros(0), zeros(0))), Inf)
# maximum([norm(contact_constraints_equality(h, x_sol[t], u_sol[t], zeros(0)), Inf) for t = 1:T-1])
# maximum([norm(quadruped_dyn(mass_matrix, dynamics_bias, h, x_sol[t+1], x_sol[t], u_sol[t], zeros(0)), Inf) for t = 1:T-1])

# ## visualize 
vis = Visualizer() 
open(vis)
q_vis = [x_sol[1][1:RD.quadruped.nq], [x[RD.quadruped.nq .+ (1:RD.quadruped.nq)] for x in x_sol]...]
for i = 1:3
    T = length(q_vis) - 1
    q_vis = mirror_gait(q_vis, T)
end
length(q_vis)
RD.visualize!(vis, RD.quadruped, q_vis, Δt=h)

# ## ghost
timestep = [t for t = 1:20:length(q_vis)] 
for t in timestep 
    name = "$t"
    RD.build_robot!(vis[name], RD.quadruped, color_opacity=0.1)
    RD.set_robot!(vis[name], RD.quadruped, q_vis[t])
end

