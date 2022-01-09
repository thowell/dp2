using RoboDojo 
using LinearAlgebra 
using DirectTrajectoryOptimization 

# ## quadruped 
include("model.jl")
nc = 4
nq = RoboDojo.quadruped.nq
nx = 2 * nq
nu = RoboDojo.quadruped.nu + nc + 8 + nc + 8 + 1
nw = RoboDojo.quadruped.nw


# ## time 
T = 31 
T_fix = 5
h = 0.01

# ## initial configuration
θ1 = pi / 4.0
θ2 = pi / 4.0
θ3 = pi / 3.0

q1 = initial_configuration(RoboDojo.quadruped, θ1, θ2, θ3)
q1[2] += 0.0
# RoboDojo.signed_distance(RoboDojo.quadruped, q1)[1:4]
vis = Visualizer()
open(vis)
RoboDojo.visualize!(vis, RoboDojo.quadruped, [q1])

# ## feet positions
pr1 = RoboDojo.quadruped_contact_kinematics[1](q1)
pr2 = RoboDojo.quadruped_contact_kinematics[2](q1)
pf1 = RoboDojo.quadruped_contact_kinematics[3](q1)
pf2 = RoboDojo.quadruped_contact_kinematics[4](q1)

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
mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.quadruped)
# d = DirectTrajectoryOptimization.Dynamics((y, x, u, w) -> quadruped_dyn(mass_matrix, dynamics_bias, [h], y, x, u, w), nx, nx, nu)
# dyn = [d for t = 1:T-1]

d1 = DirectTrajectoryOptimization.Dynamics((y, x, u, w) -> quadruped_dyn1(mass_matrix, dynamics_bias, [h], y, x, u, w), nx + nc + 1 + nx, nx, nu)
dt = DirectTrajectoryOptimization.Dynamics((y, x, u, w) -> quadruped_dynt(mass_matrix, dynamics_bias, [h], y, x, u, w), nx + nc + 1 + nx, nx + nc + 1 + nx, nu)

dyn = [d1, [dt for t = 2:T-1]...]
model = DirectTrajectoryOptimization.DynamicsModel(dyn)

# ## initial conditions

# ## objective
slack_penalty = 1.0e4
function obj1(x, u, w)
    u_ctrl = u[1:RoboDojo.quadruped.nu]
    q = x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)]

	J = 0.0 
	J += slack_penalty * u[end]
    J += 1.0e-2 * dot(u_ctrl, u_ctrl)
    J += 1.0e-3 * dot(q - qT, q - qT)
    J += 100.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
    J += 100.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
    return J
end
c1 = DirectTrajectoryOptimization.Cost(obj1, nx, nu, nw, [1])

stage_costs = DirectTrajectoryOptimization.Cost{Float64}[]
for t = 2:T-1
    function objt(x, u, w)
        u_ctrl = u[1:RoboDojo.quadruped.nu]
        q = x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)]

        J = 0.0 
        J += slack_penalty * u[end]
        J += 1.0e-2 * dot(u_ctrl, u_ctrl)
        J += 1.0e-3 * dot(q - qT, q - qT)
        J += 100.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
        J += 100.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
        J += 1000.0 * sum((pr2_ref[t] - RoboDojo.quadruped_contact_kinematics[2](q)).^2.0)
        J += 1000.0 * sum((pf2_ref[t] - RoboDojo.quadruped_contact_kinematics[4](q)).^2.0)

        return J
    end
    push!(stage_costs, DirectTrajectoryOptimization.Cost(objt, nx + nc + 1 + nx, nu, nw, [t]))
end

function objT(x, u, w)
    q = x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)]

	J = 0.0 
    J += 1.0e-3 * dot(q - qT, q - qT)
    J += 10.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
    J += 10.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
    J += 1000.0 * sum((pr2_ref[T] - RoboDojo.quadruped_contact_kinematics[2](q)).^2.0)
    J += 1000.0 * sum((pf2_ref[T] - RoboDojo.quadruped_contact_kinematics[4](q)).^2.0)

    return J
end

cT = DirectTrajectoryOptimization.Cost(objT, nx + nc + 1 + nx, 0, 0, [T])
obj = [c1, stage_costs..., cT]

# control limits
ul = zeros(nu)
ul[1:RoboDojo.quadruped.nu] .= -Inf
uu = Inf * ones(nu) 

# initial configuration
# xl1 = [-Inf * ones(RoboDojo.quadruped.nq); q1]
# xu1 = [Inf * ones(RoboDojo.quadruped.nq); q1] 
xl1 = [q1; q1]
xu1 = [q1; q1] 

# lateral position goal
xlT = [-Inf * ones(RoboDojo.quadruped.nq); qT[1]; -Inf * ones(nq - 1 + nc + 1 + nx)]
xuT = [Inf * ones(RoboDojo.quadruped.nq); qT[1]; Inf * ones(nq - 1 + nc + 1 + nx)]

bnd1 = DirectTrajectoryOptimization.Bound(nx, nu, [1], xl=xl1, xu=xu1, ul=ul, uu=uu)
bndt = DirectTrajectoryOptimization.Bound(nx + nc + 1 + nx, nu, [t for t = 2:T-1], ul=ul, uu=uu)
bndT = DirectTrajectoryOptimization.Bound(nx + nc + 1 + nx, 0, [T], xl=xlT, xu=xuT)

# con_eq1 = DirectTrajectoryOptimization.StageConstraint(stage1_eq, nx, nu, nw, [1], :equality)
# conT_eq = DirectTrajectoryOptimization.StageConstraint(terminal_con_eq, 2 * nx, nu, nw, [T], :equality)
# conT_ineq = DirectTrajectoryOptimization.StageConstraint(terminal_con_ineq, 2 * nx, nu, nw, [T], :inequality)

# pinned feet constraints 
pin_con = DirectTrajectoryOptimization.StageConstraint{Float64}[]

function pinned1!(x, u, w) 
    q = x[1:11]
    [
        pr1_ref[1] - RoboDojo.quadruped_contact_kinematics[1](q);
        pf1_ref[1] - RoboDojo.quadruped_contact_kinematics[3](q);
    ] 
end 
push!(pin_con, DirectTrajectoryOptimization.StageConstraint(pinned1!, nx, nu, nw, [1], :equality))

for t = 2:T
    function pinned1t!(x, u, w) 
        q = x[1:11]
        [
            pr1_ref[t] - RoboDojo.quadruped_contact_kinematics[1](q);
            pf1_ref[t] - RoboDojo.quadruped_contact_kinematics[3](q);
        ] 
    end
    push!(pin_con, DirectTrajectoryOptimization.StageConstraint(pinned1t!, nx + nc + 1 + nx, nu, nw, [t], :equality))
end

function pinned21!(x, u, w) 
    q = x[1:11]
    [
        pr2_ref[1] - RoboDojo.quadruped_contact_kinematics[2](q);
        pf2_ref[1] - RoboDojo.quadruped_contact_kinematics[4](q);
    ]
end
push!(pin_con, DirectTrajectoryOptimization.StageConstraint(pinned21!, nx, nu, nw, [1], :equality))

for t = 2:T_fix 
    function pinned2t!(x, u, w) 
        q = x[1:11]
        [
            pr2_ref[t] - RoboDojo.quadruped_contact_kinematics[2](q);
            pf2_ref[t] - RoboDojo.quadruped_contact_kinematics[4](q);
        ]
    end
    push!(pin_con, DirectTrajectoryOptimization.StageConstraint(pinned2t!, nx + nc + 1 + nx, nu, nw, [t], :equality))    
end

# contact constraints
contact_ineq1 = StageConstraint((x, u, w) -> contact_constraints_inequality1(h, x, u, w), nx, nu, nw, [1], :inequality)
contact_ineqt = StageConstraint((x, u, w) -> contact_constraints_inequalityt(h, x, u, w), nx + nc + 1 + nx, nu, nw, [t for t = 2:T-1], :inequality)
contact_ineqT = StageConstraint((x, u, w) -> contact_constraints_inequalityT(h, x, u, w), nx + nc + 1 + nx, nu, nw, [T], :inequality)

contact_eq1 = StageConstraint((x, u, w) -> contact_constraints_equality(h, x, u, w), nx, nu, nw, [1], :equality)
contact_eqt = StageConstraint((x, u, w) -> contact_constraints_equality(h, x, u, w), nx + nc + 1 + nx, nu, nw, [t for t = 2:T-1], :equality)

# loop constraints 
function loop!(x, u, w) 
    nq = RoboDojo.quadruped.nq
    xT = x[1:nx] 
    x1 = x[nx + nc + 1 .+ (1:nx)] 
    e = x1 - Array(cat(perm, perm, dims = (1,2))) * xT 
    nq = RoboDojo.quadruped.nq
    return [e[2:nq]; e[nq .+ (2:nq)]]
end
loop = DirectTrajectoryOptimization.StageConstraint(loop!, nx + nc + 1 + nx, nu, nw, [T], :equality)

# constraint set 
cons = DirectTrajectoryOptimization.ConstraintSet([bnd1, bndt, bndT], [contact_ineq1, contact_ineqt, contact_ineqT, contact_eq1, contact_eqt, loop, pin_con...])#, loop])

# ## problem 
trajopt = DirectTrajectoryOptimization.TrajectoryOptimizationProblem(obj, model, cons)
s = DirectTrajectoryOptimization.Solver(trajopt, options=DirectTrajectoryOptimization.Options(
    max_iter=2000,
    tol=1.0e-2,
    constr_viol_tol=1.0e-2,
))

# ## initialize
q_interp = DirectTrajectoryOptimization.linear_interpolation(q1, qT, T+1)
# q_interp = DirectTrajectoryOptimization.linear_interpolation(q1, q1, T+1)

x_interp = [[q_interp[t]; q_interp[t+1]] for t = 1:T]
u_guess = [max.(0.0, 0.001 * randn(nu)) for t = 1:T-1] # may need to run more than once to get good trajectory
z0 = zeros(s.p.num_var)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    if t == 1
        z0[idx] = x_interp[t]
    else 
        z0[idx] = [x_interp[t]; max.(0.0, 0.001 * randn(nc + 1)); x_interp[t-1]] 
    end
    # z0[idx] = x_interp[t]
end
for (t, idx) in enumerate(s.p.trajopt.model.idx.u)
    z0[idx] = u_guess[t]
end
DirectTrajectoryOptimization.initialize!(s, z0)

# ## solve
DirectTrajectoryOptimization.solve!(s)

# ## solution
@show trajopt.x[1]
@show trajopt.x[T]
sum([u[end] for u in trajopt.u[1:end-1]])
trajopt.x[T][nx + nc + 1 .+ (1:nx)] - trajopt.x[1]

# ## visualize 
vis = Visualizer() 
open(vis)
q_sol = [trajopt.x[1][1:RoboDojo.quadruped.nq], [x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)] for x in trajopt.x]...]
RoboDojo.visualize!(vis, RoboDojo.quadruped, q_sol, Δt=h)

maximum([norm(contact_constraints_equality(h, trajopt.x[t], trajopt.u[t], zeros(0)), Inf) for t = 1:T-1])
norm(max.(0.0, contact_constraints_inequality1(h, trajopt.x[1], trajopt.u[1], zeros(0))), Inf)
maximum([norm(max.(0.0, contact_constraints_inequalityt(h, trajopt.x[t], trajopt.u[t], zeros(0))), Inf) for t = 2:T-1])
norm(max.(0.0, contact_constraints_inequalityT(h, trajopt.x[T], zeros(0), zeros(0))), Inf)
maximum([norm(contact_constraints_equality(h, trajopt.x[t], trajopt.u[t], zeros(0)), Inf) for t = 1:T-1])
maximum([norm(quadruped_dyn(mass_matrix, dynamics_bias, h, trajopt.x[t+1], trajopt.x[t], trajopt.u[t], zeros(0)), Inf) for t = 1:T-1])
RoboDojo.signed_distance(RoboDojo.quadruped, trajopt.x[T][RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)])[1:4]
loop!(trajopt.x[T], zeros(0), zeros(0))
