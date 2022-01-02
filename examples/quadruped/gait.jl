using RoboDojo 
using LinearAlgebra 
using DirectTrajectoryOptimization 

function quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w) 
    model = RoboDojo.quadruped

    # dimensions
    nq = model.nq
    nu = model.nu 

    # configurations
    
    q1⁻ = x[1:nq] 
    q2⁻ = x[nq .+ (1:nq)]
    q2⁺ = y[1:nq]
    q3⁺ = y[nq .+ (1:nq)]

    # control 
    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:8)] 
    # ψ = u[nu + 4 + 8 .+ (1:4)] 
    # η = u[nu + 4 + 8 + 4 .+ (1:8)] 
    # sϕ = u[nu + 4 + 8 + 4 + 8 .+ (1:4)]
    # sψ = u[nu + 4 + 8 + 4 + 8 + 4 .+ (1:4)]
    # sα = u[nu + 4 + 8 + 4 + 8 + 4 + 4 .+ (1:1)]
    
    E = [1.0 -1.0] # friction mapping 
    J = RoboDojo.contact_jacobian(model, q2⁺)
    λ = transpose(J) * [
						[E * β[1:2]; γ[1]];
                        [E * β[3:4]; γ[2]];
						[E * β[5:6]; γ[3]];
						[E * β[7:8]; γ[4]];
					   ]

    [q2⁺ - q2⁻;
     RoboDojo.dynamics(model, mass_matrix, dynamics_bias, 
        h, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)]
end

function quadruped_dyn1(mass_matrix, dynamics_bias, h, y, x, u, w)
	model = RoboDojo.quadruped
    [
     quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
     y[model.nq .+ (1:model.nq)] - x
    ]
end

function quadruped_dynt(mass_matrix, dynamics_bias, h, y, x, u, w)
	model = RoboDojo.quadruped
    [
     quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
     y[model.nq .+ (1:model.nq)] - x[model.nq .+ (1:model.nq)]
    ]
end


function contact_constraints_inequality(h, x, u, w) 
    model = RoboDojo.quadruped

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:8)] 
    ψ = u[nu + 4 + 8 .+ (1:4)] 
    η = u[nu + 4 + 8 + 4 .+ (1:8)] 
    sϕ = u[nu + 4 + 8 + 4 + 8 .+ (1:4)]
    sψ = u[nu + 4 + 8 + 4 + 8 + 4 .+ (1:4)]
    sα = u[nu + 4 + 8 + 4 + 8 + 4 + 4 .+ (1:1)]

    ϕ = RoboDojo.signed_distance(model, q3) 
   
    v = (q3 - q2) ./ h[1]
    vT_foot = [(RoboDojo.quadruped_contact_kinematics_jacobians[i](q3) * v)[1] for i = 1:4]
    vT = vcat([[vT_foot[i]; -vT_foot[i]] for i = 1:4]...)
    
    ψ_stack = vcat([ψ[i] * ones(2) for i = 1:4]...)
    
    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)

    [
     γ .* sϕ .- sα;
     β .* η .- sα;
     ψ .* sψ  .- sα;
    ]
end

function contact_constraints_equality(h, x, u, w) 
    model = RoboDojo.quadruped

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:8)] 
    ψ = u[nu + 4 + 8 .+ (1:4)] 
    η = u[nu + 4 + 8 + 4 .+ (1:8)] 
    sϕ = u[nu + 4 + 8 + 4 + 8 .+ (1:4)]
    sψ = u[nu + 4 + 8 + 4 + 8 + 4 .+ (1:4)]
    sα = u[nu + 4 + 8 + 4 + 8 + 4 + 4 .+ (1:1)]

    ϕ = RoboDojo.signed_distance(model, q3) 
   
    v = (q3 - q2) ./ h[1]
    vT_foot = [(RoboDojo.quadruped_contact_kinematics_jacobians[i](q3) * v)[1] for i = 1:4]
    vT = vcat([[vT_foot[i]; -vT_foot[i]] for i = 1:4]...)
    
    ψ_stack = vcat([ψ[i] * ones(2) for i = 1:4]...)
    
    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)

    [
     sϕ - ϕ;
     sψ - fc;
     η - vT - ψ_stack;
    ]
end

# ## horizon 
T = 31 
T_fix = 5
tf = 0.625 
h = tf / (T - 1) 
# h = 0.01

# ## permutation matrix
perm = [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
		0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
		0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
		0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
		0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
		0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
		0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]

function ellipse_trajectory(x_start, x_goal, z, T)
	dist = x_goal - x_start
	a = 0.5 * dist
	b = z
	z̄ = 0.0
	x = range(x_start, stop = x_goal, length = T)
	z = sqrt.(max.(0.0, (b^2) * (1.0 .- ((x .- (x_start + a)).^2.0) / (a^2.0))))
	return x, z
end

function initial_configuration(model::Quadruped, θ1, θ2, θ3)
    q1 = zeros(model.nq)
    q1[3] = pi / 2.0
    q1[4] = -θ1
    q1[5] = θ2

    q1[8] = -θ1
    q1[9] = θ2

    q1[2] = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])

    q1[10] = -θ3
    q1[11] = acos((q1[2] - model.l_thigh2 * cos(q1[10])) / model.l_calf2)

    q1[6] = -θ3
    q1[7] = acos((q1[2] - model.l_thigh2 * cos(q1[6])) / model.l_calf2)

    return q1
end

θ1 = pi / 4.0
θ2 = pi / 4.0
θ3 = pi / 3.0

q1 = initial_configuration(model, θ1, θ2, θ3)
RoboDojo.visualize!(vis, model, [q1])

# feet positions
pr1 = kinematics_calf(model, q1, leg=:leg1, mode=:ee)
pr2 = kinematics_calf(model, q1, leg=:leg2, mode=:ee)
pf1 = kinematics_calf(model, q1, leg=:leg3, mode=:ee)
pf2 = kinematics_calf(model, q1, leg=:leg4, mode=:ee)

strd = 2 * (pr1 - pr2)[1]
qT = Array(perm) * copy(q1)
qT[1] += 0.5 * strd

# torso height
pt = q1[3]

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

tr = range(0, stop = tf, length = T)
plot(tr, hcat(pr1_ref...)')
plot!(tr, hcat(pf1_ref...)')

plot(tr, hcat(pr2_ref...)')
plot!(tr, hcat(pf2_ref...)')

# Bounds

# control
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 33.5 * h
_uu[end] = 2.0 * h
_ul = zeros(model.m)
_ul[model.idx_u] .= -33.5 * h
_ul[end] = 0.75 * h
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T,
    x1 = [Inf * ones(model.nq); q1],
    xT = [Inf * ones(model.nq); qT[1]; Inf * ones(model.nq - 1)])

# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])
q_ref = linear_interpolation(q1, qT, T+1)
render(vis)
visualize!(vis, model, q_ref, Δt = h)
x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e4, model.m - 1)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_time_tracking_objective(
    [1.0 * Diagonal(1.0e-5 * ones(model.n)) for t = 1:T],
    [1.0 * Diagonal([1.0e-3 * ones(model.nu)..., 1.0e-3 * ones(model.nc + model.nb)..., zeros(model.m - model.nu - model.nc - model.nb)...]) for t = 1:T-1],
    [[qT; qT] for t = 1:T],
    [zeros(model.m) for t = 1:T],
    1.0)

# quadratic velocity penalty
#Σ v' Q v
v_penalty = 0.0 * ones(model.nq)
obj_velocity = velocity_objective(
    [h * Diagonal(v_penalty) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3, 4, 5, 6, 7, 8, 9, 10, 11]))

obj_ctrl_velocity = control_velocity_objective(Diagonal([1.0e-3 * ones(model.nu)..., 1.0e-3 * ones(model.nc + model.nb)..., zeros(model.m - model.nu - model.nc - model.nb)...]))

function l_stage(x, u, t)
	q1 = view(x, 1:11)
	q2 = view(x, 11 .+ (1:11))
    J = 0.0

	# torso height
    J += 100.0 * (kinematics_1(model, q1, body = :torso, mode = :ee)[2] - kinematics_1(model, view(x0[t], 1:11), body = :torso, mode = :com)[2])^2.0
	J += 100.0 * (kinematics_1(model, q2, body = :torso, mode = :ee)[2] - kinematics_1(model, view(x0[t], 1:11), body = :torso, mode = :com)[2])^2.0

	J += 100.0 * (q1[2] - x0[1][2])^2.0
	J += 100.0 * (q2[2] - x0[1][2])^2.0

    if true
		J += 10000.0 * sum((pr2_ref[t] - kinematics_2(model, q1, body = :calf_2, mode = :ee)).^2.0)
	    J += 10000.0 * sum((pf2_ref[t] - kinematics_3(model, q1, body = :calf_4, mode = :ee)).^2.0)
	end

    return J
end

l_terminal(x) = 0.0
obj_shaping = nonlinear_stage_objective(l_stage, l_terminal)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
					  obj_shaping,
					  obj_ctrl_velocity])
# Constraints
include_constraints(["stage", "contact", "free_time", "loop"])
function pinned1!(c, x, u, t)
    q = view(x, 1:11)
    c[1:2] = pr1_ref[t] - kinematics_2(model, q, body = :calf_1, mode = :ee)
    c[3:4] = pf1_ref[t] - kinematics_3(model, q, body = :calf_3, mode = :ee)
    nothing
end

function pinned2!(c, x, u, t)
    q = view(x, 1:11)
	c[1:2] = pr2_ref[t] - kinematics_2(model, q, body = :calf_2, mode = :ee)
    c[3:4] = pf2_ref[t] - kinematics_3(model, q, body = :calf_4, mode = :ee)
    nothing
end

n_stage = 4
t_idx1 = vcat([t for t = 1:T])
t_idx2 = vcat([1:T_fix]...)
con_pinned1 = stage_constraints(pinned1!, n_stage, (1:0), t_idx1)
con_pinned2 = stage_constraints(pinned2!, n_stage, (1:0), t_idx2)

con_contact = contact_constraints(model, T)
con_free_time = free_time_constraints(T)
con_loop = loop_constraints(model, collect([(2:model.nq)...,
	(nq .+ (2:model.nq))...]), 1, T, perm = Array(cat(perm, perm, dims = (1,2))))
con = multiple_constraints([con_contact, con_loop,
    con_free_time, con_pinned1, con_pinned2])


######

# ## quadruped 
nx = 2 * RoboDojo.quadruped.nq
nu = RoboDojo.quadruped.nu + 4 + 4 + 2 + 4 + 4 + 2 + 1
nw = RoboDojo.quadruped.nw

# ## model
mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.quadruped)
d1 = DirectTrajectoryOptimization.Dynamics((y, x, u, w) -> quadruped_dyn1(mass_matrix, dynamics_bias, [h], y, x, u, w), 2 * nx, nx, nu)
dt = DirectTrajectoryOptimization.Dynamics((y, x, u, w) -> quadruped_dynt(mass_matrix, dynamics_bias, [h], y, x, u, w), 2 * nx, 2 * nx, nu)

dyn = [d1, [dt for t = 2:T-1]...]
model = DirectTrajectoryOptimization.DynamicsModel(dyn)

# ## initial conditions
q1 = [0.0; 0.5 + RoboDojo.quadruped.foot_radius; 0.0; 0.5]
qM = [0.5; 0.5 + RoboDojo.quadruped.foot_radius; 0.0; 0.5]
qT = [1.0; 0.5 + RoboDojo.quadruped.foot_radius; 0.0; 0.5]
q_ref = [0.5; 0.75 + RoboDojo.quadruped.foot_radius; 0.0; 0.25]

x1 = [q1; q1]
xM = [qM; qM]
xT = [qT; qT]
x_ref = [q_ref; q_ref]

# ## gate 
GATE = 1 
# GATE = 2 
# GATE = 3

if GATE == 1 
	r_cost = 1.0e-1 
	q_cost = 1.0e-1
elseif GATE == 2 
	r_cost = 1.0
	q_cost = 1.0
elseif GATE == 3 
	r_cost = 1.0e-3
	q_cost = 1.0e-1
end

# ## objective
function obj1(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x - x_ref) * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0]) * (x - x_ref) 
	J += 0.5 * transpose(u) * Diagonal(r_cost * ones(nu)) * u
    J += 1000.0 * u[nu]
	return J
end

function objt(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal(q_cost * [1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0]) * (x[1:nx] - x_ref)
	J += 0.5 * transpose(u) * Diagonal(r_cost * ones(nu)) * u
    J += 1000.0 * u[nu]
	return J
end

function objT(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - x_ref)
    return J
end

c1 = DirectTrajectoryOptimization.Cost(obj1, nx, nu, nw, [1])
ct = DirectTrajectoryOptimization.Cost(objt, 2 * nx, nu, nw, [t for t = 2:T-1])
cT = DirectTrajectoryOptimization.Cost(objT, 2 * nx, 0, 0, [T])
obj = [c1, ct, cT]

# ## constraints
function stage1_eq(x, u, w) 
    [
   	RoboDojo.kinematics_foot(RoboDojo.quadruped, x[1:RoboDojo.quadruped.nq]) - RoboDojo.kinematics_foot(RoboDojo.quadruped, x1[1:RoboDojo.quadruped.nq]);
	RoboDojo.kinematics_foot(RoboDojo.quadruped, x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)]) - RoboDojo.kinematics_foot(RoboDojo.quadruped, x1[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)])
    ]
end

function terminal_con_eq(x, u, w) 
	θ = x[nx .+ (1:nx)]
    [
	x[1:RoboDojo.quadruped.nq][collect([2, 3, 4])] - θ[1:RoboDojo.quadruped.nq][collect([2, 3, 4])]
	x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)][collect([2, 3, 4])] - θ[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)][collect([2, 3, 4])]
    ]
end

function terminal_con_ineq(x, u, w) 
	x_travel = 0.5
	θ = x[nx .+ (1:nx)]
    [
	x_travel - (x[1] - θ[1])
	x_travel - (x[RoboDojo.quadruped.nq + 1] - θ[RoboDojo.quadruped.nq + 1])
    ]
end

contact_ineq1 = StageConstraint((x, u, w) -> contact_constraints_inequality(h, x, u, w), nx, nu, nw, [1], :inequality)
contact_ineqt = StageConstraint((x, u, w) -> contact_constraints_inequality(h, x, u, w), 2 * nx, nu, nw, [t for t = 2:T-1], :inequality)
contact_eq1 = StageConstraint((x, u, w) -> contact_constraints_equality(h, x, u, w), nx, nu, nw, [1], :equality)
contact_eqt = StageConstraint((x, u, w) -> contact_constraints_equality(h, x, u, w), 2 * nx, nu, nw, [t for t = 2:T-1], :equality)

ql = [-Inf; 0; -Inf; 0.0]
qu = [Inf; Inf; Inf; 1.0]
xl1 = [q1; ql] 
xu1 = [q1; qu]
xlt = [ql; ql; -Inf * ones(nx)] 
xut = [qu; qu; Inf * ones(nx)]
ul = [-10.0; -10.0; zeros(nu - 2)]
uu = [10.0; 10.0; Inf * ones(nu - 2)]

bnd1 = DirectTrajectoryOptimization.Bound(nx, nu, [1], xl=xl1, xu=xu1, ul=ul, uu=uu)
bndt = DirectTrajectoryOptimization.Bound(2 * nx, nu, [t for t = 2:T-1], xl=xlt, xu=xut, ul=ul, uu=uu)
bndT = DirectTrajectoryOptimization.Bound(2 * nx, 0, [T], xl=xlt, xu=xut)

con_eq1 = DirectTrajectoryOptimization.StageConstraint(stage1_eq, nx, nu, nw, [1], :equality)
conT_eq = DirectTrajectoryOptimization.StageConstraint(terminal_con_eq, 2 * nx, nu, nw, [T], :equality)
conT_ineq = DirectTrajectoryOptimization.StageConstraint(terminal_con_ineq, 2 * nx, nu, nw, [T], :inequality)

cons = DirectTrajectoryOptimization.ConstraintSet([bnd1, bndt, bndT], [contact_ineq1, contact_ineqt, contact_eq1, contact_eqt, con_eq1, conT_eq, conT_ineq])

# ## problem 
trajopt = DirectTrajectoryOptimization.TrajectoryOptimizationProblem(obj, model, cons)
s = DirectTrajectoryOptimization.Solver(trajopt, options=DirectTrajectoryOptimization.Options(
    tol=1.0e-3,
    constr_viol_tol=1.0e-3,
))

# ## initialize
x_interpolation = [x1, [[x1; x1] for t = 2:T]...]
u_guess = [[0.0; RoboDojo.quadruped.gravity * RoboDojo.quadruped.mass_body * 0.5 * h[1]; 1.0e-1 * ones(nu - 2)] for t = 1:T-1] # may need to run more than once to get good trajectory
z0 = zeros(s.p.num_var)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    z0[idx] = x_interpolation[t]
end
for (t, idx) in enumerate(s.p.trajopt.model.idx.u)
    z0[idx] = u_guess[t]
end
DirectTrajectoryOptimization.initialize!(s, z0)

# ## solve
DirectTrajectoryOptimization.solve!(s)

# ## benchmark 
s.solver.options["print_level"] = 0
function solve(s, z) 
    DirectTrajectoryOptimization.initialize!(s, z) 
    DirectTrajectoryOptimization.solve!(s)
end

@benchmark DirectTrajectoryOptimization.solve($s, $z0)

# ## solution
@show trajopt.x[1]
@show trajopt.x[T]
sum([u[nu] for u in trajopt.u[1:end-1]])
trajopt.x[1] - trajopt.x[T][1:nx]

# ## visualize 
vis = Visualizer() 
render(vis)
q_sol = state_to_configuration([x[1:nx] for x in trajopt.x])
RoboDojo.visualize!(vis, RoboDojo.quadruped, q_sol, Δt=h)

maximum([norm(contact_constraints_equality(h, trajopt.x[t], trajopt.u[t], zeros(0)), Inf) for t = 1:T-1])
maximum([norm(max.(0.0, contact_constraints_inequality(h, trajopt.x[t], trajopt.u[t], zeros(0))), Inf) for t = 1:T-1])
maximum([norm(contact_constraints_equality(h, trajopt.x[t], trajopt.u[t], zeros(0)), Inf) for t = 1:T-1])
maximum([norm(quadruped_dyn(mass_matrix, dynamics_bias, h, trajopt.x[t+1], trajopt.x[t], trajopt.u[t], zeros(0)), Inf) for t = 1:T-1])
minimum([min.(0.0, u[2 .+ (1:nu-2)]) for u in trajopt.u[1:end-1]])

# ## comparison 
function obj1_compare(x, u, w)
	x = x[1:8] 
	u = u[1:2] 

	J = 0.0 
	J += 0.5 * transpose(x - x_ref) * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0]) * (x - x_ref) 
	J += 0.5 * transpose(u) * Diagonal(r_cost * ones(quadruped.nu)) * u
	return J
end

function objt_compare(x, u, w)
	x = x[1:8] 
	u = u[1:2] 
	J = 0.0 
	J += 0.5 * transpose(x - x_ref) * q_cost * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0]) * (x - x_ref) 
	J += 0.5 * transpose(u) * Diagonal(r_cost * ones(quadruped.nu)) * u
	return J
end

function objT_compare(x, u, w)
	x = x[1:8] 
	J = 0.0 
	J += 0.5 * transpose(x - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - x_ref) 
	return J
end

function obj_compare(x, u, w) 
	J = 0.0 
	J += obj1_compare(x[1], u[1], w[1]) 
	for t = 2:T-1 
		J += objt_compare(x[t], u[t], w[t])
	end 	
	J += objT_compare(x[T], nothing, nothing) 

	return J 
end

obj_compare(x_sol, u_sol, w)
obj_compare(trajopt.x, trajopt.u, w)
@show norm(terminal_con(trajopt.x[T], zeros(0), zeros(0))[3:4], Inf)