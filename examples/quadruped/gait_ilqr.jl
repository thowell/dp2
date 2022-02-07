using ParameterOptimization
using IterativeLQR
using RoboDojo
using LinearAlgebra
using Random

# ## visualize 
vis = Visualizer() 
render(vis)

# ## state-space model
T = 21
h = 0.05
hopper = RoboDojo.hopper

## Implicit Dynamics 
struct ImplicitDynamics{T,R,RZ,Rθ,M<:RoboDojo.Model{T},P<:RoboDojo.Policy{T},D<:RoboDojo.Disturbances{T},I} <: Model{T}
    n::Int
    m::Int
    d::Int
	eval_sim::Simulator{T,R,RZ,Rθ,M,P,D}
	grad_sim::Simulator{T,R,RZ,Rθ,M,P,D}
	q1::Vector{T} 
	q2::Vector{T} 
	v1::Vector{T}
	idx_q1::Vector{Int} 
	idx_q2::Vector{Int}
	idx_u1::Vector{Int}
	info::I
end

function get_simulator(model, h, r_func, rz_func, rθ_func; 
	T=1, r_tol=1.0e-8, κ_eval_tol=1.0e-4, nc=model.nc, nb=model.nc, diff_sol=true)

	sim = Simulator(model, T; 
        h=h, 
        residual=r_func, 
        jacobian_z=rz_func, 
        jacobian_θ=rθ_func,
        diff_sol=diff_sol,
        solver_opts=InteriorPointOptions(
            undercut=Inf,
            γ_reg=0.1,
            r_tol=r_tol,
            κ_tol=κ_eval_tol,  
            max_ls=25,
            ϵ_min=0.25,
            diff_sol=diff_sol,
            verbose=false))  

    # set trajectory sizes
	sim.traj.γ .= [zeros(nc) for t = 1:T] 
	sim.traj.b .= [zeros(nb) for t = 1:T] 

    sim.grad.∂γ1∂q1 .= [zeros(nc, model.nq) for t = 1:T] 
	sim.grad.∂γ1∂q2 .= [zeros(nc, model.nq) for t = 1:T]
	sim.grad.∂γ1∂v1 .= [zeros(nc, model.nq) for t = 1:T]
	sim.grad.∂γ1∂u1 .= [zeros(nc, model.nu) for t = 1:T]
	sim.grad.∂b1∂q1 .= [zeros(nb, model.nq) for t = 1:T] 
	sim.grad.∂b1∂q2 .= [zeros(nb, model.nq) for t = 1:T]
	sim.grad.∂b1∂v1 .= [zeros(nb, model.nq) for t = 1:T]
	sim.grad.∂b1∂u1 .= [zeros(nb, model.nu) for t = 1:T]
	
    return sim
end

function ImplicitDynamics(model, h, r_func, rz_func, rθ_func; 
	T=1, r_tol=1.0e-8, κ_eval_tol=1.0e-6, κ_grad_tol=1.0e-6, 
	no_impact=false, no_friction=false, 
	n=(2 * model.nq), m=model.nu, d=model.nw, nc=model.nc, nb=model.nc,
	info=nothing) 

	# set trajectory sizes
	no_impact && (nc = 0) 
	no_friction && (nb = 0) 

	eval_sim = get_simulator(model, h, r_func, rz_func, rθ_func; 
			T=T, r_tol=r_tol, κ_eval_tol=κ_eval_tol, nc=nc, nb=nb, diff_sol=false)

	grad_sim = get_simulator(model, h, r_func, rz_func, rθ_func; 
			T=T, r_tol=r_tol, κ_eval_tol=κ_grad_tol, nc=nc, nb=nb, diff_sol=true)

	q1 = zeros(model.nq) 
	q2 = zeros(model.nq) 
	v1 = zeros(model.nq) 

	idx_q1 = collect(1:model.nq) 
	idx_q2 = collect(model.nq .+ (1:model.nq)) 
	idx_u1 = collect(1:model.nu)
	
	ImplicitDynamics(n, m, d, 
		eval_sim, grad_sim, 
		q1, q2, v1,
		idx_q1, idx_q2, idx_u1, info)
end


struct ParameterOptInfo{T}
	idx_q1::Vector{Int} 
	idx_q2::Vector{Int} 
	idx_u1::Vector{Int}
	idx_uθ::Vector{Int}
	idx_uθ1::Vector{Int} 
	idx_uθ2::Vector{Int}
	idx_xθ::Vector{Int}
	v1::Vector{T}
end

info = ParameterOptInfo(
	collect(1:hopper.nq), 
	collect(hopper.nq .+ (1:hopper.nq)), 
	collect(1:hopper.nu), 
	collect(hopper.nu .+ (1:2 * hopper.nq)),
	collect(hopper.nu .+ (1:hopper.nq)), 
	collect(hopper.nu + hopper.nq .+ (1:hopper.nq)), 
	collect(2 * hopper.nq .+ (1:2 * hopper.nq)),
	zeros(hopper.nq)
)

im_dyn1 = ImplicitDynamics(hopper, h, 
	eval(RoboDojo.residual_expr(hopper)), 
	eval(RoboDojo.jacobian_var_expr(hopper)), 
	eval(RoboDojo.jacobian_data_expr(hopper)); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-3,
	n=(2 * hopper.nq), m=(hopper.nu + 2 * hopper.nq), nc=4, nb=2, info=info)

im_dynt = ImplicitDynamics(hopper, h, 
	eval(RoboDojo.residual_expr(hopper)), 
	eval(RoboDojo.jacobian_var_expr(hopper)), 
	eval(RoboDojo.jacobian_data_expr(hopper)); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-3,
	n=4 * hopper.nq, m=hopper.nu, nc=4, nb=2, info=info) 

function f1(d, model::ImplicitDynamics, x, u, w)

	θ = @views u[model.info.idx_uθ]
	q1 = @views u[model.info.idx_uθ1]
	q2 = @views u[model.info.idx_uθ2]
	u1 = @views u[model.info.idx_u1] 

	model.info.v1 .= q2 
	model.info.v1 .-= q1 
	model.info.v1 ./= model.eval_sim.h

	q3 = RoboDojo.step!(model.eval_sim, q2, model.info.v1, u1, 1)

	d[model.info.idx_q1] = q2 
	d[model.info.idx_q2] = q3
	d[model.info.idx_xθ] = θ

	return d
end

function f1x(dx, model::ImplicitDynamics, x, u, w)
	dx .= 0.0
	return dx
end

function f1u(du, model::ImplicitDynamics, x, u, w)
	nq = model.grad_sim.model.nq

	θ = @views u[model.info.idx_uθ]
	q1 = @views u[model.info.idx_uθ1]
	q2 = @views u[model.info.idx_uθ2]
	u1 = @views u[model.info.idx_u1] 

	model.info.v1 .= q2 
	model.info.v1 .-= q1 
	model.info.v1 ./= model.grad_sim.h

	RoboDojo.step!(model.grad_sim, q2, model.info.v1, u1, 1)

	for i = 1:nq
		du[model.info.idx_q1[i], model.info.idx_uθ[i]] = 1.0 
	end
	du[model.info.idx_q2, model.info.idx_u1] = model.grad_sim.grad.∂q3∂u1[1] 
	du[model.info.idx_q2, model.info.idx_uθ1] = model.grad_sim.grad.∂q3∂q1[1] 
	du[model.info.idx_q2, model.info.idx_uθ2] = model.grad_sim.grad.∂q3∂q2[1] 

	return du
end

function ft(d, model::ImplicitDynamics, x, u, w)

	θ = @views x[model.info.idx_xθ] 
	q1 = @views x[model.info.idx_q1]
	q2 = @views x[model.info.idx_q2] 
	u1 = u 

	model.info.v1 .= q2 
	model.info.v1 .-= q1 
	model.info.v1 ./= model.eval_sim.h 

	q3 = RoboDojo.step!(model.eval_sim, q2, model.info.v1, u1, 1)

	d[model.info.idx_q1] = q2 
	d[model.info.idx_q2] = q3
	d[model.info.idx_xθ] = θ

	return d
end

function ftx(dx, model::ImplicitDynamics, x, u, w)
	nq = model.grad_sim.model.nq

	θ = @views x[model.info.idx_xθ] 
	q1 = @views x[model.info.idx_q1]
	q2 = @views x[model.info.idx_q2] 
	u1 = u 

	model.info.v1 .= q2 
	model.info.v1 .-= q1 
	model.info.v1 ./= model.grad_sim.h 

	q3 = RoboDojo.step!(model.grad_sim, q2, model.info.v1, u1, 1)

	for i = 1:nq
		dx[model.info.idx_q1[i], model.info.idx_q2[i]] = 1.0 
	end
	dx[model.info.idx_q2, model.info.idx_q1] = model.grad_sim.grad.∂q3∂q1[1] 
	dx[model.info.idx_q2, model.info.idx_q2] = model.grad_sim.grad.∂q3∂q2[1] 
	for i in model.info.idx_xθ 
		dx[i, i] = 1.0 
	end

	return dx
end
	
function ftu(du, model::ImplicitDynamics, x, u, w)
	θ = @views x[model.info.idx_xθ] 
	q1 = @views x[model.info.idx_q1]
	q2 = @views x[model.info.idx_q2] 
	u1 = u 

	model.info.v1 .= q2 
	model.info.v1 .-= q1 
	model.info.v1 ./= model.grad_sim.h 

	q3 = RoboDojo.step!(model.grad_sim, q2, model.info.v1, u1, 1)

	du[model.info.idx_q2, model.info.idx_u1] = model.grad_sim.grad.∂q3∂u1[1]

	return du
end

# ## iLQR model
ilqr_dyn1 = IterativeLQR.Dynamics((d, x, u, w) -> f1(d, im_dyn1, x, u, w), 
					(dx, x, u, w) -> f1x(dx, im_dyn1, x, u, w), 
					(du, x, u, w) -> f1u(du, im_dyn1, x, u, w), 
					4 * hopper.nq, 2 * hopper.nq, hopper.nu + 2 * hopper.nq)  

ilqr_dynt = IterativeLQR.Dynamics((d, x, u, w) -> ft(d, im_dynt, x, u, w), 
	(dx, x, u, w) -> ftx(dx, im_dynt, x, u, w), 
	(du, x, u, w) -> ftu(du, im_dynt, x, u, w), 
	4 * hopper.nq, 4 * hopper.nq, hopper.nu)  

model = [ilqr_dyn1, [ilqr_dynt for t = 2:T-1]...]

# ## initial conditions
q1 = [0.0; 0.5 + hopper.foot_radius; 0.0; 0.5]
qM = [0.5; 0.5 + hopper.foot_radius; 0.0; 0.5]
qT = [1.0; 0.5 + hopper.foot_radius; 0.0; 0.5]
q_ref = [0.5; 0.75 + hopper.foot_radius; 0.0; 0.25]

x1 = [q1; q1]
xM = [qM; qM]
xT = [qT; qT]
x_ref = [q_ref; q_ref]

# ## objective

GATE = 1 
## GATE = 2 
## GATE = 3

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

function obj1(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x - x_ref) * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0]) * (x - x_ref) 
	J += 0.5 * transpose(u) * Diagonal([r_cost * ones(hopper.nu); 1.0e-1 * ones(hopper.nq); 1.0e-5 * ones(hopper.nq)]) * u
	return J
end

function objt(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x - [x_ref; zeros(2 * hopper.nq)]) * q_cost * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0; zeros(2 * hopper.nq)]) * (x - [x_ref; zeros(2 * hopper.nq)]) 
	J += 0.5 * transpose(u) * Diagonal(r_cost * ones(hopper.nu)) * u
	return J
end

function objT(x, u, w)
	J = 0.0 
	J += 0.5 * transpose(x - [x_ref; zeros(2 * hopper.nq)]) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; zeros(2 * hopper.nq)]) * (x - [x_ref; zeros(2 * hopper.nq)]) 
	return J
end

c1 = IterativeLQR.Cost(obj1, 2 * hopper.nq, hopper.nu + 2 * hopper.nq, 0)
ct = IterativeLQR.Cost(objt, 4 * hopper.nq, hopper.nu, 0)
cT = IterativeLQR.Cost(objT, 4 * hopper.nq, 0, 0)
obj = [c1, [ct for t = 2:T-1]..., cT]

# ## constraints
ul = [-10.0; -10.0]
uu = [10.0; 10.0]
 
function stage1_con(x, u, w) 
    [
    ul - u[1:hopper.nu]; # control limit (lower)
    u[1:hopper.nu] - uu; # control limit (upper)

	u[hopper.nu .+ (1:hopper.nq)] - x1[1:hopper.nq];

	RoboDojo.kinematics_foot(hopper, u[hopper.nu .+ (1:hopper.nq)]) - RoboDojo.kinematics_foot(hopper, x1[1:hopper.nq]);
	RoboDojo.kinematics_foot(hopper, u[hopper.nu + hopper.nq .+ (1:hopper.nq)]) - RoboDojo.kinematics_foot(hopper, x1[hopper.nq .+ (1:hopper.nq)])
    ]
end 

function staget_con(x, u, w) 
    [
    ul - u[collect(1:hopper.nu)]; # control limit (lower)
    u[collect(1:hopper.nu)] - uu; # control limit (upper)
    ]
end 

function terminal_con(x, u, w) 
	x_travel = 0.5
	θ = x[2 * hopper.nq .+ (1:(2 * hopper.nq))]
    [
	x_travel - (x[1] - θ[1])
	x_travel - (x[hopper.nq + 1] - θ[hopper.nq + 1])
	x[1:hopper.nq][collect([2, 3, 4])] - θ[1:hopper.nq][collect([2, 3, 4])]
	x[hopper.nq .+ (1:hopper.nq)][collect([2, 3, 4])] - θ[hopper.nq .+ (1:hopper.nq)][collect([2, 3, 4])]
    ]
end

con1 = IterativeLQR.Constraint(stage1_con, 2 * hopper.nq, hopper.nu + 2 * hopper.nq, idx_ineq=collect(1:4))
cont = IterativeLQR.Constraint(staget_con, 4 * hopper.nq, hopper.nu, idx_ineq=collect(1:4))
conT = IterativeLQR.Constraint(terminal_con, 4 * hopper.nq, 0, idx_ineq=collect(1:2))
cons = [con1, [cont for t = 2:T-1]..., conT]

# ## rollout
ū_stand = [t == 1 ? [0.0; hopper.gravity * hopper.mass_body * 0.5 * h; x1] : [0.0; hopper.gravity * hopper.mass_body * 0.5 * h] for t = 1:T-1]
w = [zeros(hopper.nw) for t = 1:T-1]
x̄ = IterativeLQR.rollout(model, x1, ū_stand)
q̄ = state_to_configuration(x̄)
RoboDojo.visualize!(vis, hopper, x̄, Δt=h)

# ## problem
prob = IterativeLQR.problem_data(model, obj, cons)
IterativeLQR.initialize_controls!(prob, ū_stand)
IterativeLQR.initialize_states!(prob, x̄)

# ## solve
IterativeLQR.reset!(prob.s_data)
@time IterativeLQR.solve!(prob, 
	linesearch = :armijo,
	α_min=1.0e-5,
	obj_tol=1.0e-3,
	grad_tol=1.0e-3,
	max_iter=10,
	max_al_iter=15,
	con_tol=0.001,
	ρ_init=1.0, 
	ρ_scale=10.0, 
	verbose=false)

@show IterativeLQR.eval_obj(prob.m_data.obj.costs, prob.m_data.x, prob.m_data.u, prob.m_data.w)
@show prob.s_data.iter[1]
@show norm(terminal_con(prob.m_data.x[T], zeros(0), zeros(0))[3:4], Inf)
@show prob.s_data.obj[1] # augmented Lagrangian cost
    
# ## solution
x_sol, u_sol = IterativeLQR.get_trajectory(prob)
q_sol = state_to_configuration(x_sol)
RoboDojo.visualize!(vis, hopper, q_sol, Δt=h)

## benchmark (NOTE: gate 3 seems to break @benchmark, just run @time instead...)
@benchmark IterativeLQR.solve!($prob, $x̄, $ū_stand, 
	linesearch = :armijo,
	α_min=1.0e-5,
	obj_tol=1.0e-3,
	grad_tol=1.0e-3,
	max_iter=10,
	max_al_iter=15,
	con_tol=0.001,
	ρ_init=1.0, 
	ρ_scale=10.0, 
	verbose=false)


