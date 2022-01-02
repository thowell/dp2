# PREAMBLE

# PKG_SETUP

# ## Setup

using DirectTrajectoryOptimization
using LinearAlgebra
using Plots

# ## drone
include("model_drone.jl") 

nx = drone.nx
nu = drone.nu
nw = drone.nw

# ## initialization
x1 = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0] 
xT = [1.0; 1.0; 0.0; 0.0; 0.0; 0.0]
θ_angle = 0.25 
u_hover = hover_controls(drone, θ_angle)

# ## linear policy 
# function policy(θ, x) 
#     M = reshape(θ, nu, nx) 
#     M * (x - xT) 
# end

# nθ = nu * nx

# ## (1-layer) multi-layer perceptron policy
l_input = nx
l1 = 6
l2 = nu
nθ = l1 * l_input + l2 * l1 

function policy(θ, x, goal) 
    shift = 0
    # input 
    input = x - goal

    # layer 1
    W1 = reshape(θ[shift .+ (1:(l1 * l_input))], l1, l_input) 
    z1 = W1 * input
    o1 = tanh.(z1)
    shift += l1 * l_input

    # layer 2 
    W2 = reshape(θ[shift .+ (1:(l2 * l1))], l2, l1) 
    z2 = W2 * o1 

    o2 = z2
    return o2
end

# ## horizon 
T = 21

# ## model
h = 0.1

function f1(y, x, u, w)
    u_ctrl = u[1:nu] 
    x_di = x[1:nx] 
    y_di = y[1:nx]
    θ = u[nu .+ (1:nθ)] 
    yθ = y[nx .+ (1:nθ)]
    [
        dynamics(drone, h, y_di, x_di, u_ctrl, w);
        yθ - θ;
    ]
    # dynamics(drone, h, y_di, x_di, u_ctrl, w)
end

function ft(y, x, u, w)
    u_ctrl = u[1:nu] 
    x_di = x[1:nx] 
    y_di = y[1:nx]
    θ = x[nx .+ (1:nθ)] 
    yθ = y[nx .+ (1:nθ)]
    [
        dynamics(drone, h, y_di, x_di, u_ctrl, w);
        yθ - θ;
    ]
    # dynamics(drone, h, y_di, x_di, u_ctrl, w)
end

dyn1 = DirectTrajectoryOptimization.Dynamics(f1, nx + nθ, nx, nu + nθ)
dynt = DirectTrajectoryOptimization.Dynamics(ft, nx + nθ, nx + nθ, nu)

dyn = [dyn1, [dynt for t = 2:T-1]...]
model = DirectTrajectoryOptimization.DynamicsModel(dyn)

# ## objective 
function o1(x, u, w)
    J = 0.0
    q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ex = x - xT
    J += 0.5 * transpose(ex) * Diagonal(q) * ex
    J += 1.0e-2 * dot(u[1:nu] - u_hover, u[1:nu] - u_hover) 
    J += 1.0e-1 * dot(u[nu .+ (1:nθ)], u[nu .+ (1:nθ)])
    return J
end

function ot(x, u, w) 
    J = 0.0
    q = [10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
    ex = x[1:nx] - xT
    J += 0.5 * transpose(ex) * Diagonal(q) * ex 
    J += 1.0e-2 * dot(u - u_hover, u_hover) 
    J += 1.0e-1 * dot(x[nx .+ (1:nθ)], x[nx .+ (1:nθ)])
    return J 
end

function ott(x, u, w) 
    J = 0.0
    q = [100.0, 100.0, 100.0, 1.0, 1.0, 1.0]
    ex = x[1:nx] - xT
    J += 0.5 * transpose(ex) * Diagonal(q) * ex 
    J += 1.0e-2 * dot(u - u_hover, u_hover) 
    J += 1.0e-1 * dot(x[nx .+ (1:nθ)], x[nx .+ (1:nθ)])
    return J 
end

function oT(x, u, w) 
    J = 0.0
    q = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    ex = x[1:nx] - xT
    J += 0.5 * transpose(ex) * Diagonal(q) * ex 
    J += 1.0e-1 * dot(x[nx .+ (1:nθ)], x[nx .+ (1:nθ)])
    return J 
end

c1 = DirectTrajectoryOptimization.Cost(o1, nx, nu + nθ, nw, [1])
ct = DirectTrajectoryOptimization.Cost(ot, nx + nθ, nu, nw, [t for t = 2:16])
ctt = DirectTrajectoryOptimization.Cost(ott, nx + nθ, nu, nw, [t for t = 17:(T-1)])
cT = DirectTrajectoryOptimization.Cost(oT, nx + nθ, 0, nw, [T])
obj = [c1, ct, ctt, cT]

# ## constraints
ul = [0.0; -0.5 * π; 0.0; -0.5 * π] 
uu = [10.0; 0.5 * π; 10.0; 0.5 * π]
bnd1 = Bound(nx, nu + nθ, [1], xl=x1, xu=x1, ul=[ul; -Inf * ones(nθ)], uu=[uu; Inf * ones(nθ)])
bndt = Bound(nx + nθ, nu, [t for t = 2:T-1], ul=ul, uu=uu)
# bndT = Bound(nx + nθ, 0, [T], xl=[xT; -Inf * ones(nθ)], xu=[xT; Inf * ones(nθ)])

function policy1(x, u, w) 
    θ = u[nu .+ (1:nθ)]
    u[1:nu] - policy(θ, x[1:nx], xT)
end

function policyt(x, u, w) 
    θ = x[nx .+ (1:nθ)]
    u[1:nu] - policy(θ, x[1:nx], xT)
end

con_policy1 = StageConstraint(policy1, nx, nu + nθ, nw, [1], :equality)
con_policyt = StageConstraint(policyt, nx + nθ, nu, nw, [t for t = 2:T-1], :equality)

cons = ConstraintSet([bnd1, bndt], 
    [con_policy1, con_policyt])

# ## problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt, 
    options=Options(
        tol=1.0e-3,
        constr_viol_tol=1.0e-3,
    ))

# ## initialize
θ0 = 0.001 * randn(nθ)
u_guess = [[u_hover; θ0], [u_hover for t = 2:T-1]...]
z0 = zeros(s.p.num_var)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    z0[idx] = t == 1 ? x1 : [x1; randn(nθ)]
end
for (t, idx) in enumerate(s.p.trajopt.model.idx.u)
    z0[idx] = u_guess[t]
end
initialize!(s, z0)

# ## solve
@time DirectTrajectoryOptimization.solve!(s)

# ## solution
@show trajopt.u[1]
@show trajopt.x[1]
@show trajopt.x[T]
θ_sol = trajopt.u[1][nu .+ (1:nθ)]

# policy1(trajopt.x[1], trajopt.u[1], nothing)
# [policyt(trajopt.x[t], trajopt.u[t], nothing) for t = 2:T-1]

# ## state
plot(hcat([x[1:nx] for x in trajopt.x]...)', label="", color=:orange, width=2.0)

# ## control
plot(hcat([u[1:nu] for u in trajopt.u[1:end-1]]..., trajopt.u[end-1])', linetype = :steppost)

# ## plot xy 
plot([x[1] for x in trajopt.x], [x[2] for x in trajopt.x], label="", color=:black, width=2.0)

# ## visualization 
include("visuals.jl")
vis = Visualizer()
open(vis) 
visualize_drone!(vis, drone, [trajopt.x], [[trajopt.u[1:end-1]..., trajopt.u[end-1]]]; Δt=h, xT=[xT])

# ## simulate policy 
x_hist = [x1] 
u_hist = Vector{eltype(x1)}[]

for t = 1:T-1 
    push!(u_hist, policy(θ_sol, x_hist[end], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    push!(x_hist, dynamics(drone, h, x_hist[end], u_hist[end], zeros(nw)))
end

visualize!(vis, drone, [x_hist], [u_hist]; Δt=h, xT=[xT])


plot([x[1] for x in x_hist], [x[2] for x in x_hist], label="", color=:magenta, width=2.0)

