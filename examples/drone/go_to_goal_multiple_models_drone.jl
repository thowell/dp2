# PREAMBLE

# PKG_SETUP

# ## Setup

using DirectTrajectoryOptimization
using LinearAlgebra
using Plots
# using JLD2

# ## double integrator 
include("model_drone.jl") 

N = 8
nx = drone.nx
nu = drone.nu
nw = drone.nw

Nx = nx * N 
Nu = nu * N 
Nw = nw * N 

# ## initialization
θ_angle = 0.25
u_hover = hover_controls(drone, θ_angle)

# x1 = [
#         [-0.5; 0.125; 0.0; 0.0; 0.0; 0.0],
#         [0.25; 0.25; 0.0; 0.0; 0.0; 0.0],
#         [0.1; 0.1; 0.0; 0.0; 0.0; 0.0], 
#         [-0.3; 0.05; 0.0; 0.0; 0.0; 0.0],
#         [0.6; -0.5; 0.0; 0.0; 0.0; 0.0],
#         [0.75; -0.6; 0.0; 0.0; 0.0; 0.0],
#         [-0.9; -0.2; 0.0; 0.0; 0.0; 0.0],
#         [0.0; 1.0; 0.0; 0.0; 0.0; 0.0]
#      ]
# xT = [
#         [0.75; 0.85; 0.0; 0.0; 0.0; 0.0],
#         [0.0; 0.0; 0.0; 0.0; 0.0; 0.0],
#         [1.0; 1.0; 0.0; 0.0; 0.0; 0.0],
#         [-1.0; -1.0; 0.0; 0.0; 0.0; 0.0],
#         [-0.6; -0.5; 0.0; 0.0; 0.0; 0.0],
#         [-0.9; -1.0; 0.0; 0.0; 0.0; 0.0],
#         [-0.7; 1.0; 0.0; 0.0; 0.0; 0.0],
#         [0.85; 0.0; 0.0; 0.0; 0.0; 0.0]
#      ]

radius = 1.0
dia = sqrt(2.0) / 2.0 * radius 
# dia^2 + dia^2 - radius^2
x1 = [
        [radius; 0.0; 0.0; 0.0; 0.0; 0.0],
        [-radius; 0.0; 0.0; 0.0; 0.0; 0.0],
        [0.0; radius; 0.0; 0.0; 0.0; 0.0],
        [0.0; -radius; 0.0; 0.0; 0.0; 0.0],
        [dia; dia; 0.0; 0.0; 0.0; 0.0],
        [-dia; -dia; 0.0; 0.0; 0.0; 0.0],
        [-dia; dia; 0.0; 0.0; 0.0; 0.0],
        [dia; -dia; 0.0; 0.0; 0.0; 0.0],
     ]
xT = [
        [-radius; 0.0; 0.0; 0.0; 0.0; 0.0],
        [radius; 0.0; 0.0; 0.0; 0.0; 0.0],
        [0.0; -radius; 0.0; 0.0; 0.0; 0.0],
        [0.0; radius; 0.0; 0.0; 0.0; 0.0],
        [-dia; -dia; 0.0; 0.0; 0.0; 0.0],
        [dia; dia; 0.0; 0.0; 0.0; 0.0],
        [dia; -dia; 0.0; 0.0; 0.0; 0.0],
        [-dia; dia; 0.0; 0.0; 0.0; 0.0],
     ]

@assert length(x1) == N 
@assert length(xT) == N
X1 = vcat(x1...)
XT = vcat(xT...)

# ## (1-layer) multi-layer perceptron policy
l_input = nx 
l1 = 8
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
T = 31

# ## model
h = 0.05

function f1(y, x, u, w)
    ui = [u[nu * (i - 1) .+ (1:nu)] for i = 1:N]
    xi = [x[nx * (i - 1) .+ (1:nx)] for i = 1:N] 
    yi = [y[nx * (i - 1) .+ (1:nx)] for i = 1:N]
    wi = [w[nw * (i - 1) .+ (1:nw)] for i = 1:N]

    θ = u[Nu .+ (1:nθ)] 
    yθ = y[Nx .+ (1:nθ)]

    [
        vcat([dynamics(drone, h, yi[i], xi[i], ui[i], wi[i]) for i = 1:N]...)
        yθ - θ;
    ]
end

function ft(y, x, u, w)
    ui = [u[nu * (i - 1) .+ (1:nu)] for i = 1:N]
    xi = [x[nx * (i - 1) .+ (1:nx)] for i = 1:N] 
    yi = [y[nx * (i - 1) .+ (1:nx)] for i = 1:N]
    wi = [w[nw * (i - 1) .+ (1:nw)] for i = 1:N]

    θ = x[Nx .+ (1:nθ)] 
    yθ = y[Nx .+ (1:nθ)]

    [
        vcat([dynamics(drone, h, yi[i], xi[i], ui[i], wi[i]) for i = 1:N]...)
        yθ - θ;
    ]
end

dyn1 = DirectTrajectoryOptimization.Dynamics(f1, Nx + nθ, Nx, Nu + nθ)
dynt = DirectTrajectoryOptimization.Dynamics(ft, Nx + nθ, Nx + nθ, Nu)

dyn = [dyn1, [dynt for t = 2:T-1]...]
model = DirectTrajectoryOptimization.DynamicsModel(dyn)

# ## objective 
function o1(x, u, w)
    ui = [u[nu * (i - 1) .+ (1:nu)] for i = 1:N]
    xi = [x[nx * (i - 1) .+ (1:nx)] for i = 1:N] 
    wi = [w[nw * (i - 1) .+ (1:nw)] for i = 1:N]
    θ = u[Nu .+ (1:nθ)] 

    J = 0.0
    q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i = 1:N 
        ex = xi[i] - xT[i]
        eu = ui[i] - u_hover
        J += transpose(ex) * Diagonal(q) * ex  ./ N
        J += 1.0e-2 * dot(eu, eu) ./ N
    end
    J += 1.0e-1 * dot(θ, θ)
    return J
end

function ot(x, u, w) 
    ui = [u[nu * (i - 1) .+ (1:nu)] for i = 1:N]
    xi = [x[nx * (i - 1) .+ (1:nx)] for i = 1:N] 
    wi = [w[nw * (i - 1) .+ (1:nw)] for i = 1:N]
    θ = x[Nx .+ (1:nθ)] 

    J = 0.0
    q = [10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
    for i = 1:N
        ex = xi[i] - xT[i]
        eu = ui[i] - u_hover
        J += transpose(ex) * Diagonal(q) * ex  ./ N
        J += 1.0e-2 * dot(eu, eu) 
    end
    J += 1.0e-1 * dot(θ, θ)
    return J 
end

function ott(x, u, w) 
    ui = [u[nu * (i - 1) .+ (1:nu)] for i = 1:N]
    xi = [x[nx * (i - 1) .+ (1:nx)] for i = 1:N] 
    wi = [w[nw * (i - 1) .+ (1:nw)] for i = 1:N]
    θ = x[Nx .+ (1:nθ)] 

    J = 0.0
    q = [1000.0; 1000.0; 1000.0; 100.0; 100.0; 100.0]
    for i = 1:N
        ex = xi[i] - xT[i]
        eu = ui[i] - u_hover
        J += transpose(ex) * Diagonal(q) * ex  ./ N
        J += 1.0e-2 * dot(eu, eu) 
    end
    J += 1.0e-1 * dot(θ, θ)
    return J 
end

function oT(x, u, w) 
    xi = [x[nx * (i - 1) .+ (1:nx)] for i = 1:N] 
    wi = [w[nw * (i - 1) .+ (1:nw)] for i = 1:N]
    θ = x[Nx .+ (1:nθ)]

    J = 0.0
    q = [1000.0; 1000.0; 1500.0; 1000.0; 1000.0; 1000.0]
    for i = 1:N
        ex = xi[i] - xT[i]
        J += transpose(ex) * Diagonal(q) * ex  ./ N
    end
    J += 1.0e-1 * dot(θ, θ)
    return J 
end

c1 = DirectTrajectoryOptimization.Cost(o1, Nx, Nu + nθ, Nw, [1])
ct = DirectTrajectoryOptimization.Cost(ot, Nx + nθ, Nu, Nw, [t for t = 2:15])
ctt = DirectTrajectoryOptimization.Cost(ott, Nx + nθ, Nu, Nw, [t for t = 16:T-1])
cT = DirectTrajectoryOptimization.Cost(oT, Nx + nθ, 0, Nw, [T])
obj = [c1, ct, ctt, cT]

# ## constraints
ul = [0.0; -0.5 * π; 0.0; -0.5 * π]  
uu = [10.0; 0.5 * π; 10.0; 0.5 * π]
Ul = vcat([ul for i = 1:N]...) 
Uu = vcat([uu for i = 1:N]...)
bnd1 = Bound(Nx, Nu + nθ, [1], xl=X1, xu=X1, ul=[Ul; -Inf * ones(nθ)], uu=[Uu; Inf * ones(nθ)])
bndt = Bound(Nx + nθ, Nu, [t for t = 2:T-1], ul=Ul, uu=Uu)
# bndT = Bound(Nx + nθ, 0, [T], xl=[XT; -Inf * ones(nθ)], xu=[XT; Inf * ones(nθ)])

function policy1(x, u, w) 
    ui = [u[nu * (i - 1) .+ (1:nu)] for i = 1:N]
    xi = [x[nx * (i - 1) .+ (1:nx)] for i = 1:N] 
    θ = u[Nu .+ (1:nθ)]
    vcat([ui[i] - policy(θ, xi[i], xT[i]) for i = 1:N]...)
end

function policyt(x, u, w) 
    ui = [u[nu * (i - 1) .+ (1:nu)] for i = 1:N]
    xi = [x[nx * (i - 1) .+ (1:nx)] for i = 1:N] 
    θ = x[Nx .+ (1:nθ)]
    vcat([ui[i] - policy(θ, xi[i], xT[i]) for i = 1:N]...)
end

con_policy1 = StageConstraint(policy1, Nx, Nu + nθ, Nw, [1], :equality)
con_policyt = StageConstraint(policyt, Nx + nθ, Nu, Nw, [t for t = 2:T-1], :equality)

cons = ConstraintSet([bnd1, bndt], 
    [con_policy1, con_policyt])

# ## problem 
trajopt = TrajectoryOptimizationProblem(obj, model, cons)
s = Solver(trajopt, 
    options=Options(
        max_cpu_time=2500.0,
        max_iter=2500,
        tol=1.0e-3,
        constr_viol_tol=1.0e-3,
    ))

# ## initialize
θ0 = randn(nθ)
U_hover = vcat([u_hover for i = 1:N]...)
u_guess = [[U_hover; θ0], [U_hover for t = 2:T-1]...]
z0 = zeros(s.p.num_var)
for (t, idx) in enumerate(s.p.trajopt.model.idx.x)
    z0[idx] = t == 1 ? X1 : [X1; randn(nθ)]
end
for (t, idx) in enumerate(s.p.trajopt.model.idx.u)
    z0[idx] = u_guess[t]
end
initialize!(s, z0)

# ## solve
@time DirectTrajectoryOptimization.solve!(s)

# ## solution
X_sol = [[x[(i - 1) * nx .+ (1:nx)] for x in trajopt.x] for i = 1:N]
U_sol = [[u_hover, [u[(i - 1) * nu .+ (1:nu)] for u in trajopt.u[1:(end-1)]]...] for i = 1:N]
θ_sol = trajopt.u[1][Nu .+ (1:nθ)]

# # ## state
# plt = plot();
# for i = 1:N
#     plt = plot!(hcat(X_sol[i]...)', label="", color=:orange, width=2.0)
# end
# display(plt) 

# # ## control
# plt = plot();
# for i = 1:N
#     plt = plot!(hcat(U_sol[i]..., U_sol[i][end])', linetype = :steppost)
# end
# display(plt) 

# # ## plot xy 
# plt = plot(); 
# for i = 1:N
#     plt = plot!([x[1] for x in X_sol[i]], [x[2] for x in X_sol[i]], label="", width=2.0)
# end  
# display(plt)

# ## visualization 
include("visuals_drone.jl")
vis = Visualizer()
open(vis) 
visualize_drone!(vis, drone, X_sol, U_sol; Δt=h, xT=xT)

# ## simulate policy 
i = 1
x_init = x1[i] 
x_goal = xT[i]
x_init = [0.0; -0.6; 0.0; 0.0; 0.0; 0.0]
x_goal = [0.5; 0.5; 0.0; 0.0; 0.0; 0.0]
x_hist = [x_init] 
u_hist = [u_hover]

for t = 1:5 * T
    push!(u_hist, policy(θ_sol, x_hist[end], x_goal))
    push!(x_hist, dynamics(drone, h, x_hist[end], u_hist[end], zeros(nw)))
end

visualize_drone!(vis, drone, [x_hist], [u_hist]; Δt=h, xT=[x_goal])

# ## save policy
# @save joinpath(@__DIR__, "policy.jld2") θ_sol
@load joinpath(@__DIR__, "policy.jld2") θ_sol
