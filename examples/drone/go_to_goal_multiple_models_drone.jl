# PREAMBLE

# PKG_SETUP

# ## Setup

using DirectTrajectoryOptimization
using LinearAlgebra
using Plots

# ## double integrator 
include("model_drone.jl") 

N = 4
nx = drone.nx
nu = drone.nu
nw = drone.nw

Nx = nx * N 
Nu = nu * N 
Nw = nw * N 

# ## initialization
x1 = [
        [-0.5; 0.125; 0.0; 0.0; 0.0; 0.0],
        [0.25; 0.25; 0.0; 0.0; 0.0; 0.0],
        [0.0; 0.0; 0.0; 0.0; 0.0; 0.0], 
        [0.0; 0.0; 0.0; 0.0; 0.0; 0.0],
     ]
xT = [
        [0.75; 0.85; 0.0; 0.0; 0.0; 0.0],
        [0.0; 0.0; 0.0; 0.0; 0.0; 0.0],
        [1.0; 1.0; 0.0; 0.0; 0.0; 0.0],
        [-1.0; -1.0; 0.0; 0.0; 0.0; 0.0]
     ]
@assert length(x1) == N 
@assert length(xT) == N
X1 = vcat(x1...)
XT = vcat(xT...)

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
    for i = 1:N 
        J += 1.0 * dot(xi[i] - xT[i], xi[i] - xT[i]) ./ N
        # J += 0.0 * dot(ui[i], ui[i]) 
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
    for i = 1:N
        J += 1.0 * dot(xi[i] - xT[i], xi[i] - xT[i]) ./ N
        # J += 0.0 * dot(ui[i], ui[i]) 
    end
    J += 1.0e-1 * dot(θ, θ)
    return J 
end

function oT(x, u, w) 
    xi = [x[nx * (i - 1) .+ (1:nx)] for i = 1:N] 
    wi = [w[nw * (i - 1) .+ (1:nw)] for i = 1:N]
    θ = x[Nx .+ (1:nθ)]

    J = 0.0
    for i = 1:N
        J += 1000.0 * dot(xi[i] - xT[i], xi[i] - xT[i]) ./ N
    end
    J += 1.0e-1 * dot(θ, θ)
    return J 
end

c1 = DirectTrajectoryOptimization.Cost(o1, Nx, Nu + nθ, Nw, [1])
ct = DirectTrajectoryOptimization.Cost(ot, Nx + nθ, Nu, Nw, [t for t = 2:T-1])
cT = DirectTrajectoryOptimization.Cost(oT, Nx + nθ, 0, Nw, [T])
obj = [c1, ct, cT]

# ## constraints
ul = [0.0; -0.5 * π; 0.0; -0.5 * π]  
uu = [100.0; 0.5 * π; 100.0; 0.5 * π]
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
        tol=1.0e-3,
        constr_viol_tol=1.0e-3,
    ))

# ## initialize
θ0 = randn(nθ)
u_guess = [[0.001 * randn(Nu); θ0], [0.001 * randn(Nu) for t = 2:T-1]...]
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
U_sol = [[u[(i - 1) * nu .+ (1:nu)] for u in trajopt.u[1:(end-1)]] for i = 1:N]
θ_sol = trajopt.u[1][Nu .+ (1:nθ)]

# ## state
plt = plot();
for i = 1:N
    plt = plot!(hcat(X_sol[i]...)', label="", color=:orange, width=2.0)
end
display(plt) 

# ## control
plt = plot();
for i = 1:N
    plt = plot!(hcat(U_sol[i]..., U_sol[i][end])', linetype = :steppost)
end
display(plt) 

# ## plot xy 
plt = plot(); 
for i = 1:N
    plt = plot!([x[1] for x in X_sol[i]], [x[2] for x in X_sol[i]], label="", width=2.0)
end  
display(plt)

# ## visualization 
include("visuals.jl")
vis = Visualizer()
open(vis) 
visualize!(vis, drone, X_sol, U_sol; Δt=h, xT=xT)

# ## simulate policy 
i = 1
x_hist = [x1[1]] 
u_hist = Vector{eltype(x1[i])}[]

for t = 1:T-1 
    push!(u_hist, policy(θ_sol, x_hist[end], 
        xT[i]
        ))
    push!(x_hist, dynamics(drone, h, x_hist[end], u_hist[end], zeros(nw)))
end

plot([x[1] for x in x_hist], [x[2] for x in x_hist], label="", color=:magenta, width=2.0)

