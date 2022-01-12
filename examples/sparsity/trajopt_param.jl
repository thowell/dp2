using Symbolics 
using LinearAlgebra
using Plots

# dimensions
T = 5

n = [t == 1 ? 1 : 2 for t = 1:T] 
m = [t == 1 ? 2 : 1 for t = 1:T-1]

# base
nz = sum(n) + sum(m)
ny = sum(n[2:T]) 
na = nz + ny

function unpack(z, y) 
    x = Vector{eltype(z)}[] 
    u = Vector{eltype(z)}[]
    λ = Vector{eltype(y)}[]
    shift_z = 0
    shift_y = 0

    for t = 1:T-1 
        xt = z[shift_z .+ (1:n[t])] 
        push!(x, xt)
        shift_z += n[t]
        ut = z[shift_z .+ (1:m[t])] 
        push!(u, ut)
        shift_z += m[t]

        λt = y[shift_y .+ (1:n[t + 1])] 
        push!(λ, λt)
        shift_y += n[t + 1]
    end

    xT = z[shift_z .+ (1:n[T])]
    push!(x, xT) 
    shift_z += n[T]

    return x, u, λ
end

z0 = rand(nz)
y0 = rand(ny)
x0, u0, λ0 = unpack(z0, y0)

# problem 
dynamics(x, u, θ) = x + u + θ
cost(x, u, θ) = 0.5 * dot(x, x) + 0.5 * dot(u, u) + dot(x, u) + 0.5 * dot(θ, θ) + dot(x, θ) + dot(u, θ)
cost_terminal(x, θ) = 0.5 * dot(x, x) + 0.5 * dot(θ, θ) + dot(x, θ)

function lagrangian(z, y) 
    x, u, λ = unpack(z, y) 
    L = 0 
    for t = 1:T-1 
        xt = x[t][1:1]
        ut = u[t][1:1]
        xtt = x[t+1][1:1]
        θ = (t == 1 ? u[t][2:2] : x[t][2:2])
        L += cost(xt, ut, θ) 
        L += dot(λ[t], [dynamics(x[t][1:1], u[t][1:1], θ) - x[t+1][1:1]; 
                        x[t+1][2:2] - θ]) 
    end
    θ = x[T][2:2]
    L += cost_terminal(x[T][1:1], θ)
    return L
end

@variables z[1:nz] y[1:ny]

l = lagrangian(z, y)
dl = Symbolics.gradient(l, [z; y])
ddl = Symbolics.jacobian(dl, [z; y])
ddl_func = eval(Symbolics.build_function(ddl, z, y)[1])

plot(Gray.(ones(nz + ny, nz + ny) - abs.(ddl_func(ones(nz), ones(ny)))))








