using Symbolics 
using LinearAlgebra
using Plots

# dimensions
T = 5

n = [1 for t = 1:T] 
m = [1 for t = 1:T-1]
p = 1

# base
nz = sum(n) + sum(m) + p
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

    θ = z[sum(n) + sum(m) .+ (1:p)]

    return x, u, λ, θ
end

z0 = rand(nz)
y0 = rand(ny)
x0, u0, λ0 = unpack(z0, y0)

# problem 
dynamics(x, u, θ) = x + u + θ
cost(x, u, θ) = 0.5 * dot(x, x) + 0.5 * dot(u, u) + dot(x, u) + 0.5 * dot(θ, θ) + dot(x, θ) + dot(u, θ)
cost_terminal(x, θ) = 0.5 * dot(x, x) + 0.5 * dot(θ, θ) + dot(x, θ)

function lagrangian(z, y) 
    x, u, λ, θ = unpack(z, y) 
    L = 0 
    for t = 1:T-1 
        xt = x[t]
        ut = u[t]
        xtt = x[t+1]
        L += cost(xt, ut, θ) 
        L += dot(λ[t], dynamics(xt, ut, θ) - xtt)          
    end
    L += cost_terminal(x[T], θ)
    return L
end

@variables z[1:nz] y[1:ny]

l = lagrangian(z, y)
dl = Symbolics.gradient(l, [z; y])
ddl = Symbolics.jacobian(dl, [z; y])
ddl_func = eval(Symbolics.build_function(ddl, z, y)[1])

spar = ones(nz + ny, nz + ny) - abs.(ddl_func(ones(nz), ones(ny)))
plot(Gray.(spar))

## 
A = ones(22, 22)
A[22-na+1:end, 22-na+1:end] = spar
plot(Gray.(A))

