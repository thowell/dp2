using Symbolics 
using LinearAlgebra
using Plots

# dimensions
n = 1 
m = 1 
p = 0 
T = 5

# base
nz = T * n + (T - 1) * m
ny = (T - 1) * n 
na = nz + ny

function unpack(z, y) 
    x = Vector{eltype(z)}[] 
    u = Vector{eltype(z)}[]
    λ = Vector{eltype(y)}[]
    shift_z = 0
    shift_y = 0

    for t = 1:T-1 
        xt = z[shift_z .+ (1:n)] 
        push!(x, xt)
        shift_z += n 
        ut = z[shift_z .+ (1:m)] 
        push!(u, ut)
        shift_z += m 

        λt = y[shift_y .+ (1:n)] 
        push!(λ, λt)
        shift_y += n
    end

    xT = z[shift_z .+ (1:n)]
    push!(x, xT) 
    shift_z += n

    return x, u, λ
end

z0 = rand(nz)
y0 = rand(ny)
x0, u0, λ0 = unpack(z0, y0)

# problem 
dynamics(x, u) = x + u
cost(x, u) = 0.5 * dot(x, x) + 0.5 * dot(u, u) + dot(x, u)
cost_terminal(x) = 0.5 * dot(x, x) 
policy(θ, x) = -θ * x 

function lagrangian(z, y) 
    x, u, λ = unpack(z, y) 
    L = 0 
    for t = 1:T-1 
        L += cost(x[t], u[t]) 
        L += dot(λ[t], dynamics(x[t], u[t]) - x[t+1]) 
    end
    L += cost_terminal(x[T])
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









