include("quaternions.jl") 

"""
      Satellite
"""

struct Satellite{T}
      nx::Int
      nu::Int
      nw::Int
      m::T
      J::Diagonal{T,Vector{T}} # inertia matrix
      dim::Vector{T}
end

# continuous-time dynamics
function dynamics(model::Satellite, J, z, u, w)
      # states
      r = z[1:3]
      ω = z[4:6]

      # controls
      τ = u[1:3]

      [0.25 * ((1.0 - r' * r) * ω - 2.0 * cross(ω, r) + 2.0 * (ω' * r) * r);
       J \ (τ - cross(ω, J * ω))]
end

# explicit midpoint dynamics 
function dynamics(model::Satellite, h, J, x, u, w)
      x + h * dynamics(model, J, x + 0.5 * h * dynamics(model, J, x, u, w), u, w)
end

# implicit midpoint dynamics 
function dynamics(model::Satellite, h, J, y, x, u, w)
      h = 0.05 # timestep 
      y - (x + h * dynamics(model, J, 0.5 * (x + y), u, w))
end

function kinematics(model::Satellite, x)
	p = [0.1, 0.0, 0.0]
	k = Rotations.MRP(x[1:3]...) * p
      # k = rotation_matrix(cayley(x[1:3])) * p
	return k
end

# model
nx, nu, nw = 6, 3, 0
satellite = Satellite(nx, nu, nw, 6.0, Diagonal([1.0, 1.0, 1.0]), [1.0, 1.0, 1.0])
