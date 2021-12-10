include("quaternions.jl") 

"""
      Satellite
"""

struct Satellite{T}
      nx::Int
      nu::Int
      nw::Int
      J::Diagonal{T,Vector{T}} # inertia matrix
end

function dynamics(model::Satellite, h, y, x, u, w)
      q1⁻ = x[1:4] 
      q2⁻ = x[4 .+ (1:4)] 
      q2⁺ = y[1:4] 
      q3 = y[4 .+ (1:4)] 
      τ = u[1:3]

      [
       q2⁺ - q2⁻;
       integrator(h, model.J, q1⁻, q2⁺, q3, τ); 
       1.0 - (q3[1]^2 + q3[2]^2 + q3[3]^2 + q3[4]^2)^0.5;
      ]
end

function dynamics_inertia(model::Satellite, h, z, u, w)
      q1⁻ = x[1:4] 
      q2⁻ = x[4 .+ (1:4)] 
      q2⁺ = y[1:4] 
      q3 = y[4 .+ (1:4)] 
      τ = u[1:3] 
      J = Diagonal(u[3 .+ (1:3)])

      [
       q2⁺ - q2⁻;
       integrator(h, J, q1⁻, q2⁺, q3, τ); 
       1.0 - (q3[1]^2 + q3[2]^2 + q3[3]^2 + q3[4]^2)^0.5;
      ]
end

function kinematics(model::Satellite, q)
	p = [0.1, 0.0, 0.0]
      k = rotation_matrix(q) * p
	return k
end

# model
satellite = Satellite(8, 3, 0, Diagonal([1.0, 2.0, 3.0]))
