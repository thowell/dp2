"""
      Satellite

      Orientation represented with
      Modified Rodrigues Parameters.
"""

struct Satellite{T}
      n::Int
      m::Int
      d::Int
      J::Diagonal{T,Vector{T}} # inertia matrix
end

function satellite(model::Satellite, y, x, u, w)
      q1⁻ = x[1:4] 
      q2⁻ = x[4 .+ (1:4)] 
      q2⁺ = y[1:4] 
      q3⁺ = y[4 .+ (1:4)] 

      

end

function satellite_inertia(model::SatelliteInertia, z, u, w)
      q1⁻ = x[1:4] 
      q2⁻ = x[4 .+ (1:4)] 
      q2⁺ = y[1:4] 
      q3⁺ = y[4 .+ (1:4)] 
end

function kinematics(model::Satellite, q)
	p = [0.1, 0.0, 0.0]
      k = rotate(p, q)
	return k
end

# model
n, m, d = 7, 3, 0
J = Diagonal(@SVector[1.0, 2.0, 3.0])

model = Satellite{Midpoint, FixedTime}(n, m, d, J)
