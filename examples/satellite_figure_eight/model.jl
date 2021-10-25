"""
      Satellite

      Orientation represented with
      Modified Rodrigues Parameters.
"""

struct Satellite{I, T} <: Model{I, T}
      n::Int
      m::Int
      d::Int

      J      # inertia matrix
end

function satellite(model::Satellite, z, u, w)
      # states
      r = view(z, 1:3)
      ω = view(z, 4:6)

      # controls
      τ = view(u, 1:3)

      SVector{6}([0.25 * ((1.0 - r' * r) * ω - 2.0 * cross(ω, r) + 2.0 * (ω' * r) * r);
                  model.J \ (τ - cross(ω, model.J * ω))])
end

function satellite_inertia(model::SatelliteInertia, z, u, w)
    # states
    r = view(z, 1:3)
    ω = view(z, 4:6)

    # controls
    τ = view(u, 1:3)
    J = Diagonal(view(u, 4:6))

    SVector{6}([0.25 * ((1.0 - r' * r) * ω - 2.0 * cross(ω, r) + 2.0 * (ω' * r) * r);
                J \ (τ - cross(ω, J * ω))])
end

function kinematics(model::Satellite, q)
	p = @SVector [0.1, 0.0, 0.0]
	k = MRP(view(q, 1:3)...) * p
	return k
end

# model
n, m, d = 6, 3, 0
J = Diagonal(@SVector[1.0, 2.0, 3.0])

model = Satellite{Midpoint, FixedTime}(n, m, d, J)
