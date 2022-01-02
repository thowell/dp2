"""
    Mountain Car

    https://en.wikipedia.org/wiki/Mountain_car_problem
"""

struct MountainCar{T}
    nx::Int
    nu::Int
    nw::Int

    gravity::T
    mass::T
    steepness::T
    friction::T
end

function dynamics(model::MountainCar, h, y, x, u, w)
    y - [x[1] + x[2] * h;
         x[2] + (model.gravity * model.mass * cos(model.steepness * x[1]) 
            + u[1] / model.mass - model.friction * x[2]) * h]
end

nx, nu, nw = 2, 1, 0
gravity = 9.81
mass = 0.2
steepness = 3.0
friction = 0.3
xl = [-1.2; -1.5]
xu = [0.5; 1.5]
ul = [-2.0]
uu = [2.0]
model = MountainCar(nx, nu, nw, gravity, mass, steepness, friction)

