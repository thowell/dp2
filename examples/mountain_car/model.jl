"""
    Mountain Car
    Reinforcement Learning: An Introduction - Sutton & Barto 
        example 9.2
"""

struct MountainCar
    nx::Int
    nu::Int
    nw::Int
end

function dynamics(model::MountainCar, h, x, u, w)
    [
        x[1] + x[2] * h[1];
        x[2] + (0.001 * u[1] - 0.0025 * cos(3.0 * x[1])) * h[1]
    ]
end

function dynamics(model::MountainCar, h, y, x, u, w)
    y[1:2] - dynamics(model, h, x, u, w)
end

nx, nu, nw = 2, 1, 0

mountain_car = MountainCar(nx, nu, nw)

