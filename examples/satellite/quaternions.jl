# skew-symmetric matrix
function hat(x)
    return [0 -x[3] x[2];
            x[3] 0 -x[1];
           -x[2] x[1] 0]
end

# left quaternion multiply as matrix
function L_mult(x)
    [x[1] -transpose(x[2:4]); 
     x[2:4] x[1] * I(3) + hat(x[2:4])]
end

# right quaternion multiply as matrix
function R_mult(x)
    [x[1] -transpose(x[2:4]); x[2:4] x[1] * I(3) - hat(x[2:4])]
end

# https://roboticexplorationlab.org/papers/planning_with_attitude.pdf
function attitude_jacobian(x)
    H = [zeros(1, 3); I(3)]
    return L_mult(x) * H
end

# rotation matrix
function rotation_matrix(q) 
    H = [zeros(1, 3); I(3)]
    transpose(H) * L_mult(q) * transpose(R_mult(q)) * H
end

# right discrete Legendre transform for a free rigid body
function DLT1(h, J, q1, q2)
    H = [zeros(1, 3); I(3)]
    T = Diagonal([1.0; -1; -1; -1]) 
    (2.0 / h) * transpose(L_mult(q1) * H) * T * transpose(R_mult(q2)) * H * J * transpose(H) * transpose(L_mult(q1)) * q2
end

# left discrete Legendre transform for a free rigid body
function DLT2(h, J, q1, q2)
    H = [zeros(1, 3); I(3)]    
    (2.0 / h) * transpose(L_mult(q2) * H) * L_mult(q1) * H * J * transpose(H) * transpose(L_mult(q1)) * q2
end

# discrete Euler-Lagrange equation for a free rigid body
function integrator(h, J, q1, q2, q3, τ)   
    DLT2(h, J, q1, q2) + DLT1(h, J, q2, q3) + τ
end

# finite-difference angular velocity 
function angular_velocity(h, q1, q2) 
    H = [zeros(1, 3); I(3)]
    2.0 * transpose(H) * transpose(L_mult(q1)) * (q2 - q1) / h
end

## test 
# variables
@variables q1[1:4] q2[1:4] q3[1:4] h[1:1]

# finite-difference angular velocities
ω1 = vec(angular_velocity(h, q1, q2))
ω2 = vec(angular_velocity(h, q2, q3))

# Lagrangian 
J = Diagonal([1.0, 2.0, 3.0])
L = h[1] * (0.5 * transpose(ω1) * J * ω1 + 0.5 * transpose(ω2) * J * ω2)
∂L∂q2 = Symbolics.gradient(L, q2)

# attitude Jacobian 
G = attitude_jacobian(q2)

# integrator 
f = transpose(G) * ∂L∂q2
f_func = eval(Symbolics.build_function(f, q1, q2, q3, h)[1])

# test random values 
h0 = 0.01
qb = rand(4) 
qb ./= norm(qb) 
ωa = randn(3) 
ωb = randn(3)
qc = qb + 0.5 * h0 * L_mult(qb) * [zeros(1, 3); I(3)] * ωb
qc ./= norm(qc)
qa = qb + 0.5 * h0 * L_mult(qb) * [zeros(1, 3); I(3)] * (-ωa)
qa ./= norm(qa) 

@assert abs(norm(qa) - 1.0) < 1.0e-8
@assert abs(norm(qb) - 1.0) < 1.0e-8
@assert abs(norm(qc) - 1.0) < 1.0e-8

@assert norm(2.0 / h0 * [zeros(1, 3); I(3)]' * L_mult(Diagonal([1.0; -1; -1; -1]) * qa) * qb - ωa) < 1.0e-4
@assert norm(2.0 / h0 * [zeros(1, 3); I(3)]' * L_mult(Diagonal([1.0; -1; -1; -1]) * qb) * qc - ωb) < 1.0e-4

i1 = f_func(qa, qb, qc, h0)
i2 = (4.0 / h0) * attitude_jacobian(qb)' * L_mult(qa) * [zeros(1, 3); I(3)] * J * [zeros(1, 3); I(3)]' * L_mult(qa)' * qb + (4.0 / h0) * attitude_jacobian(qb)' * Diagonal([1.0; -1; -1; -1]) * R_mult(qc)' * [zeros(1, 3); I(3)] * J * [zeros(1, 3); I(3)]' * L_mult(qb)' * qc

norm(i1 - i2)

function integrator_3(h, ωa, ωb)
    D1 = -J * ωa * sqrt(4.0 / h^2.0 - ωa' * ωa) + hat(ωa) * J * ωa
    D2 = J * ωb * sqrt(4.0 / h^2.0 - ωb' * ωb) + hat(ωb) * J * ωb
    h * (D2 + D1)
end

function integrator_4(h, qa, qb, qc) 
    ωa = angular_velocity(h, qa, qb)
    ωb = angular_velocity(h, qb, qc) 
    integrator_3(h, ωa, ωb) 
end

i3 = integrator_3(h0, ωa, ωb)
i4 = integrator_4(h0, qa, qb, qc)

norm(i3 - i4)

norm(i2 + i3)
