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

# rotate vector x by quaternion q
function rotate(x, q) 
    H = [zeros(1, 3); I(3)]
    transpose(H) * L_mult(q) * transpose(R_mult(q)) * H * x 
end

# right discrete Legendre transform for a free rigid body
function DLT1(h, J, q1, q2)
    H = [zeros(1, 3); I(3)]
    (2.0 / h) * transpose(attitude_jacobian(q1)) * T * transpose(R_mult(q2)) * H * J * transpose(H) * transpose(L_mult(q1)) * q2
end

# left discrete Legendre transform for a free rigid body
function DLT2(h, J, q1,q2)
    H = [zeros(1, 3); I(3)]    
    (2.0 / h) * transpose(attitude_jacobian(q2)) * L_mult(q1) * H * J * transpose(H) * transpose(L(q1)) * q2
end

# discrete Euler-Lagrange equation for a free rigid body
function integrator(h, J, q1, q2, q3, τ)   
    DLT2(h, J, q1, q2) + DLT1(h, J, q2, q3) + τ
end