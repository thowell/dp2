function quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w) 
    model = RoboDojo.quadruped

    # dimensions
    nq = model.nq
    nu = model.nu 

    # configurations
    
    q1⁻ = x[1:nq] 
    q2⁻ = x[nq .+ (1:nq)]
    q2⁺ = y[1:nq]
    q3⁺ = y[nq .+ (1:nq)]

    # control 
    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:8)] 
    # ψ = u[nu + 4 + 8 .+ (1:4)] 
    # η = u[nu + 4 + 8 + 4 .+ (1:8)] 
    # sϕ = u[nu + 4 + 8 + 4 + 8 .+ (1:4)]
    # sψ = u[nu + 4 + 8 + 4 + 8 + 4 .+ (1:4)]
    # sα = u[nu + 4 + 8 + 4 + 8 + 4 + 4 .+ (1:1)]
    
    E = [1.0 -1.0] # friction mapping 
    J = RoboDojo.contact_jacobian(model, q2⁺)
    λ = transpose(J[1:8, :]) * [
                                [E * β[1:2]; γ[1]];
                                [E * β[3:4]; γ[2]];
                                [E * β[5:6]; γ[3]];
                                [E * β[7:8]; γ[4]];
                               ]

    [
        q2⁺ - q2⁻;
        RoboDojo.dynamics(model, mass_matrix, dynamics_bias, 
            h, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)
    ]
end

function quadruped_dyn1(mass_matrix, dynamics_bias, h, y, x, u, w)
	model = RoboDojo.quadruped
    [
        quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
        y[2 * model.nq .+ (1:(2 * model.nq))] - x
    ]
end

function quadruped_dynt(mass_matrix, dynamics_bias, h, y, x, u, w)
	model = RoboDojo.quadruped
    [
        quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
        y[2 * model.nq .+ (1:(2 * model.nq))] - x[2 * model.nq .+ (1:(2 * model.nq))]
    ]
end

function contact_constraints_inequality(h, x, u, w) 
    model = RoboDojo.quadruped

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:8)] 
    ψ = u[nu + 4 + 8 .+ (1:4)] 
    η = u[nu + 4 + 8 + 4 .+ (1:8)] 
    sϕ = u[nu + 4 + 8 + 4 + 8 .+ (1:4)]
    sψ = u[nu + 4 + 8 + 4 + 8 + 4 .+ (1:4)]
    sα = u[nu + 4 + 8 + 4 + 8 + 4 + 4 .+ (1:1)]

    ϕ = RoboDojo.signed_distance(model, q3) 
   
    v = (q3 - q2) ./ h[1]
    vT_foot = [(RoboDojo.quadruped_contact_kinematics_jacobians[i](q3) * v)[1] for i = 1:4]
    vT = vcat([[vT_foot[i]; -vT_foot[i]] for i = 1:4]...)
    
    ψ_stack = vcat([ψ[i] * ones(2) for i = 1:4]...)
    
    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)

    [
     γ .* sϕ .- sα;
     β .* η .- sα;
     ψ .* sψ  .- sα;
    ]
end

function contact_constraints_equality(h, x, u, w) 
    model = RoboDojo.quadruped

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:8)] 
    ψ = u[nu + 4 + 8 .+ (1:4)] 
    η = u[nu + 4 + 8 + 4 .+ (1:8)] 
    sϕ = u[nu + 4 + 8 + 4 + 8 .+ (1:4)]
    sψ = u[nu + 4 + 8 + 4 + 8 + 4 .+ (1:4)]
    sα = u[nu + 4 + 8 + 4 + 8 + 4 + 4 .+ (1:1)]

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
   
    v = (q3 - q2) ./ h[1]
    vT_foot = [(RoboDojo.quadruped_contact_kinematics_jacobians[i](q3) * v)[1] for i = 1:4]
    vT = vcat([[vT_foot[i]; -vT_foot[i]] for i = 1:4]...)
    
    ψ_stack = vcat([ψ[i] * ones(2) for i = 1:4]...)
    
    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)

    [
     sϕ - ϕ;
     sψ - fc;
     η - vT - ψ_stack;
    ]
end

# ## permutation matrix
perm = [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
		0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
		0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
		0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
		0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
		0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
		0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]

function initial_configuration(model::RoboDojo.Quadruped, θ1, θ2, θ3)
    q1 = zeros(model.nq)
    q1[3] = 0.0
    q1[4] = -θ1
    q1[5] = θ2

    q1[8] = -θ1
    q1[9] = θ2

    q1[2] = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])

    q1[10] = -θ3
    q1[11] = acos((q1[2] - model.l_thigh2 * cos(q1[10])) / model.l_calf2)

    q1[6] = -θ3
    q1[7] = acos((q1[2] - model.l_thigh2 * cos(q1[6])) / model.l_calf2)

    return q1
end

function ellipse_trajectory(x_start, x_goal, z, T)
	dist = x_goal - x_start
	a = 0.5 * dist
	b = z
	z̄ = 0.0
	x = range(x_start, stop = x_goal, length = T)
	z = sqrt.(max.(0.0, (b^2) * (1.0 .- ((x .- (x_start + a)).^2.0) / (a^2.0))))
	return x, z
end