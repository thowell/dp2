function visualize!(vis, model::Quadruped, q;
    r = 0.025, Δt = 0.1)

    default_background!(vis)

    torso = Cylinder(Point3f0(0.0, 0.0, 0.0), Point3f0(0.0, 0.0, model.l_torso),
        convert(Float32, 0.035))
    setobject!(vis["torso"], torso,
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    thigh_1 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh1),
        convert(Float32, 0.0175))
    setobject!(vis["thigh1"], thigh_1,
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    calf_1 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf1),
        convert(Float32, 0.0125))
    setobject!(vis["leg1"], calf_1,
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    thigh_2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh2),
        convert(Float32, 0.0175))
    setobject!(vis["thigh2"], thigh_2,
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    calf_2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf2),
        convert(Float32, 0.0125))
    setobject!(vis["leg2"], calf_2,
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    thigh_3 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh3),
        convert(Float32, 0.0175))
    setobject!(vis["thigh3"], thigh_3,
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    calf_3 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf3),
        convert(Float32, 0.0125))
    setobject!(vis["leg3"], calf_3,
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    thigh_4 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh4),
        convert(Float32, 0.0175))
    setobject!(vis["thigh4"], thigh_4,
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    calf_4 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf4),
        convert(Float32, 0.0125))
    setobject!(vis["leg4"], calf_4,
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    hip1 = setobject!(vis["hip1"], Sphere(Point3f0(0),
        convert(Float32, 0.035)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

    hip2 = setobject!(vis["hip2"], Sphere(Point3f0(0),
        convert(Float32, 0.035)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

    knee1 = setobject!(vis["knee1"], Sphere(Point3f0(0),
        convert(Float32, 0.025)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

    knee2 = setobject!(vis["knee2"], Sphere(Point3f0(0),
        convert(Float32, 0.025)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

    knee3 = setobject!(vis["knee3"], Sphere(Point3f0(0),
        convert(Float32, 0.025)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

    knee4 = setobject!(vis["knee4"], Sphere(Point3f0(0),
        convert(Float32, 0.025)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

    feet1 = setobject!(vis["feet1"], Sphere(Point3f0(0),
        convert(Float32, r)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

    feet2 = setobject!(vis["feet2"], Sphere(Point3f0(0),
        convert(Float32, r)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

    feet3 = setobject!(vis["feet3"], Sphere(Point3f0(0),
        convert(Float32, r)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

    feet4 = setobject!(vis["feet4"], Sphere(Point3f0(0),
        convert(Float32, r)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

    T = length(q)
    p_shift = [0.0, 0.0, r]
    for t = 1:T
        MeshCat.atframe(anim, t) do
            p = [q[t][1]; 0.0; q[t][2]] + p_shift

            k_torso = kinematics_1(model, q[t], body = :torso, mode = :ee)
            p_torso = [k_torso[1], 0.0, k_torso[2]] + p_shift

            k_thigh_1 = kinematics_1(model, q[t], body = :thigh_1, mode = :ee)
            p_thigh_1 = [k_thigh_1[1], 0.0, k_thigh_1[2]] + p_shift

            k_calf_1 = kinematics_2(model, q[t], body = :calf_1, mode = :ee)
            p_calf_1 = [k_calf_1[1], 0.0, k_calf_1[2]] + p_shift

            k_thigh_2 = kinematics_1(model, q[t], body = :thigh_2, mode = :ee)
            p_thigh_2 = [k_thigh_2[1], 0.0, k_thigh_2[2]] + p_shift

            k_calf_2 = kinematics_2(model, q[t], body = :calf_2, mode = :ee)
            p_calf_2 = [k_calf_2[1], 0.0, k_calf_2[2]] + p_shift


            k_thigh_3 = kinematics_2(model, q[t], body = :thigh_3, mode = :ee)
            p_thigh_3 = [k_thigh_3[1], 0.0, k_thigh_3[2]] + p_shift

            k_calf_3 = kinematics_3(model, q[t], body = :calf_3, mode = :ee)
            p_calf_3 = [k_calf_3[1], 0.0, k_calf_3[2]] + p_shift

            k_thigh_4 = kinematics_2(model, q[t], body = :thigh_4, mode = :ee)
            p_thigh_4 = [k_thigh_4[1], 0.0, k_thigh_4[2]] + p_shift

            k_calf_4 = kinematics_3(model, q[t], body = :calf_4, mode = :ee)
            p_calf_4 = [k_calf_4[1], 0.0, k_calf_4[2]] + p_shift

            settransform!(vis["thigh1"], cable_transform(p, p_thigh_1))
            settransform!(vis["leg1"], cable_transform(p_thigh_1, p_calf_1))
            settransform!(vis["thigh2"], cable_transform(p, p_thigh_2))
            settransform!(vis["leg2"], cable_transform(p_thigh_2, p_calf_2))
            settransform!(vis["thigh3"], cable_transform(p_torso, p_thigh_3))
            settransform!(vis["leg3"], cable_transform(p_thigh_3, p_calf_3))
            settransform!(vis["thigh4"], cable_transform(p_torso, p_thigh_4))
            settransform!(vis["leg4"], cable_transform(p_thigh_4, p_calf_4))
            settransform!(vis["torso"], cable_transform(p, p_torso))
            settransform!(vis["hip1"], Translation(p))
            settransform!(vis["hip2"], Translation(p_torso))
            settransform!(vis["knee1"], Translation(p_thigh_1))
            settransform!(vis["knee2"], Translation(p_thigh_2))
            settransform!(vis["knee3"], Translation(p_thigh_3))
            settransform!(vis["knee4"], Translation(p_thigh_4))
            settransform!(vis["feet1"], Translation(p_calf_1))
            settransform!(vis["feet2"], Translation(p_calf_2))
            settransform!(vis["feet3"], Translation(p_calf_3))
            settransform!(vis["feet4"], Translation(p_calf_4))
        end
    end

    settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(-pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end
