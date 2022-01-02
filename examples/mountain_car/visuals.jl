function visualize!(vis, model::MountainCar, x; Δt=0.1)

    default_background!(vis)

    obj_path = joinpath(pwd(),"models/cybertruck/cybertruck.obj")
    mtl_path = joinpath(pwd(),"models/cybertruck/cybertruck.mtl")

    m = ModifiedMeshFileObject(obj_path, mtl_path, scale=0.05)
    setobject!(vis["car"], m)

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(x)
        MeshCat.atframe(anim,t) do
            settransform!(vis["car"],
                compose(Translation(x[t][1], x[t][2], 0.0),
                    LinearMap(RotZ(x[t][3] + pi) * RotX(pi / 2.0))))
        end
    end

    settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, 0.0, 0.0),LinearMap(RotY(-pi/2.5))))

    MeshCat.setanimation!(vis, anim)
end
