
function visualize!(vis, p::Quadrotor, q; Δt = 0.1)
    default_background!(vis)

    obj_path = joinpath(pwd(),
      "models/quadrotor/drone.obj")
    mtl_path = joinpath(pwd(),
      "models/quadrotor/drone.mtl")

    ctm = ModifiedMeshFileObject(obj_path, mtl_path, scale = 1.0)
    setobject!(vis["drone"], ctm)
    settransform!(vis["drone"], LinearMap(RotZ(pi) * RotX(pi / 2.0)))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim, t) do
            settransform!(vis["drone"],
                  compose(Translation(q[t][1:3]),
                        LinearMap(MRP(q[t][4:6]...) * RotX(pi / 2.0))))
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(0.0, 0.0, 0.0),
    # LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis, anim)
end