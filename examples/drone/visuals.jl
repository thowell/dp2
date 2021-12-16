function visualize_double_integrator_2D!(vis, Q;
    Δt=0.1,
    r=0.1, 
    xT=[zero(q[end]) for q in Q],
    color=RGBA(1,0,0,1.0),
    color_goal=RGBA(0,1,0,1.0))

    N = length(Q)
    T = length(Q[1])

    setvisible!(vis["/Background"], true)
	setprop!(vis["/Background"], "top_color", RGBA(1.0, 1.0, 1.0, 1.0))
	setprop!(vis["/Background"], "bottom_color", RGBA(1.0, 1.0, 1.0, 1.0))
	setvisible!(vis["/Axes"], false)
	setvisible!(vis["/Grid"], false)

    sphere = GeometryBasics.Sphere(Point(0.0,0.0,0.0), r)
    sphere_goal = GeometryBasics.Sphere(Point(0.0,0.0,0.0), r)

    for i = 1:N
        setobject!(vis["goal_$i"], sphere_goal, MeshPhongMaterial(color=color_goal))
        settransform!(vis["goal_$i"], Translation([xT[i][1]; 0.0; xT[i][2]]))

        setobject!(vis["d$i"], sphere, MeshPhongMaterial(color=color))
    end

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    
    for t = 1:T 
        MeshCat.atframe(anim,t) do
            for i = 1:N
                settransform!(vis["d$i"], Translation([Q[i][t][1]; 0.0; Q[i][t][2]]))
            end
        end
    end

    # set camera
    settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, -50.0, -1.0), LinearMap(RotZ(- pi / 2))))
    setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 15)

    MeshCat.setanimation!(vis,anim)
end