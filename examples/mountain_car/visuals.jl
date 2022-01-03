function visualize_mountain_car!(vis, model::MountainCar, x; mesh=false, Δt=0.1)

    N = 100
    x_domain = range(xl[1], stop=xu[1], length=N)
    z_domain = sin.(3.0 .* x_domain)
    points = Vector{Point{3,Float64}}()
    for i = 1:N
        push!(points, Point(x_domain[i], 0.0, z_domain[i]))
    end
    
    line_mat = LineBasicMaterial(color=color=RGBA(0.0, 0.0, 0.0, 1.0), linewidth=25.0)
    setobject!(vis[:traj], MeshCat.Line(points, line_mat))
    
    setvisible!(vis["/Background"], true)
    setprop!(vis["/Background"], "top_color", RGBA(1.0, 1.0, 1.0, 1.0))
    setprop!(vis["/Background"], "bottom_color", RGBA(1.0, 1.0, 1.0, 1.0))
    setvisible!(vis["/Axes"], false)
    setvisible!(vis["/Grid"], false)
    MeshCat.settransform!(vis["/Cameras/default"],
            MeshCat.compose(MeshCat.Translation(0.0, -25.0, -1.0), MeshCat.LinearMap(Rotations.RotZ(-pi / 2.0))))
            setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 15)

    if mesh 
        obj_path = joinpath("/home/taylor/Desktop/motion_planning/models/cybertruck/cybertruck.obj")
        mtl_path = joinpath("/home/taylor/Desktop/motion_planning/models/cybertruck/cybertruck.mtl")
        
        m = ModifiedMeshFileObject(obj_path, mtl_path, scale=0.04);
        setobject!(vis["car"], m)
        settransform!(vis["car"],
                compose(Translation([0.0; 0.0; 0.0]),
                LinearMap(RotZ(pi + 0.0 * pi / 2) * RotX(pi / 2.0))))
    else 
        car = GeometryBasics.Sphere(Point(0.0,0.0,0.0), 0.05)
        setobject!(vis["car"], car, MeshPhongMaterial(color=RGBA(0.0, 1.0, 0.0, 1.0)))
    end

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))
    p1 = [0.1; 0.0] 
    p2 = [-0.1; 0.0] 
    p3 = [-0.1; 0.0]

    for t = 1:length(x)
        dir = surface_normal(x[t][1]) 
        ang = rotation_angle(dir)
        mat = rotation_matrix(ang)

        scaling = 0.0
        c1 = transpose(mat) * p1 + [x[t][1]; sin(3.0 * x[t][1])] + scaling * dir
        c2 = transpose(mat) * p2 + [x[t][1]; sin(3.0 * x[t][1])] + scaling * dir
        c3 = transpose(mat) * p3 + [x[t][1]; sin(3.0 * x[t][1])] + scaling * dir

        s1 = sin(3.0 * c1[1])
        s2 = sin(3.0 * c2[1])
        s3 = sin(3.0 * c3[1])

        # # shift down
        # k = 1 
        # while k < 100 && c1[2] > s1 && c2[2] > s2 && c3[2] > s3
        #     scaling -= 0.001
        #     k += 1

        #     c1 = transpose(mat) * p1 + [x[t][1]; sin(3.0 * x[t][1])] + scaling * dir
        #     c2 = transpose(mat) * p2 + [x[t][1]; sin(3.0 * x[t][1])] + scaling * dir
        #     c3 = transpose(mat) * p3 + [x[t][1]; sin(3.0 * x[t][1])] + scaling * dir

        #     s1 = sin(3.0 * c1[1])
        #     s2 = sin(3.0 * c2[1])
        #     s3 = sin(3.0 * c3[1])
        # end

        # shift up 
        k = 1 
        while k < 100 && c1[2] < s1 && c2[2] < s2 #&& c3[2] < s3
            scaling += 0.001
            k += 1

            c1 = transpose(mat) * p1 + [x[t][1]; sin(3.0 * x[t][1])] + scaling * dir
            c2 = transpose(mat) * p2 + [x[t][1]; sin(3.0 * x[t][1])] + scaling * dir
            #c3 = transpose(mat) * p3 + [x[t][1]; sin(3.0 * x[t][1])] + scaling * dir

            s1 = sin(3.0 * c1[1])
            s2 = sin(3.0 * c2[1])
           # s3 = sin(3.0 * c3[1])
        end
        @show scaling
        shift = scaling * dir
        MeshCat.atframe(anim,t) do
            settransform!(vis["car"],
                compose(Translation(x[t][1] + shift[1], 0.0, sin(3.0 * x[t][1]) + shift[2]),
                    LinearMap(RotY(ang) * RotZ(pi + 0.0 * pi / 2) * RotX(pi / 2.0))))
        end
    end

    MeshCat.setanimation!(vis, anim)
end

function ModifiedMeshFileObject(obj_path::String, material_path::String;
    scale::T = 0.1) where {T}
    obj = MeshFileObject(obj_path)
    rescaled_contents = rescale_contents(obj_path, scale = scale)
    material = select_material(material_path)
    mod_obj = MeshFileObject(
        rescaled_contents,
        obj.format,
        material,
        obj.resources,
        )
    return mod_obj
end

function rescale_contents(obj_path::String; scale::T = 0.1) where T
    lines = readlines(obj_path)
    rescaled_lines = copy(lines)
    for (k,line) in enumerate(lines)
        if length(line) >= 2
            if line[1] == 'v'
                stringvec = split(line, " ")
                vals = map(x -> parse(Float64, x), stringvec[2:end])
                rescaled_vals = vals .* scale
                rescaled_lines[k] = join([stringvec[1]; string.(rescaled_vals)], " ")
            end
        end
    end
    rescaled_contents = join(rescaled_lines, "\r\n")
    return rescaled_contents
end

function select_material(material_path::String)
    mtl_file = open(material_path)
    mtl = read(mtl_file, String)
    return mtl
end

function rotation_angle(ns)
	# world-frame normal
	nw = [0.0, 1.0]

	angle = atan(nw[2], nw[1]) - atan(ns[2], ns[1])

    return angle
	# # Rw->s
	# [cos(angle) -sin(angle); sin(angle) cos(angle)]
end

function surface_normal(x) 
    dir = [-3.0 * cos(3.0 * x); 1.0]
    return dir ./ norm(dir) 
end

function rotation_matrix(angle)
	# Rw->s
	[cos(angle) -sin(angle); sin(angle) cos(angle)]
end
