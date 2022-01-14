# visuals
function build_satellite(vis, p::Satellite; dim=[1.0, 1.0, 1.0], name="satellite", transparency=1.0)
	orange = RGBA(255/255,127/255,0/255,transparency)
	cyan = RGBA(0/255,255/255,255/255,transparency)
	magenta = RGBA(255/255,0/255,255/255,transparency)

	x_dim = 0.25 * dim[1] * 2.0
	y_dim = 0.25 * dim[2] * 2.0
	z_dim = 0.25 * dim[3] * 2.0

    setobject!(vis[name],
    	Rect(Vec(-x_dim, -y_dim, -z_dim),Vec(2.0 * x_dim, 2.0 * y_dim, 2.0 * z_dim)),
    	MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, transparency)))

    arrow_x = ArrowVisualizer(vis[name][:arrow_x])
    mat = MeshPhongMaterial(color=orange)
    setobject!(arrow_x, mat)
    settransform!(arrow_x,
    	Point(0.0, 0.0, 0.0),
    	Vec(0.75, 0.0, 0.0),
    	shaft_radius=0.05,
    	max_head_radius=0.1)

    arrow_y = ArrowVisualizer(vis[name][:arrow_y])
    mat = MeshPhongMaterial(color=cyan)
    setobject!(arrow_y, mat)
    settransform!(arrow_y,
    	Point(0.0, 0.0, 0.0),
    	Vec(0.0, 0.75, 0.0),
    	shaft_radius=0.05,
    	max_head_radius=0.1)

    arrow_z = ArrowVisualizer(vis[name][:arrow_z])
    mat = MeshPhongMaterial(color=magenta)
    setobject!(arrow_z, mat)
    settransform!(arrow_z,
    	Point(0.0, 0.0, 0.0),
    	Vec(0.0, 0.0, 0.75),
    	shaft_radius=0.05,
    	max_head_radius=0.1)

	return vis
end


# visuals
function set_satellite!(vis, p::Satellite, q; name="satellite")
	settransform!(vis[name],
		compose(Translation(0.0 * [-0.25; -0.25; -0.25]...),
				LinearMap(rotation_matrix(q))))
end

# visuals
function visualize_satellite!(vis, p::Satellite, q; Δt=0.1, dim=[1.0, 1.0, 1.0], name="satellite", transparency=1.0)
	setvisible!(vis["/Background"], true)
	setprop!(vis["/Background"], "top_color", RGBA(1.0, 1.0, 1.0, 1.0))
	setprop!(vis["/Background"], "bottom_color", RGBA(1.0, 1.0, 1.0, 1.0))
	setvisible!(vis["/Axes"], false)
	setvisible!(vis["/Grid"], false)

	build_satellite(vis, p, dim=dim, name=name, transparency=transparency)
    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))
	for t = 1:length(q)
		MeshCat.atframe(anim, t) do
			set_satellite!(vis, p, q[t], name=name)
		end
	end
    MeshCat.setanimation!(vis, anim)
end

function ghost(vis, p::Satellite, q; dim=[1.0, 1.0, 1.0], timestep=[t for t = 1:length(q)], transparency=[1.0 for t = 1:length(q)])
	for (i, t) in enumerate(timestep)
		name = "satellite_$t"
		build_satellite(vis, p, dim=dim, name=name, transparency=transparency[i])
		set_satellite!(vis, p, q[t], name=name)
	end
end