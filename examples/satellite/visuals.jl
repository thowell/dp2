# visuals
function build_satellite(vis, p::Satellite; dim=p.dim, name="satellite", transparency=1.0, body_scale=1.0, arrow_scale=1.0)
	orange = RGBA(255/255,127/255,0/255,transparency)
	cyan = RGBA(0/255,255/255,255/255,transparency)
	magenta = RGBA(255/255,0/255,255/255,transparency)

	x_dim = 0.5 * dim[1] * body_scale 
	y_dim = 0.5 * dim[2] * body_scale
	z_dim = 0.5 * dim[3] * body_scale
 
    setobject!(vis[name],
    	Rect(Vec(-x_dim, -y_dim, -z_dim),Vec(2.0 * x_dim, 2.0 * y_dim, 2.0 * z_dim)),
    	MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, transparency)))

    arrow_x = ArrowVisualizer(vis[name][:arrow_x])
    mat = MeshPhongMaterial(color=orange)
    setobject!(arrow_x, mat)
    settransform!(arrow_x,
    	Point(0.0, 0.0, 0.0),
    	Vec(0.75, 0.0, 0.0),
    	shaft_radius=0.05 * arrow_scale,
    	max_head_radius=0.1 * arrow_scale)

    arrow_y = ArrowVisualizer(vis[name][:arrow_y])
    mat = MeshPhongMaterial(color=cyan)
    setobject!(arrow_y, mat)
    settransform!(arrow_y,
    	Point(0.0, 0.0, 0.0),
    	Vec(0.0, 0.75, 0.0),
    	shaft_radius=0.05 * arrow_scale,
    	max_head_radius=0.1 * arrow_scale)

    arrow_z = ArrowVisualizer(vis[name][:arrow_z])
    mat = MeshPhongMaterial(color=magenta)
    setobject!(arrow_z, mat)
    settransform!(arrow_z,
    	Point(0.0, 0.0, 0.0),
    	Vec(0.0, 0.0, 0.75),
    	shaft_radius=0.05 * arrow_scale,
    	max_head_radius=0.1 * arrow_scale)

	return vis
end

# visuals
function set_satellite!(vis, p::Satellite, q; name="satellite", orientation=:quaternion)
	settransform!(vis[name],
		compose(Translation(0.0 * [-0.25; -0.25; -0.25]...),
				LinearMap(orientation == :quaternion ? rotation_matrix(q[1:4]) : Rotations.MRP(q[1:3]...))))
end

# visuals
function visualize_satellite!(vis, p::Satellite, q; Δt=0.1, dim=p.dim, name="satellite", transparency=1.0, body_scale=1.0, arrow_scale=1.0, orientation=:quaternion)
	setvisible!(vis["/Background"], true)
	setprop!(vis["/Background"], "top_color", RGBA(1.0, 1.0, 1.0, 1.0))
	setprop!(vis["/Background"], "bottom_color", RGBA(1.0, 1.0, 1.0, 1.0))
	setvisible!(vis["/Axes"], false)
	setvisible!(vis["/Grid"], false)

	build_satellite(vis, p, dim=dim, name=name, transparency=transparency, body_scale=body_scale, arrow_scale=arrow_scale)
    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))
	for t = 1:length(q)
		MeshCat.atframe(anim, t) do
			set_satellite!(vis, p, q[t], name=name, orientation=orientation)
		end
	end
    MeshCat.setanimation!(vis, anim)
end

function ghost(vis, p::Satellite, q; dim=[1.0, 1.0, 1.0], timestep=[t for t = 1:length(q)], transparency=[1.0 for t = 1:length(q)], body_scale=1.0, arrow_scale=1.0)
	for (i, t) in enumerate(timestep)
		name = "satellite_$t"
		build_satellite(vis, p, dim=dim, name=name, transparency=transparency[i], body_scale=body_scale, arrow_scale=arrow_scale)
		set_satellite!(vis, p, q[t], name=name)
	end
end