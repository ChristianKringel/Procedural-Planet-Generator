import trimesh

mesh = trimesh.load('client/public/models/12219_boat_v2_L2.obj')
simplified = mesh.simplify_quadric_decimation(face_count=50)  # 50 faces apenas
simplified.export('client/public/models/boat_simplified.obj')