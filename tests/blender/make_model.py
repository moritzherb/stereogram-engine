"""Runs inside Blender: export a test .glb. Args after --: out scale x y z [clip_name]."""
import math
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
out, scale = args[0], float(args[1])
loc = tuple(float(v) for v in args[2:5])
clip = args[5] if len(args) > 5 else None
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
cube = bpy.context.active_object
cube.scale = (scale, scale * 0.6, scale)
if clip:
    for frame, angle in ((1, 0.0), (12, math.radians(90)), (24, 0.0)):
        cube.rotation_euler = (0.0, 0.0, angle)
        cube.keyframe_insert("rotation_euler", frame=frame)
    cube.animation_data.action.name = clip
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_animations=bool(clip))
