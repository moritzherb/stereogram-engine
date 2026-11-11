"""Runs INSIDE Blender: builds a scene from a job file and renders depth frames.

Usage: blender -b --factory-startup --python render_depth.py -- job.json

Depth frames are written as 16-bit grayscale PNGs (depth_0001.png, ...),
1.0 = nearest, 0.0 = background. meta.json is written last, so its presence
means the render completed.
"""
import json
import math
import os
import sys
import time

import bpy


def read_job():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("usage: blender -b --python render_depth.py -- job.json")
    with open(argv[argv.index("--") + 1]) as f:
        return json.load(f)


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.curves, bpy.data.actions):
        for block in list(coll):
            coll.remove(block)


def setup_render(scene, width, height, samples=8):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_layers[0].use_pass_mist = True
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")


def setup_mist(scene, start, depth):
    mist = scene.world.mist_settings
    mist.start = start
    mist.depth = depth
    mist.falloff = "LINEAR"


def setup_depth_output(scene, out_dir):
    """Compositor: depth = (1 - mist) * alpha  ->  16-bit BW PNG file output."""
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    inv = tree.nodes.new("CompositorNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    tree.links.new(rl.outputs["Mist"], inv.inputs[1])
    mul = tree.nodes.new("CompositorNodeMath")
    mul.operation = "MULTIPLY"
    mul.use_clamp = True
    tree.links.new(inv.outputs[0], mul.inputs[0])
    tree.links.new(rl.outputs["Alpha"], mul.inputs[1])
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.base_path = out_dir
    out.file_slots[0].path = "depth_"
    out.format.file_format = "PNG"
    out.format.color_mode = "BW"
    out.format.color_depth = "16"
    out.format.compression = 15
    tree.links.new(mul.outputs[0], out.inputs[0])


def add_ortho_camera(scene, location, ortho_scale):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = location
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)  # looks along +Y
    scene.camera = cam
    return cam


def render_frames(scene, first, last):
    """Render each frame (compositor writes the files); returns seconds per frame."""
    times = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=False)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def write_meta(out_dir, meta):
    meta["blender"] = bpy.app.version_string
    tmp = os.path.join(out_dir, "meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, os.path.join(out_dir, "meta.json"))


def run_probe(job):
    """Milestone 1 check: spinning cube, 3 frames at 108x192."""
    out_dir = job["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    clear_scene()
    setup_render(scene, 108, 192)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    cube = bpy.context.active_object
    scene.frame_start, scene.frame_end = 1, 3
    for frame, angle in ((1, 0.0), (3, math.radians(60))):
        cube.rotation_euler = (0.0, 0.0, angle)
        cube.keyframe_insert("rotation_euler", frame=frame)
    add_ortho_camera(scene, (0.0, -5.0, 0.0), ortho_scale=2.5)
    setup_mist(scene, start=4.0, depth=2.0)
    setup_depth_output(scene, out_dir)
    spf = render_frames(scene, 1, 3)
    write_meta(out_dir, {"frames": 3, "seconds_per_frame": spf, "engine": "CYCLES"})


def main():
    job = read_job()
    if job.get("probe"):
        run_probe(job)
    else:
        raise SystemExit("full job mode not implemented yet")


if __name__ == "__main__":
    main()
