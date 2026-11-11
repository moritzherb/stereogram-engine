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
import mathutils


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
    """Milestone 1 check: spinning cube, 3 frames (default 108x192)."""
    out_dir = job["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    clear_scene()
    setup_render(scene, job.get("width", 108), job.get("height", 192), samples=job.get("samples", 8))
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


# ---------------------------------------------------------------- full job mode

MOTIONS = ("spin", "float", "bob", "zoom_in", "fly_toward", "swim_past", "ripple", "pulse")
DRAWABLE = {"MESH", "CURVE", "FONT", "SURFACE", "META"}


def fail(code, message):
    sys.stderr.write(message + "\n")
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(code)


def clay_material():
    mat = bpy.data.materials.new("Clay")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.78, 0.74, 0.70, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def _curve_object(name, points, extrude, bevel):
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "2D"
    curve.fill_mode = "BOTH"
    curve.extrude = extrude
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for p, (x, y) in zip(spline.points, points):
        p.co = (x, y, 0.0, 1.0)
    spline.use_cyclic_u = True
    tmp = bpy.data.objects.new(name + "_curve", curve)
    bpy.context.scene.collection.objects.link(tmp)
    # Convert to a mesh and round the edges with a clamped bevel (a curve
    # bevel shoots spikes out of sharp corners such as the heart's tips).
    mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(tmp, do_unlink=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("Bevel", "BEVEL")
    mod.width = bevel
    mod.segments = 4
    mod.limit_method = "ANGLE"
    mod.use_clamp_overlap = True
    obj.rotation_euler = (math.pi / 2, 0.0, 0.0)  # face the camera (-Y)
    return obj


def build_shape(name):
    ops = bpy.ops.mesh
    if name == "sphere":
        ops.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=32)
    elif name == "cube":
        ops.primitive_cube_add(size=1.6)
        bev = bpy.context.active_object.modifiers.new("Bevel", "BEVEL")
        bev.width, bev.segments = 0.15, 4
    elif name == "torus":
        ops.primitive_torus_add(major_radius=1.0, minor_radius=0.35, major_segments=64, minor_segments=24)
        bpy.context.active_object.rotation_euler = (math.pi / 2, 0.0, 0.0)
    elif name == "cone":
        ops.primitive_cone_add(vertices=64, radius1=1.0, depth=1.6)
        bpy.context.active_object.rotation_euler = (math.pi / 2, 0.0, 0.0)  # tip towards the viewer
    elif name == "heart":
        pts = []
        for k in range(96):
            t = 2 * math.pi * k / 96
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append((x / 16.0, y / 16.0))
        return [_curve_object("Heart", pts, 0.2, 0.12)]
    elif name == "star":
        pts = []
        for k in range(10):
            r = 1.0 if k % 2 == 0 else 0.45
            a = math.pi / 2 + k * math.pi / 5
            pts.append((r * math.cos(a), r * math.sin(a)))
        return [_curve_object("Star", pts, 0.2, 0.08)]
    else:
        fail(4, f"IMPORT_FAILED: unknown shape {name}")
    return [bpy.context.active_object]


def build_text(text):
    curve = bpy.data.curves.new("Text", type="FONT")
    curve.body = text
    curve.extrude = 0.12
    curve.bevel_depth = 0.02
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    obj = bpy.data.objects.new("Text", curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.rotation_euler = (math.pi / 2, 0.0, 0.0)
    return [obj]


def import_model(path):
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".glb", ".gltf"):
            result = bpy.ops.import_scene.gltf(filepath=path)
        elif ext == ".fbx":
            result = bpy.ops.import_scene.fbx(filepath=path)
        else:
            fail(4, f"IMPORT_FAILED: unsupported file type {ext}")
    except Exception as e:  # noqa: BLE001 - any importer failure
        fail(4, f"IMPORT_FAILED: {e}".replace("\n", " ")[:500])
    if "FINISHED" not in result:
        fail(4, "IMPORT_FAILED: the importer did not finish")
    new = [o for o in bpy.data.objects if o not in before]
    for o in list(new):
        if o.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(o, do_unlink=True)
            new.remove(o)
    if not any(o.type in DRAWABLE for o in new):
        fail(4, "IMPORT_FAILED: the file contains no visible objects")
    return new


def world_bbox(objects):
    dg = bpy.context.evaluated_depsgraph_get()
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for o in objects:
        if o.type not in DRAWABLE:
            continue
        ev = o.evaluated_get(dg)
        for corner in ev.bound_box:
            w = ev.matrix_world @ mathutils.Vector(corner)
            for k in range(3):
                lo[k] = min(lo[k], w[k])
                hi[k] = max(hi[k], w[k])
    return lo, hi


def find_clip(name, clips):
    if name is None:
        return None
    for act in bpy.data.actions:
        if act.name == name:
            return act
    for act in bpy.data.actions:
        if act.name.lower() == name.lower():
            return act
    fail(3, f"UNKNOWN_CLIP: {name}; available: {', '.join(clips) or '(none)'}")


def assign_clip(action, objects):
    for o in objects:
        if o.animation_data is None:
            continue
        o.animation_data.action = action
        if hasattr(o.animation_data, "action_slot") and o.animation_data.action_slot is None:
            slots = getattr(action, "slots", [])
            if len(slots):
                o.animation_data.action_slot = slots[0]


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


class Rig:
    """MOTION (animated, at the origin) > NORM (normalises size/position) > objects."""

    def __init__(self, scene, objects, job):
        self.scene = scene
        self.objects = objects
        self.fps = job["fps"]
        self.speed = job["speed"]
        self.duration = job["duration"]
        self.motion = job.get("motion")
        self.frames = max(1, round(self.duration * self.fps))
        self.travel = 0.0
        self.motion_obj = bpy.data.objects.new("MOTION", None)
        self.norm = bpy.data.objects.new("NORM", None)
        scene.collection.objects.link(self.motion_obj)
        scene.collection.objects.link(self.norm)
        self.norm.parent = self.motion_obj
        for o in objects:
            if o.parent is None:
                o.parent = self.norm
                o.matrix_parent_inverse = self.norm.matrix_world.inverted()
        self.clip = None
        self.clip_range = (1.0, 1.0)
        self.waves = []

    def set_clip(self, action):
        self.clip = action
        if action is not None:
            assign_clip(action, self.objects)
            start, end = action.frame_range
            self.clip_range = (start, max(end, start + 1))

    def normalise(self):
        self.apply(0, with_motion=False)
        lo, hi = world_bbox(self.objects)
        centre = [(a + b) / 2 for a, b in zip(lo, hi)]
        diag = math.sqrt(sum((b - a) ** 2 for a, b in zip(lo, hi))) or 1.0
        s = 2.0 / diag
        self.norm.scale = (s, s, s)
        self.norm.location = (-centre[0] * s, -centre[1] * s, -centre[2] * s)
        self.local_scale = s
        if self.motion == "ripple":
            for o in self.objects:
                if o.type in ("MESH", "CURVE", "FONT"):
                    wave = o.modifiers.new("Ripple", "WAVE")
                    wave.height = 0.05 / s
                    wave.width = 0.5 / s
                    wave.speed = 0.02 / s
                    self.waves.append(wave)

    def apply(self, i, with_motion=True):
        t = i / self.fps
        frame, sub = 1, 0.0
        if self.clip is not None:
            start, end = self.clip_range
            ct = start + (t * self.fps * self.speed) % (end - start)
            frame, sub = int(ct), ct - int(ct)
        self.scene.frame_set(frame, subframe=sub)
        m = self.motion_obj
        m.location = (0.0, 0.0, 0.0)
        m.rotation_euler = (0.0, 0.0, 0.0)
        m.scale = (1.0, 1.0, 1.0)
        if with_motion and self.motion:
            self._motion(m, t)
        for wave in self.waves:
            wave.time_offset = frame - t * self.fps * self.speed
        bpy.context.view_layer.update()

    def _motion(self, m, t):
        sp, two_pi = self.speed, 2 * math.pi
        p = min(t * sp / self.duration, 1.0)
        if self.motion == "spin":
            m.rotation_euler[2] = two_pi * t * sp / 6.0
        elif self.motion == "float":
            m.location[2] = 0.08 * math.sin(two_pi * t * sp / 4.0)
            m.rotation_euler[2] = math.radians(15) * math.sin(two_pi * t * sp / 7.0)
        elif self.motion == "bob":
            m.location[2] = 0.15 * math.sin(two_pi * t * sp / 1.2)
        elif self.motion == "zoom_in":
            k = 0.6 + 0.4 * ease(p)
            m.scale = (k, k, k)
        elif self.motion == "fly_toward":
            m.location[1] = 3.0 * (1.0 - ease(p))
            k = 0.5 + 0.5 * ease(p)
            m.scale = (k, k, k)
        elif self.motion == "swim_past":
            q = (t * sp / self.duration) % 1.0 if sp > 1 else p
            m.location[0] = -self.travel + 2 * self.travel * q
            m.location[2] = 0.05 * math.sin(two_pi * t * sp / 2.0)
        elif self.motion == "pulse":
            u = (t * sp) % 1.0
            beat = u / 0.1 if u < 0.1 else math.exp(-(u - 0.1) * 6.0)
            k = 1.0 + 0.08 * beat
            m.scale = (k, k, k)

    def sample_boxes(self, with_motion=True):
        step = max(1, self.frames // 300)
        boxes = []
        for i in range(0, self.frames, step):
            self.apply(i, with_motion)
            boxes.append(world_bbox(self.objects))
        return boxes


def frame_camera(scene, rig, job):
    w, h = job["width"], job["height"]
    safe = job["safe"]
    sw = 1.0 - safe["left"] - safe["right"]
    sh = 1.0 - safe["top"] - safe["bottom"]
    margin = 0.95
    if rig.motion == "swim_past":
        boxes = rig.sample_boxes(with_motion=False)
        bw = max(b[1][0] - b[0][0] for b in boxes)
        bh = max(b[1][2] - b[0][2] for b in boxes)
        cx = sum((b[0][0] + b[1][0]) / 2 for b in boxes) / len(boxes)
        cz = sum((b[0][2] + b[1][2]) / 2 for b in boxes) / len(boxes)
        world_w = max(bw / (0.3 * sw), bh / (sh * margin) * w / h)
        rig.travel = (sw * world_w - bw) / 2
        boxes = rig.sample_boxes(with_motion=True)
    else:
        boxes = rig.sample_boxes(with_motion=True)
    lo = [min(b[0][k] for b in boxes) for k in range(3)]
    hi = [max(b[1][k] for b in boxes) for k in range(3)]
    bw, bd, bh = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    if rig.motion != "swim_past":
        cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    world_h = max(bh / (sh * margin), bw / (sw * margin) * h / w)
    world_w = world_h * w / h
    ortho = max(world_w, world_h)
    cx_frac = (safe["left"] + 1.0 - safe["right"]) / 2
    cy_frac = (safe["top"] + 1.0 - safe["bottom"]) / 2
    cam_x = cx - (cx_frac - 0.5) * world_w
    cam_z = cz + (cy_frac - 0.5) * world_h
    cam_y = lo[1] - 10.0
    add_ortho_camera(scene, (cam_x, cam_y, cam_z), ortho)
    setup_mist(scene, start=lo[1] - cam_y, depth=max(bd, 0.6 * max(bw, bh)))


def render_job_frames(scene, rig, out_dir):
    raw = os.path.join(out_dir, "_raw")
    os.makedirs(raw, exist_ok=True)
    setup_depth_output(scene, raw)
    times = []
    for i in range(rig.frames):
        rig.apply(i)
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=False)
        times.append(time.perf_counter() - t0)
        src = os.path.join(raw, f"depth_{scene.frame_current:04d}.png")
        os.replace(src, os.path.join(out_dir, f"depth_{i + 1:04d}.png"))
        print(f"FRAME {i + 1}/{rig.frames}", flush=True)
    os.rmdir(raw)
    return sum(times) / len(times)


def pick_reveal_frame(rig):
    """The most readable pose: facing the camera, at full size, near the centre.

    Judged from the motion preset (a turned-away or still-small object reads
    badly in the answer image). Near-ties go to the frame closest to the middle.
    """
    step = max(1, rig.frames // 300)
    scores = {}
    m = rig.motion_obj
    for i in range(0, rig.frames, step):
        rig.apply(i)
        score = math.cos(m.rotation_euler[2]) * m.scale[0]
        if rig.travel:
            score -= 0.5 * abs(m.location[0]) / rig.travel
        scores[i] = score
    best = max(scores.values())
    near = [i for i, sc in scores.items() if sc >= best - 0.005 * abs(best)]
    return min(near, key=lambda i: abs(i - rig.frames // 2))


def render_reveal(scene, rig, out_dir, needs_clay, frame):
    rig.apply(frame)
    if needs_clay:
        mat = clay_material()
        for o in rig.objects:
            if o.type in DRAWABLE and hasattr(o.data, "materials"):
                o.data.materials.clear()
                o.data.materials.append(mat)
    scene.use_nodes = False
    scene.render.use_compositing = False
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.cycles.samples = 16
    scene.cycles.max_bounces = 4
    scene.world.use_nodes = False
    scene.world.color = (0.55, 0.56, 0.58)
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), math.radians(-20), math.radians(-30))
    scene.collection.objects.link(sun)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = os.path.join(out_dir, "reveal.png")
    bpy.ops.render.render(write_still=True)


def run_job(job):
    out_dir = job["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    clear_scene()
    scene.render.fps = job["fps"]
    setup_render(scene, job["width"], job["height"], samples=job.get("samples", 4))
    src = job["source"]
    if "model" in src:
        objects = import_model(src["model"])
    elif "text3d" in src:
        objects = build_text(src["text3d"])
    else:
        objects = build_shape(src["shape"])
    clips = sorted(a.name for a in bpy.data.actions)
    rig = Rig(scene, objects, job)
    action = find_clip(job.get("animation"), clips)
    if action is None and "model" in src and len(clips) == 1:
        action = bpy.data.actions[clips[0]]
    rig.set_clip(action)
    rig.normalise()
    frame_camera(scene, rig, job)
    spf = render_job_frames(scene, rig, out_dir)
    reveal_frame = pick_reveal_frame(rig)
    if job.get("reveal"):
        render_reveal(scene, rig, out_dir, needs_clay="model" not in src, frame=reveal_frame)
    write_meta(out_dir, {"frames": rig.frames, "clips": clips, "seconds_per_frame": spf, "engine": "CYCLES",
                         "reveal_frame": reveal_frame})


def main():
    job = read_job()
    if job.get("probe"):
        run_probe(job)
    else:
        run_job(job)


if __name__ == "__main__":
    main()
