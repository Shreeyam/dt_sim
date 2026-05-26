"""
Soft cartoon/volumetric cloud icon for diagrams.
- Transparent PNG, 256x256
- Orthographic camera
- AgX color management, Very High Contrast look
- Soft drop shadow via Cycles shadow catcher
- Slightly cool-grey cloud, MacOS-icon vibe

Usage in Blender:
    1. Open Blender (4.x recommended)
    2. Open the Scripting workspace (or any Text Editor)
    3. Open this file, then click "Run Script" (Alt+P)
    4. Output is written next to this .py file as `cloud_icon.png`
       (or to /tmp/cloud_icon.png if the file hasn't been saved yet)
"""

import bpy
import math
import os

# ---------- Reset scene ----------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for coll in [bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.objects]:
    for item in list(coll):
        if item.users == 0:
            try:
                coll.remove(item)
            except RuntimeError:
                pass

scene = bpy.context.scene

# ---------- Render / output ----------
scene.render.engine = 'CYCLES'
scene.cycles.samples = 256
scene.cycles.use_denoising = True
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.image_settings.color_depth = '16'

# ---------- Color management: AgX, Very High Contrast ----------
scene.view_settings.view_transform = 'AgX'
preferred_looks = [
    'AgX - Very High Contrast',
    'Very High Contrast',
    'AgX - High Contrast',
    'High Contrast',
]
for look_name in preferred_looks:
    try:
        scene.view_settings.look = look_name
        print(f"Look set to: {look_name}")
        break
    except (TypeError, ValueError):
        continue

# ---------- Cloud material ----------
mat = bpy.data.materials.new("CloudMat")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)

out_node = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
out_node.location = (300, 0)
bsdf.location = (0, 0)

# Slightly cool grey/white
bsdf.inputs["Base Color"].default_value = (0.86, 0.87, 0.9, 1.0)
bsdf.inputs["Roughness"].default_value = 1.0

# Subsurface for that soft, slightly translucent fluffy look (handles 4.x and 3.x naming)
def set_input(node, names, value):
    for n in names:
        if n in node.inputs:
            try:
                node.inputs[n].default_value = value
                return True
            except Exception:
                pass
    return False

set_input(bsdf, ["Subsurface Weight", "Subsurface"], 0.35)
set_input(bsdf, ["Subsurface Radius"], (0.5, 0.5, 0.55))
set_input(bsdf, ["Subsurface Color"], (0.95, 0.95, 1.0, 1.0))

nt.links.new(bsdf.outputs[0], out_node.inputs[0])

# ---------- Build cloud (cluster of spheres, voxel-remeshed into one blob) ----------
puffs = [
    # (x, y, z, radius)
    (-1.45, 0.0,  0.05, 0.80),
    (-0.55, 0.05, 0.30, 1.00),
    ( 0.40, 0.0,  0.40, 1.00),
    ( 1.30, 0.0,  0.10, 0.85),
    (-1.00, 0.05,-0.10, 0.70),
    ( 0.90, 0.05,-0.05, 0.72),
    ( 0.00,-0.05,-0.15, 0.90),
    (-0.20, 0.10, 0.55, 0.55),
    ( 0.55, 0.10, 0.60, 0.50),
]

sphere_objs = []
for i, (x, y, z, r) in enumerate(puffs):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=r, location=(x, y, z), segments=48, ring_count=24)
    obj = bpy.context.active_object
    obj.name = f"Puff_{i}"
    bpy.ops.object.shade_smooth()
    sphere_objs.append(obj)

# Merge into a single mesh
bpy.ops.object.select_all(action='DESELECT')
for obj in sphere_objs:
    obj.select_set(True)
bpy.context.view_layer.objects.active = sphere_objs[0]
bpy.ops.object.join()
cloud = bpy.context.active_object
cloud.name = "Cloud"

# Voxel remesh fuses overlapping spheres into one organic blob
remesh = cloud.modifiers.new("Remesh", type='REMESH')
remesh.mode = 'VOXEL'
remesh.voxel_size = 0.05
remesh.use_smooth_shade = True
bpy.ops.object.modifier_apply(modifier=remesh.name)

# Soften the surface
smooth = cloud.modifiers.new("Smooth", type='SMOOTH')
smooth.factor = 0.5
smooth.iterations = 12
bpy.ops.object.modifier_apply(modifier=smooth.name)

# Subsurf for nice silhouette
subsurf = cloud.modifiers.new("Subsurf", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 2

# Apply material
cloud.data.materials.clear()
cloud.data.materials.append(mat)

# ---------- Shadow catcher plane ----------
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, -1.55))
plane = bpy.context.active_object
plane.name = "ShadowCatcher"
# Cycles shadow catcher (Blender 3.x and 4.x)
try:
    plane.is_shadow_catcher = True       # Blender 3.0+
except AttributeError:
    pass
try:
    plane.cycles.is_shadow_catcher = True # older fallback
except Exception:
    pass

# ---------- Lighting ----------
# Sun for the directional drop shadow
bpy.ops.object.light_add(type='SUN', location=(3, -2, 8))
sun = bpy.context.active_object
sun.data.energy = 3.5
sun.data.angle = math.radians(10)   # soft penumbra
sun.rotation_euler = (math.radians(45), math.radians(10), math.radians(30))

# Soft area fill from camera direction
bpy.ops.object.light_add(type='AREA', location=(0, -6, 2))
fill = bpy.context.active_object
fill.data.energy = 120
fill.data.size = 8
fill.rotation_euler = (math.radians(80), 0, 0)

# Slight warm rim from above-back to give shape
bpy.ops.object.light_add(type='AREA', location=(-1, 4, 4))
rim = bpy.context.active_object
rim.data.energy = 60
rim.data.size = 5
rim.rotation_euler = (math.radians(-110), math.radians(0), math.radians(0))
rim.data.color = (1.0, 0.98, 0.95)

# ---------- Camera (orthographic, looking along -Y) ----------
bpy.ops.object.camera_add(location=(0, -10, 0.1))
cam = bpy.context.active_object
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 4.6
cam.rotation_euler = (math.radians(90), 0, 0)
scene.camera = cam

# ---------- World: dark/transparent so it doesn't tint the cloud ----------
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs[1].default_value = 0.0

# ---------- Output path & render ----------
if bpy.data.filepath:
    out_dir = os.path.dirname(bpy.data.filepath)
else:
    # script may be run unsaved; fall back to /tmp on macOS/Linux
    out_dir = os.path.expanduser("~/Desktop")
    if not os.path.isdir(out_dir):
        out_dir = "/tmp"

scene.render.filepath = os.path.join(out_dir, "cloud_icon.png")
print(f"Rendering to: {scene.render.filepath}")

bpy.ops.render.render(write_still=True)
print("Done.")
