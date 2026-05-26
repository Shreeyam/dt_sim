"""Render a stylised Earth panel background in Blender.

UV-mapped Earth sphere using a high-resolution albedo texture, even (ambient)
lighting, grazing camera angle, desaturated colour, and Freestyle line strokes.
The output is intended to sit behind a paper1 panel.

Run headless:

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python paper1/scripts/render_earth_panel_bg.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import bpy

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_PATH = Path(bpy.path.abspath(__file__) if "__file__" in globals() else
                   sys.argv[0]).resolve()
# Allow running both with and without Blender expanding __file__.
for candidate in [SCRIPT_PATH, Path(__file__).resolve() if "__file__" in globals()
                  else SCRIPT_PATH]:
    if candidate.exists():
        SCRIPT_PATH = candidate
        break

REPO_ROOT = Path("/Users/shreeyam/Projects/phd_extensions")
OUT_DIR = REPO_ROOT / "paper1" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "earth_panel_bg.png"

ALBEDO_PATH = Path("/Users/shreeyam/Library/CloudStorage/OneDrive-Personal/"
                   "MIT/PhD/Media/Earth Blender/59-earth/textures/earth albedo.jpg")
if not ALBEDO_PATH.exists():
    raise FileNotFoundError(f"Earth albedo not found at {ALBEDO_PATH}")

# --------------------------------------------------------------------------- #
# Render settings
# --------------------------------------------------------------------------- #
RES_X, RES_Y = 1920, 1200          # panel-friendly aspect
SAMPLES = 64                        # Cycles samples
TEXTURE_ALPHA = 0.55                # texture weight when mixed over white;
                                    # stronger than the footprint plot's 0.15
                                    # because we are zoomed in and would
                                    # otherwise lose terrain detail
EARTH_RADIUS = 1.0
# "Grazing" satellite POV: camera just above the surface looking forward.
CAMERA_ALTITUDE = 0.08              # height above surface in Earth radii (≈ 510 km)
CAMERA_PITCH_DEG = 22.0             # angle below local horizontal toward ground
CAMERA_LENS_MM = 50
EARTH_Z_ROT_DEG = -90.0             # rotates a chosen region under the camera
WORLD_STRENGTH = 1.0                # ambient illumination — keeps lighting even
FREESTYLE_THICKNESS = 4.5
LINE_RGB = (0.05, 0.05, 0.07)


# --------------------------------------------------------------------------- #
# Scene reset
# --------------------------------------------------------------------------- #
def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                  bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            block.remove(item)


def make_earth() -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=EARTH_RADIUS, segments=128,
                                         ring_count=64, location=(0, 0, 0))
    earth = bpy.context.active_object
    earth.name = "Earth"
    bpy.ops.object.shade_smooth()

    # UV sphere already has a spherical UV map; equirectangular textures map directly.
    mat = bpy.data.materials.new("EarthMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    mix = nt.nodes.new("ShaderNodeMixRGB")  # white ⊕ texture as the base colour
    tex = nt.nodes.new("ShaderNodeTexImage")
    coords = nt.nodes.new("ShaderNodeTexCoord")

    img = bpy.data.images.load(str(ALBEDO_PATH))
    tex.image = img
    tex.interpolation = "Cubic"

    # Mix: result = (1 - Fac) * white + Fac * texture — washes the albedo
    # toward white, keeping shading from the principled BSDF on top.
    mix.blend_type = "MIX"
    mix.inputs["Fac"].default_value = TEXTURE_ALPHA
    mix.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)

    bsdf.inputs["Roughness"].default_value = 0.95
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.05
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.05

    nt.links.new(coords.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], mix.inputs["Color2"])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    earth.data.materials.append(mat)
    earth.rotation_euler = (0.0, 0.0, math.radians(EARTH_Z_ROT_DEG))
    return earth


def setup_world_even_lighting() -> None:
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = WORLD_STRENGTH
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    # Pure ambient — no sun lamp — keeps the planet evenly lit, no terminator.


def setup_camera() -> bpy.types.Object:
    """Low-orbit grazing POV: camera sits just above the surface and looks
    forward along the local tangent, pitched slightly down so the foreground
    surface and the curved horizon are both in frame.

    Local frame at the camera's surface point:
        outward (radial)   = (0, -1, 0)
        forward (tangent)  = (+1, 0, 0)   (heading "east")
        down (to centre)   = (0, +1, 0)
        sky / world up     = (0, 0, +1)
    """
    from mathutils import Matrix, Vector

    d = EARTH_RADIUS + CAMERA_ALTITUDE  # camera distance from sphere centre
    cam_loc = Vector((0.0, -d, 0.0))    # above surface point at lat=0, lon=-90°

    pitch = math.radians(CAMERA_PITCH_DEG)
    # look_dir points from the camera toward what it sees (forward + down).
    look_dir = Vector((math.cos(pitch), math.sin(pitch), 0.0)).normalized()
    # Image up is the local outward radial (sky direction at the camera) — this
    # is the satellite-window view: surface below, horizon up top.
    outward = cam_loc.normalized()

    # Camera basis in world coords (Blender camera looks down its local -Z):
    #   +Z_cam (back of cam)  = -look_dir
    #   +Y_cam (image up)     = outward, projected perpendicular to +Z_cam
    #   +X_cam (image right)  = +Y_cam × +Z_cam
    z_cam = -look_dir
    y_cam = (outward - outward.dot(z_cam) * z_cam).normalized()
    x_cam = y_cam.cross(z_cam).normalized()

    rot = Matrix((
        (x_cam.x, y_cam.x, z_cam.x, 0.0),
        (x_cam.y, y_cam.y, z_cam.y, 0.0),
        (x_cam.z, y_cam.z, z_cam.z, 0.0),
        (0.0,     0.0,     0.0,     1.0),
    ))

    bpy.ops.object.camera_add(location=cam_loc, rotation=rot.to_euler())
    cam = bpy.context.active_object
    cam.data.lens = CAMERA_LENS_MM

    bpy.context.scene.camera = cam
    return cam


def setup_render() -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    try:
        scene.cycles.device = "GPU"
    except Exception:
        scene.cycles.device = "CPU"

    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(OUT_PATH)

    # Color management: Filmic looks great but desaturation already happens
    # in-shader, so use Standard view transform to keep the look predictable.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"


def setup_freestyle() -> None:
    scene = bpy.context.scene
    scene.render.use_freestyle = True
    view_layer = scene.view_layers[0]
    view_layer.use_freestyle = True

    # Use existing "default" lineset; add silhouette / crease only.
    fs_settings = view_layer.freestyle_settings
    # Wipe any existing linesets so the script is idempotent.
    while fs_settings.linesets:
        fs_settings.linesets.remove(fs_settings.linesets[0])
    lineset = fs_settings.linesets.new("OutlineSet")
    lineset.select_silhouette = True
    lineset.select_border = True
    lineset.select_crease = False
    lineset.select_contour = True
    lineset.select_external_contour = True

    linestyle = lineset.linestyle
    linestyle.color = LINE_RGB
    linestyle.thickness = FREESTYLE_THICKNESS
    linestyle.thickness_position = "INSIDE"
    linestyle.alpha = 1.0
    linestyle.use_chaining = True


def main() -> None:
    clear_scene()
    make_earth()
    setup_world_even_lighting()
    setup_camera()
    setup_render()
    setup_freestyle()

    bpy.ops.render.render(write_still=True)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
