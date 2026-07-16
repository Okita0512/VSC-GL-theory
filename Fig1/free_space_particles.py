"""
Blender Python script: Free-Space Particle Ensemble

Run from Blender's Scripting tab or via:
    blender --background --python free_space_particles.py

This script creates a clean scientific-rendering style image of particles
floating freely in space.  There are no cavity mirrors, no cavity beam, and no
central aggregate.  The particles are distributed with jittered grid placement
plus rejection checks, giving a dispersed cloud with suppressed local clumping.
"""

import bpy
import random
import math
try:
    from mathutils import Vector  # type: ignore  # Blender-internal, not on system path
except ImportError:
    pass


# ============================================================================
#  ADJUSTABLE PARAMETERS - edit here without touching the functions below
# ============================================================================

RENDER_QUALITY = 'preview'    # 'preview' (fast) | 'final' (high quality)
MASTER_SEED    = 42
OUTPUT_PATH    = "C:/Users/Wenxiang Ying/Desktop/Fig1/free_space_particles.png"

# Free-space particle cloud.  Bounds define a box-like region centered near the
# origin.  Particles are jittered inside grid cells, then checked for separation.
PARTICLE_COUNT        = 64
X_BOUNDS              = (-5.0, 5.0)
Y_BOUNDS              = (-3.5, 3.5)
Z_BOUNDS              = (-3.0, 3.0)
PARTICLE_RADIUS_RANGE = (0.18, 0.36)
MIN_CENTER_SEPARATION = 0.72     # suppresses visible clumps
SURFACE_GAP_FACTOR    = 1.05     # center distance must exceed this*(r1+r2)
PLACEMENT_JITTER      = 0.62     # 0 = grid centers, 1 = full cell jitter
MAX_PLACEMENT_PASSES  = 5
SPHERE_SUBDIVISIONS   = 2

# Balanced scientific palette.  Weights are deliberately even: no central or
# dominant color group should stand out.
MATERIAL_SEQUENCE = (
    'red',
    'pale_blue',
    'light_gray',
    'light_beige',
)

# Camera - open frontal perspective, far enough to see the full ensemble.
CAM_LOCATION  = (0.0, -17.0, 2.2)
CAM_TARGET    = (0.0,   0.0, 0.0)
CAM_FOCAL_MM  = 48
CAM_FSTOP     = 4.5

# Quality presets
QUALITY = {
    'preview': dict(samples=64,  resolution=900,  denoise=True),
    'final':   dict(samples=300, resolution=1400, denoise=True),
}


# ============================================================================
#  SCENE HELPERS
# ============================================================================

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for db in (bpy.data.meshes, bpy.data.materials,
               bpy.data.lights, bpy.data.cameras):
        for block in list(db):
            db.remove(block)


# ============================================================================
#  MATERIALS
# ============================================================================

def _apply_principled(bsdf, base_color, roughness, metallic, subsurface):
    bsdf.inputs['Base Color'].default_value = (*base_color, 1.0)
    bsdf.inputs['Roughness'].default_value  = roughness
    bsdf.inputs['Metallic'].default_value   = metallic
    for key in ('Specular', 'Specular IOR Level'):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = 0.4 if metallic > 0.5 else 0.10
            break
    for key in ('Subsurface Weight', 'Subsurface'):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = subsurface
            break
    if subsurface > 0 and 'Subsurface Radius' in bsdf.inputs:
        bsdf.inputs['Subsurface Radius'].default_value = (0.10, 0.06, 0.06)


def create_material(name, base_color, roughness=0.35,
                    metallic=0.0, subsurface=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out  = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    out.location, bsdf.location = (300, 0), (0, 0)
    _apply_principled(bsdf, base_color, roughness, metallic, subsurface)
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def build_material_palette():
    return {
        'red': create_material(
            'MutedRed',
            base_color = (0.72, 0.17, 0.14),
            roughness  = 0.42,
            subsurface = 0.03,
        ),
        'pale_blue': create_material(
            'PaleBlue',
            base_color = (0.64, 0.79, 0.94),
            roughness  = 0.42,
            subsurface = 0.04,
        ),
        'light_gray': create_material(
            'LightGray',
            base_color = (0.80, 0.80, 0.83),
            roughness  = 0.46,
            subsurface = 0.03,
        ),
        'light_beige': create_material(
            'LightBeige',
            base_color = (0.93, 0.88, 0.77),
            roughness  = 0.43,
            subsurface = 0.04,
        ),
    }


# ============================================================================
#  SPHERE INSTANCING - one shared mesh per (material, subdivision) pair
# ============================================================================

_sphere_mesh_cache: dict = {}


def _get_sphere_mesh(material, subdivisions: int):
    key = (material.name, subdivisions)
    if key not in _sphere_mesh_cache:
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=subdivisions, radius=1.0, location=(0, 0, 0))
        tmp = bpy.context.active_object
        tmp.data.name = f'SphereMesh_sub{subdivisions}_{material.name}'
        tmp.data.materials.append(material)
        for poly in tmp.data.polygons:
            poly.use_smooth = True
        _sphere_mesh_cache[key] = tmp.data
        bpy.data.objects.remove(tmp, do_unlink=True)
    return _sphere_mesh_cache[key]


def create_sphere(location, radius, material, subdivisions=2):
    """Instance a shared unit icosphere; no unique mesh per particle."""
    mesh = _get_sphere_mesh(material, subdivisions)
    obj  = bpy.data.objects.new('Particle', mesh)
    obj.location = location
    obj.scale    = (radius, radius, radius)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# ============================================================================
#  FREE-SPACE PARTICLE PLACEMENT
# ============================================================================

def _grid_dimensions(count, bounds):
    """Choose grid dimensions roughly matched to the box aspect ratio."""
    spans = [hi - lo for lo, hi in bounds]
    volume = spans[0] * spans[1] * spans[2]
    cell = (volume / count) ** (1.0 / 3.0)
    dims = [max(1, int(round(span / cell))) for span in spans]
    while dims[0] * dims[1] * dims[2] < count:
        axis = max(range(3), key=lambda i: spans[i] / dims[i])
        dims[axis] += 1
    return tuple(dims)


def _candidate_cells(bounds, count, rng):
    dims = _grid_dimensions(count, bounds)
    cells = [(ix, iy, iz)
             for ix in range(dims[0])
             for iy in range(dims[1])
             for iz in range(dims[2])]
    rng.shuffle(cells)
    return cells, dims


def _point_in_cell(cell, dims, bounds, jitter, rng):
    coords = []
    for axis, idx in enumerate(cell):
        lo, hi = bounds[axis]
        span = hi - lo
        step = span / dims[axis]
        center = lo + (idx + 0.5) * step
        offset = rng.uniform(-0.5, 0.5) * step * jitter
        coords.append(center + offset)
    return tuple(coords)


def _is_separated(candidate, radius, placed):
    cx, cy, cz = candidate
    for px, py, pz, pr in placed:
        dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2)
        required = max(MIN_CENTER_SEPARATION, (radius + pr) * SURFACE_GAP_FACTOR)
        if dist < required:
            return False
    return True


def _balanced_material_keys(count, material_keys, rng):
    keys = []
    while len(keys) < count:
        block = list(material_keys)
        rng.shuffle(block)
        keys.extend(block)
    return keys[:count]


def scatter_free_particles(count, mats, bounds, radius_range,
                           subdivisions=2, rng=None):
    """
    Create a dispersed, approximately uniform free-space cloud.

    The first stage samples one jittered point per grid cell, which avoids the
    obvious local clumping of naive random sampling.  Rejection checks then
    remove candidates that would overlap or sit too close to existing spheres.
    """
    if rng is None:
        rng = random

    placed = []
    objects = []
    material_keys = _balanced_material_keys(count, MATERIAL_SEQUENCE, rng)

    jitter = PLACEMENT_JITTER
    for placement_pass in range(MAX_PLACEMENT_PASSES):
        cells, dims = _candidate_cells(bounds, count * 2, rng)
        for cell in cells:
            if len(objects) >= count:
                return objects

            loc = _point_in_cell(cell, dims, bounds, jitter, rng)
            radius = rng.uniform(*radius_range)
            if not _is_separated(loc, radius, placed):
                continue

            key = material_keys[len(objects)]
            objects.append(create_sphere(loc, radius, mats[key], subdivisions))
            placed.append((*loc, radius))

        # Later passes slightly relax the grid jitter so empty areas can fill in.
        if placement_pass == MAX_PLACEMENT_PASSES - 2:
            jitter = min(0.80, jitter + 0.12)

    if len(objects) < count:
        print(f'Warning: placed {len(objects)} of {count} requested particles.')
    return objects


# ============================================================================
#  CAMERA
# ============================================================================

def setup_camera(cam_loc, target, focal_mm, fstop):
    bpy.ops.object.camera_add(location=cam_loc)
    cam_obj = bpy.context.active_object

    direction = Vector(target) - Vector(cam_loc)
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    cam              = cam_obj.data
    cam.lens         = focal_mm
    cam.sensor_width = 36

    focus_dist             = direction.length
    cam.dof.use_dof        = True
    cam.dof.focus_distance = focus_dist
    cam.dof.aperture_fstop = fstop

    bpy.context.scene.camera = cam_obj
    return cam_obj


# ============================================================================
#  LIGHTING
# ============================================================================

def setup_lighting():
    # Studio-like area-light setup: broad, bright, and still soft.
    # (location, energy, size, color_rgb)
    lights = [
        (( 0.0, -10.5,  7.0), 1250, 10, (1.00, 0.98, 0.95)),  # frontal key
        ((-8.5,  -7.0,  5.0),  560, 16, (0.94, 0.97, 1.00)),  # soft fill
        (( 7.0,   8.0,  5.5),  300, 13, (0.86, 0.91, 1.00)),  # rim / back
        (( 0.0,  -4.0, -5.5),  170, 12, (1.00, 0.96, 0.90)),  # weak bounce
    ]
    for loc, energy, size, col in lights:
        bpy.ops.object.light_add(type='AREA', location=loc)
        obj = bpy.context.active_object
        direction = Vector((0, 0, 0)) - Vector(loc)
        obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        obj.data.energy = energy
        obj.data.size   = size
        obj.data.color  = col


# ============================================================================
#  WORLD BACKGROUND
# ============================================================================

def setup_world():
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new('World')
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputWorld')
    bg  = nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value    = (0.03, 0.03, 0.05, 1.0)
    bg.inputs['Strength'].default_value = 0.70
    links.new(bg.outputs['Background'], out.inputs['Surface'])


# ============================================================================
#  RENDER SETTINGS
# ============================================================================

def setup_render(output_path, quality_key='preview'):
    q     = QUALITY[quality_key]
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    # GPU detection
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for dtype in ('OPTIX', 'CUDA', 'HIP', 'METAL'):
            try:
                prefs.compute_device_type = dtype
                prefs.get_devices()
                if any(d.type != 'CPU' for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = True
                    scene.cycles.device = 'GPU'
                    print(f'GPU acceleration: {dtype}')
                    break
            except Exception:
                continue
        else:
            scene.cycles.device = 'CPU'
    except Exception:
        scene.cycles.device = 'CPU'

    scene.cycles.samples       = q['samples']
    scene.cycles.use_denoising = q['denoise']

    scene.render.resolution_x          = q['resolution']
    scene.render.resolution_y          = q['resolution']
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode  = 'RGBA'
    scene.render.film_transparent           = True

    scene.render.filepath = output_path

    scene.cycles.max_bounces          = 8
    scene.cycles.diffuse_bounces      = 4
    scene.cycles.glossy_bounces       = 4
    scene.cycles.transmission_bounces = 2
    scene.cycles.volume_bounces       = 0

    try:
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look           = 'Medium High Contrast'
        scene.view_settings.exposure       = 0.30
        scene.view_settings.gamma          = 1.0
    except TypeError:
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look           = 'Medium Contrast'
        scene.view_settings.exposure       = 0.30
        scene.view_settings.gamma          = 1.0


# ============================================================================
#  MAIN
# ============================================================================

def main():
    random.seed(MASTER_SEED)

    print('-- Clearing scene ----------------------------------------')
    clear_scene()

    print('-- Building materials -----------------------------------')
    mats = build_material_palette()

    print('-- Free-space particle cloud ----------------------------')
    bounds = (X_BOUNDS, Y_BOUNDS, Z_BOUNDS)
    rng_cloud = random.Random(MASTER_SEED)
    scatter_free_particles(
        count        = PARTICLE_COUNT,
        mats         = mats,
        bounds       = bounds,
        radius_range = PARTICLE_RADIUS_RANGE,
        subdivisions = SPHERE_SUBDIVISIONS,
        rng          = rng_cloud,
    )

    print('-- Camera ------------------------------------------------')
    setup_camera(
        cam_loc  = CAM_LOCATION,
        target   = CAM_TARGET,
        focal_mm = CAM_FOCAL_MM,
        fstop    = CAM_FSTOP,
    )

    print('-- Lighting ----------------------------------------------')
    setup_lighting()

    print('-- World -------------------------------------------------')
    setup_world()

    print('-- Render settings ---------------------------------------')
    setup_render(OUTPUT_PATH, RENDER_QUALITY)

    print(f'-- Rendering [{RENDER_QUALITY}] --------------------------')
    bpy.ops.render.render(write_still=True)
    print(f'\n  Saved: {OUTPUT_PATH}')


main()
