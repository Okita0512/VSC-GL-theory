"""
Blender Python script: Outside-Cavity Liquid with Weak Rayleigh Scattering

Run from Blender's Scripting tab or via:
    blender --background --python free_space_particles_weak_rayleigh.py

This script creates a dispersed, non-clustered free-space particle ensemble and
adds a weak 400-nm-style probe/scattering motif for use as the outside-cavity
control in a TOC graphic.  The background remains transparent and no text is
rendered; labels can be added later in Illustrator/PowerPoint.
"""

import bpy
import math
import os
import random
try:
    from mathutils import Vector  # type: ignore  # Blender-internal
except ImportError:
    pass


# ============================================================================
#  ADJUSTABLE PARAMETERS
# ============================================================================

RENDER_QUALITY = 'preview'    # 'preview' (fast) | 'final' (high quality)
MASTER_SEED    = 42

# Keep the output beside this script, i.e. in the same folder as the VSC asset.
if '__file__' in globals():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    SCRIPT_DIR = r"C:\Users\Wenxiang Ying\Desktop\Fig1"
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "free_space_weak_rayleigh.png")

# Dispersed outside-cavity liquid: enough particles to read as a liquid, but
# sparse enough to remain legible when reduced to TOC size.
PARTICLE_COUNT        = 36
X_BOUNDS              = (-4.4, 4.4)
Y_BOUNDS              = (-2.6, 2.6)
Z_BOUNDS              = (-2.15, 2.15)
PARTICLE_RADIUS_RANGE = (0.17, 0.31)
MIN_CENTER_SEPARATION = 0.70
SURFACE_GAP_FACTOR    = 1.06
PLACEMENT_JITTER      = 0.66
MAX_PLACEMENT_PASSES  = 6
SPHERE_SUBDIVISIONS   = 2

MATERIAL_SEQUENCE = (
    'red',
    'pale_blue',
    'light_gray',
    'light_beige',
)

# Weak optical response.  Relative to the VSC panel, the control has fewer,
# shorter, thinner rays and no strong scattering halo.
SHOW_WEAK_RAYLEIGH = True
SHOW_PROBE_BEAM     = True   # set False for an even cleaner control inset
OPTICAL_COLOR       = (0.28, 0.36, 0.95)   # blue-violet, visible on white
OUTLINE_COLOR       = (0.10, 0.13, 0.22)   # dark blue-gray stroke

PROBE_START              = (-5.15, -2.95,  2.15)
PROBE_CORE_RADIUS        = 0.040
PROBE_OUTLINE_RADIUS     = 0.066
PROBE_HALO_RADIUS        = 0.105
PROBE_ARROW_LENGTH       = 0.38
PROBE_ARROW_RADIUS       = 0.13
PROBE_CORE_STRENGTH      = 2.2
PROBE_HALO_STRENGTH      = 0.70
PROBE_HALO_OPACITY       = 0.18

WEAK_SCATTER_RAY_COUNT   = 4
WEAK_SCATTER_ANGLES_DEG  = (-30.0, -9.0, 12.0, 33.0)
WEAK_SCATTER_LENGTHS     = (1.15, 1.45, 1.35, 1.05)
WEAK_SCATTER_CORE_RADIUS = 0.026
WEAK_SCATTER_OUTLINE_RADIUS = 0.045
WEAK_SCATTER_ARROW_LENGTH   = 0.25
WEAK_SCATTER_ARROW_RADIUS   = 0.085
WEAK_SCATTER_STRENGTH       = 1.25

# Camera / output are landscape-oriented so the asset can sit compactly in the
# upper-left portion of a composite TOC.
CAM_LOCATION  = (0.0, -17.0, 1.45)
CAM_TARGET    = (0.0,   0.0, 0.0)
CAM_FOCAL_MM  = 49
CAM_FSTOP     = 5.0

QUALITY = {
    'preview': dict(samples=64,  resolution_x=1000, resolution_y=620, denoise=True),
    'final':   dict(samples=300, resolution_x=1600, resolution_y=992, denoise=True),
}


# ============================================================================
#  SCENE HELPERS
# ============================================================================

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for db in (bpy.data.meshes, bpy.data.materials,
               bpy.data.lights, bpy.data.cameras, bpy.data.curves):
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


def create_emission_material(name, color, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*color, 1.0)
    emission.inputs['Strength'].default_value = strength
    links.new(emission.outputs['Emission'], out.inputs['Surface'])
    return mat


def create_transparent_emission_material(name, color, strength, opacity):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    emission = nodes.new('ShaderNodeEmission')
    mix = nodes.new('ShaderNodeMixShader')
    emission.inputs['Color'].default_value = (*color, 1.0)
    emission.inputs['Strength'].default_value = strength
    mix.inputs[0].default_value = opacity
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], out.inputs['Surface'])
    return mat


def build_material_palette():
    return {
        'red': create_material(
            'MutedRed',
            base_color=(0.72, 0.17, 0.14),
            roughness=0.42,
            subsurface=0.03,
        ),
        'pale_blue': create_material(
            'PaleBlue',
            base_color=(0.64, 0.79, 0.94),
            roughness=0.42,
            subsurface=0.04,
        ),
        'light_gray': create_material(
            'LightGray',
            base_color=(0.80, 0.80, 0.83),
            roughness=0.46,
            subsurface=0.03,
        ),
        'light_beige': create_material(
            'LightBeige',
            base_color=(0.93, 0.88, 0.77),
            roughness=0.43,
            subsurface=0.04,
        ),
        'optical_core': create_emission_material(
            'WeakRayleighCore', OPTICAL_COLOR, WEAK_SCATTER_STRENGTH),
        'probe_core': create_emission_material(
            'ProbeCore', OPTICAL_COLOR, PROBE_CORE_STRENGTH),
        'optical_outline': create_material(
            'OpticalOutline', OUTLINE_COLOR, roughness=0.38),
        'probe_halo': create_transparent_emission_material(
            'ProbeHalo', OPTICAL_COLOR,
            PROBE_HALO_STRENGTH, PROBE_HALO_OPACITY),
    }


# ============================================================================
#  SPHERE INSTANCING
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
    mesh = _get_sphere_mesh(material, subdivisions)
    obj = bpy.data.objects.new('Particle', mesh)
    obj.location = location
    obj.scale = (radius, radius, radius)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# ============================================================================
#  FREE-SPACE PARTICLE PLACEMENT
# ============================================================================

def _grid_dimensions(count, bounds):
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

        if placement_pass == MAX_PLACEMENT_PASSES - 2:
            jitter = min(0.82, jitter + 0.12)

    if len(objects) < count:
        print(f'Warning: placed {len(objects)} of {count} requested particles.')
    return objects


# ============================================================================
#  OPTICAL ANNOTATIONS
# ============================================================================

def create_cylinder_between(start, end, radius, material, name):
    start_v = Vector(start)
    end_v = Vector(end)
    direction = end_v - start_v
    length = direction.length
    if length <= 1.0e-8:
        raise ValueError(f'Cannot create zero-length cylinder: {name}')

    midpoint = (start_v + end_v) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32, radius=radius, depth=length, location=midpoint)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    obj.data.materials.append(material)
    return obj


def create_cone_tip(end, direction, radius, length, material, name):
    unit = Vector(direction).normalized()
    center = Vector(end) - unit * (0.5 * length)
    bpy.ops.mesh.primitive_cone_add(
        vertices=32, radius1=radius, radius2=0.0,
        depth=length, location=center)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = unit.to_track_quat('Z', 'Y').to_euler()
    obj.data.materials.append(material)
    return obj


def create_stroked_arrow(start, end,
                         core_radius, outline_radius,
                         arrow_length, arrow_radius,
                         core_material, outline_material,
                         name_prefix):
    start_v = Vector(start)
    end_v = Vector(end)
    direction = end_v - start_v
    unit = direction.normalized()
    shaft_end = end_v - unit * arrow_length

    # Dark shell first, then the colored core.
    create_cylinder_between(
        start_v, shaft_end, outline_radius, outline_material,
        f'{name_prefix}_OutlineShaft')
    create_cone_tip(
        end_v, unit, arrow_radius * 1.28, arrow_length * 1.14,
        outline_material, f'{name_prefix}_OutlineHead')

    create_cylinder_between(
        start_v, shaft_end, core_radius, core_material,
        f'{name_prefix}_CoreShaft')
    create_cone_tip(
        end_v, unit, arrow_radius, arrow_length,
        core_material, f'{name_prefix}_CoreHead')


def choose_probe_particle(particles):
    """Pick a near-central, camera-facing particle as the weak scatterer."""
    if not particles:
        raise RuntimeError('No particles are available for optical annotation.')

    def score(obj):
        # Favor center in the image plane and the camera-facing half of the cloud.
        x, y, z = obj.location
        return x * x + 1.25 * z * z + 0.12 * (y + 1.5) ** 2

    return min(particles, key=score)


def add_weak_probe_and_scattering(particles, mats):
    scatterer = choose_probe_particle(particles)
    radius = float(scatterer.scale.x)

    # Put the optical geometry slightly in front of the liquid so it remains
    # visible after compositing on a white TOC background.
    front_y = float(scatterer.location.y) - radius * 1.35 - 0.08
    impact = Vector((
        float(scatterer.location.x) - 0.12,
        front_y,
        float(scatterer.location.z) + 0.05,
    ))

    # Incident probe: optional, because a very small TOC inset may read more
    # cleanly with only the weak scattered rays.
    if SHOW_PROBE_BEAM:
        probe_start = Vector(PROBE_START)
        probe_start.y = front_y
        direction = impact - probe_start
        unit = direction.normalized()
        shaft_end = impact - unit * PROBE_ARROW_LENGTH

        create_cylinder_between(
            probe_start, shaft_end, PROBE_HALO_RADIUS,
            mats['probe_halo'], 'Probe_Halo')

        create_stroked_arrow(
            probe_start, impact,
            PROBE_CORE_RADIUS, PROBE_OUTLINE_RADIUS,
            PROBE_ARROW_LENGTH, PROBE_ARROW_RADIUS,
            mats['probe_core'], mats['optical_outline'],
            'Probe')

    # Weak Rayleigh response: only four short, thin rays with no large halo.
    rng = random.Random(MASTER_SEED + 101)
    for i in range(WEAK_SCATTER_RAY_COUNT):
        angle = math.radians(WEAK_SCATTER_ANGLES_DEG[i])
        length = WEAK_SCATTER_LENGTHS[i] * rng.uniform(0.94, 1.06)
        direction_xz = Vector((math.cos(angle), 0.0, math.sin(angle)))
        start = impact + direction_xz * 0.06
        end = start + direction_xz * length

        create_stroked_arrow(
            start, end,
            WEAK_SCATTER_CORE_RADIUS,
            WEAK_SCATTER_OUTLINE_RADIUS,
            WEAK_SCATTER_ARROW_LENGTH,
            WEAK_SCATTER_ARROW_RADIUS,
            mats['optical_core'], mats['optical_outline'],
            f'WeakScatter_{i:02d}')


# ============================================================================
#  CAMERA / LIGHTING / WORLD / RENDER
# ============================================================================

def setup_camera(cam_loc, target, focal_mm, fstop):
    bpy.ops.object.camera_add(location=cam_loc)
    cam_obj = bpy.context.active_object

    direction = Vector(target) - Vector(cam_loc)
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    cam = cam_obj.data
    cam.lens = focal_mm
    cam.sensor_width = 36
    cam.dof.use_dof = True
    cam.dof.focus_distance = direction.length
    cam.dof.aperture_fstop = fstop

    bpy.context.scene.camera = cam_obj
    return cam_obj


def setup_lighting():
    lights = [
        (( 0.0, -10.5,  7.0), 1250, 10, (1.00, 0.98, 0.95)),
        ((-8.5,  -7.0,  5.0),  560, 16, (0.94, 0.97, 1.00)),
        (( 7.0,   8.0,  5.5),  300, 13, (0.86, 0.91, 1.00)),
        (( 0.0,  -4.0, -5.5),  170, 12, (1.00, 0.96, 0.90)),
    ]
    for loc, energy, size, col in lights:
        bpy.ops.object.light_add(type='AREA', location=loc)
        obj = bpy.context.active_object
        direction = Vector((0, 0, 0)) - Vector(loc)
        obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        obj.data.energy = energy
        obj.data.size = size
        obj.data.color = col


def setup_world():
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new('World')
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputWorld')
    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.03, 0.03, 0.05, 1.0)
    bg.inputs['Strength'].default_value = 0.70
    links.new(bg.outputs['Background'], out.inputs['Surface'])


def setup_render(output_path, quality_key='preview'):
    q = QUALITY[quality_key]
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

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

    scene.cycles.samples = q['samples']
    scene.cycles.use_denoising = q['denoise']

    scene.render.resolution_x = q['resolution_x']
    scene.render.resolution_y = q['resolution_y']
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True
    scene.render.filepath = output_path

    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 2
    scene.cycles.volume_bounces = 0

    try:
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'Medium High Contrast'
        scene.view_settings.exposure = 0.30
        scene.view_settings.gamma = 1.0
    except TypeError:
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'Medium Contrast'
        scene.view_settings.exposure = 0.30
        scene.view_settings.gamma = 1.0


# ============================================================================
#  MAIN
# ============================================================================

def main():
    random.seed(MASTER_SEED)

    print('-- Clearing scene ----------------------------------------')
    clear_scene()

    print('-- Building materials -----------------------------------')
    mats = build_material_palette()

    print('-- Outside-cavity liquid --------------------------------')
    bounds = (X_BOUNDS, Y_BOUNDS, Z_BOUNDS)
    rng_cloud = random.Random(MASTER_SEED)
    particles = scatter_free_particles(
        count=PARTICLE_COUNT,
        mats=mats,
        bounds=bounds,
        radius_range=PARTICLE_RADIUS_RANGE,
        subdivisions=SPHERE_SUBDIVISIONS,
        rng=rng_cloud,
    )

    if SHOW_WEAK_RAYLEIGH:
        print('-- Weak Rayleigh response -------------------------------')
        add_weak_probe_and_scattering(particles, mats)

    print('-- Camera ------------------------------------------------')
    setup_camera(
        cam_loc=CAM_LOCATION,
        target=CAM_TARGET,
        focal_mm=CAM_FOCAL_MM,
        fstop=CAM_FSTOP,
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
