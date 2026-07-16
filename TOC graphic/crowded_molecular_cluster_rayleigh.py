"""
Blender Python script: Molecules Inside a Fabry-Pérot Cavity
Run from Blender's Scripting tab or via:
    blender --background --python crowded_molecular_cluster_rayleigh.py

Geometry note
─────────────
Camera is at (0, -16.5, 0) looking straight along +Y.
Cavity mirrors are thin plates in the XY plane at Z = ±CAVITY_HALF_HEIGHT.
The camera sees each mirror's large inner face (XY extent) foreshortened by
the elevation angle arctan(CAVITY_HALF_HEIGHT / cam_distance) ≈ 17 °, giving
an apparent height of MIRROR_DEPTH × sin(17°) ≈ 1.7 units per mirror.
Reducing MIRROR_THICKNESS makes the plate thinner without losing visibility.

Additive note
─────────────
This file is a strict additive fork of crowded_molecular_cluster.py. Every
function and setting from the original is unchanged; the only new content is
a single 400 nm incident beam and an outgoing Rayleigh-scattering ray fan,
each with a dark inverted-hull outline shell for visibility against the
white cavity region, added as new objects at the end of main(). No existing
material, world, camera, lighting, or render setting is modified, and no
compositor post-processing is added. This version carries no text — labels
are added externally after render.
"""

import bpy
import random
import math
try:
    from mathutils import Vector  # type: ignore  # Blender-internal, not on system path
    import bmesh  # type: ignore  # Blender-internal, not on system path
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
#  ADJUSTABLE PARAMETERS — edit here without touching the functions below
# ══════════════════════════════════════════════════════════════════════════════

RENDER_QUALITY = 'preview'    # 'preview' (fast) | 'final' (high quality)
MASTER_SEED    = 42
OUTPUT_PATH    = "C:/Users/Wenxiang Ying/Desktop/Fig1/cavity_molecular_cluster_rayleigh.png"

# ── Central cluster ───────────────────────────────────────────────────────────
CENTRAL_SPHERE_COUNT = 65     # spheres in the dominant red cluster
CENTRAL_BASE_RADIUS  = 0.38   # mean sphere radius
CENTRAL_CLUSTER_R    = 2.0    # packing radius of the aggregate

# ── Surrounding clusters ──────────────────────────────────────────────────────
# Each entry: (n_spheres, color_key, offset_x, offset_y, offset_z)
# All offsets are relative to central_center = (0, 0, 0).
# X ≥ 3.5 keeps clusters outside the GLOW_RADIUS = 2.5 beam boundary.
# Varied Y (depth) offsets spread them in the DOF blur gradient:
#   negative Y → closer to camera → foreground blur
#   positive Y → farther from camera → background blur
SURROUNDING_CLUSTERS = [
    ( 6, 'red_weak',    -5.5, -2.0,  1.8),   # left, slightly in front
    ( 7, 'red_weak',     5.2,  3.2, -1.2),   # right, behind focal plane
    ( 5, 'pale_blue',   -5.0,  5.2,  0.8),   # left, far background
    ( 6, 'light_gray',   5.0, -3.8, -1.0),   # right, in front
    ( 5, 'light_beige',  3.8, -5.5, -1.8),   # centre-right, far foreground
]

# ── Isolated monomers ─────────────────────────────────────────────────────────
ISOLATED_COUNT = 4            # a few lone spheres for spatial context

# ── Cavity mirrors ────────────────────────────────────────────────────────────
# These are intentionally thin plates — the "thin Fabry-Pérot mirror" look.
# Visibility comes from the large inner face seen at ~17° elevation via
# foreshortening: apparent height ≈ MIRROR_DEPTH × sin(17°) ≈ 1.7 units.
# Do NOT increase MIRROR_THICKNESS back to large values — it makes them
# look like gold ingots instead of optical cavity mirrors.
CAVITY_HALF_HEIGHT = 5.0      # mirror centres at Z = ±CAVITY_HALF_HEIGHT
MIRROR_WIDTH       = 8.4      # X extent — ~62 % of frame width at 44 mm/16.5 m
MIRROR_DEPTH       = 6.0      # Y extent — determines foreshortened face height
MIRROR_THICKNESS   = 0.13     # Z thickness — intentionally very thin plate
MIRROR_BEVEL       = 0.025    # bevel edge width in world units after scale apply

# ── Cavity mode glow ──────────────────────────────────────────────────────────
CAVITY_GLOW   = True
GLOW_RADIUS   = 2.5           # XY beam-waist radius
GLOW_COLOR    = (0.60, 0.82, 1.0)   # light cyan
GLOW_STRENGTH = 0.22          # subtle — must not overpower spheres
# GLOW_DENSITY  = 0.04          # low density keeps glow semi-transparent
GLOW_DENSITY  = 0.02          # low density keeps glow semi-transparent#

# ── Camera — straight-on frontal view ────────────────────────────────────────
CAM_LOCATION  = (0.0, -16.5, 0.0)   # centred on cavity axis
CAM_TARGET    = (0.0,   0.0, 0.0)   # exact cavity centre
CAM_FOCAL_MM  = 44                   # keeps full mirror height inside frame
CAM_FSTOP     = 3.2                  # moderate DOF: centre sharp, sides soft (unchanged)

# ── Quality presets ───────────────────────────────────────────────────────────
QUALITY = {
    'preview': dict(samples=64,  resolution=900,  denoise=True),
    'final':   dict(samples=300, resolution=1400, denoise=True),
}

# ── Optical annotations (additive only — new objects, nothing above is touched) ─
SHOW_OPTICAL_ANNOTATIONS = True

# Incident 400 nm probe beam
BEAM_START             = (-5.8, -1.8, 2.8)
BEAM_END                = (-0.7, -1.8, 0.4)
BEAM_RADIUS             = 0.045
BEAM_ARROW_LENGTH       = 0.35
BEAM_ARROW_RADIUS       = 0.11
BEAM_COLOR              = (0.30, 0.35, 1.00)   # blue-violet, reads as ~400 nm and stays visible on white cavity glow
BEAM_EMISSION_STRENGTH  = 5.0

# Soft outer glow sleeve around the incident beam only (not the scattering rays)
BEAM_GLOW_RADIUS_SCALE      = 4.2     # outer sleeve radius = BEAM_RADIUS * this (broader halo)
BEAM_GLOW_EMISSION_STRENGTH = 1.5     # softer than the core, slightly reduced for the larger area
BEAM_GLOW_ALPHA              = 0.16   # semi-transparent, restrained despite the wider sleeve

# Outgoing enhanced Rayleigh scattering — fan of rays from the cluster surface
SCATTER_ORIGIN            = (0.1, -2.0, 0.1)
SCATTER_RAY_COUNT         = 9
SCATTER_ANGLE_MIN_DEG     = -40.0   # relative to screen-space +X direction
SCATTER_ANGLE_MAX_DEG     = 40.0
SCATTER_LENGTH_MIN        = 1.6
SCATTER_LENGTH_MAX        = 2.9
SCATTER_RADIUS            = 0.030
SCATTER_ARROW_LENGTH      = 0.26
SCATTER_ARROW_RADIUS      = 0.075
SCATTER_COLOR             = BEAM_COLOR   # matches the incident beam exactly, per request
SCATTER_EMISSION_STRENGTH = 7.0
SCATTER_SEED              = MASTER_SEED + 101   # independent RNG — never touches molecule layout

# Dark outline / stroke around the incident beam and scattering rays, for
# visibility against the white cavity region. Implemented as an enlarged,
# inverted-normal, backface-culled shell around each shaft/arrowhead (classic
# inverted-hull outline): the shell is invisible everywhere except the thin
# rim where it extends past the colored core, so the core stays visible.
OUTLINE_COLOR             = (0.15, 0.18, 0.28)   # dark blue-gray — reads clearly on white, stays elegant
OUTLINE_EMISSION_STRENGTH = 1.6                  # flat, modest — a stroke, not another glow
BEAM_OUTLINE_RADIUS_SCALE    = 1.7   # outline shell radius = beam core/arrow radius * this
SCATTER_OUTLINE_RADIUS_SCALE = 1.7   # outline shell radius = ray core/arrow radius * this

# ══════════════════════════════════════════════════════════════════════════════
#  SCENE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for db in (bpy.data.meshes, bpy.data.materials,
               bpy.data.lights, bpy.data.cameras):
        for block in list(db):
            db.remove(block)


# ══════════════════════════════════════════════════════════════════════════════
#  MATERIALS
# ══════════════════════════════════════════════════════════════════════════════

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


def create_volume_emission_material(name, color, density, emit_strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    vol = nodes.new('ShaderNodeVolumePrincipled')
    vol.inputs['Emission Color'].default_value    = (*color, 1.0)
    vol.inputs['Emission Strength'].default_value = emit_strength
    vol.inputs['Density'].default_value           = density
    links.new(vol.outputs['Volume'], out.inputs['Volume'])
    return mat


def build_material_palette():
    return {
        # Primary red: muted brick-rose — slightly brightened (was 0.70, 0.15, 0.13)
        'red': create_material(
            'Red',
            base_color = (0.74, 0.18, 0.16),
            roughness  = 0.40,
            subsurface = 0.03,
        ),
        # Weaker red for side clusters — slightly brightened (was 0.48, 0.09, 0.08)
        'red_weak': create_material(
            'RedWeak',
            base_color = (0.54, 0.12, 0.10),
            roughness  = 0.46,
            subsurface = 0.02,
        ),
        # Neutral tones — slightly brightened for consistency with red bump
        'pale_blue':   create_material('PaleBlue',   (0.64, 0.78, 0.93), 0.42, subsurface=0.04),  # was 0.58, 0.73, 0.90
        'light_gray':  create_material('LightGray',  (0.79, 0.79, 0.82), 0.46, subsurface=0.03),  # was 0.74, 0.74, 0.77
        'light_beige': create_material('LightBeige', (0.91, 0.86, 0.76), 0.43, subsurface=0.04),  # was 0.88, 0.83, 0.72
        # Gold mirror: fully metallic, low roughness for strong reflections
        'gold_mirror': create_material(
            'GoldMirror',
            base_color = (0.831, 0.686, 0.215),
            roughness  = 0.10,
            metallic   = 1.0,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SPHERE INSTANCING — one shared mesh per (material, subdivision) pair
# ══════════════════════════════════════════════════════════════════════════════

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
    """Instance a shared unit icosphere — no unique mesh per sphere."""
    mesh = _get_sphere_mesh(material, subdivisions)
    obj  = bpy.data.objects.new('Sph', mesh)
    obj.location = location
    obj.scale    = (radius, radius, radius)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTER BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def create_cluster(center, num_spheres, base_radius, radius_var,
                   material, cluster_radius, subdivisions=2, rng=None):
    if rng is None:
        rng = random
    placed, objects = [], []
    max_attempts = num_spheres * 30
    attempts = 0
    while len(objects) < num_spheres and attempts < max_attempts:
        attempts += 1
        r     = cluster_radius * (rng.random() ** (1.0 / 3.0))
        cos_t = 2.0 * rng.random() - 1.0
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t ** 2))
        phi   = rng.uniform(0, 2 * math.pi)
        sx = center[0] + r * sin_t * math.cos(phi)
        sy = center[1] + r * sin_t * math.sin(phi)
        sz = center[2] + r * cos_t
        sr = base_radius * rng.uniform(1.0 - radius_var, 1.0 + radius_var)
        ok = all(
            math.sqrt((sx-px)**2 + (sy-py)**2 + (sz-pz)**2) >= (sr+pr) * 0.68
            for px, py, pz, pr in placed
        )
        if ok:
            objects.append(create_sphere((sx, sy, sz), sr, material, subdivisions))
            placed.append((sx, sy, sz, sr))
    return objects


# ══════════════════════════════════════════════════════════════════════════════
#  CAVITY MIRRORS
# ══════════════════════════════════════════════════════════════════════════════

def create_cavity_mirrors(gold_mat, half_height, width, depth, thickness, bevel):
    """
    Two thin gold rectangular plates at Z = ±half_height.

    Visibility from the straight-on frontal camera
    ──────────────────────────────────────────────
    The camera at (0, -16.5, 0) sees each mirror's large inner XY face via
    foreshortening.  The apparent face height on screen is approximately
        depth × sin( arctan(half_height / |cam_y|) )
    so visibility is controlled by MIRROR_DEPTH, not by MIRROR_THICKNESS.
    Keep MIRROR_THICKNESS small for the thin-plate Fabry-Pérot look.
    """
    for sign, name in ((+1, 'Mirror_Top'), (-1, 'Mirror_Bot')):
        bpy.ops.mesh.primitive_cube_add(
            size=1.0, location=(0.0, 0.0, sign * half_height))
        obj      = bpy.context.active_object
        obj.name = name
        obj.scale = (width, depth, thickness)

        # Apply scale so bevel.width operates in world-space units
        bpy.ops.object.transform_apply(scale=True)

        bev          = obj.modifiers.new('Bevel', 'BEVEL')
        bev.width    = bevel     # small: thin plate needs small corner radius
        bev.segments = 3

        for poly in obj.data.polygons:
            poly.use_smooth = True

        obj.data.materials.clear()
        obj.data.materials.append(gold_mat)


# ══════════════════════════════════════════════════════════════════════════════
#  CAVITY MODE GLOW
# ══════════════════════════════════════════════════════════════════════════════

def create_cavity_glow(center, radius_xy, height, color, strength, density):
    """Volumetric emission cylinder — soft hint of the cavity field mode."""
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius_xy, depth=height, vertices=48, location=center)
    obj      = bpy.context.active_object
    obj.name = 'CavityMode'
    mat = create_volume_emission_material('CavityModeMat', color, density, strength)
    obj.data.materials.append(mat)
    return obj


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  LIGHTING
# ══════════════════════════════════════════════════════════════════════════════

def setup_lighting():
    # Studio-like area-light setup: brighter but still broad and soft.
    # (location, energy, size, color_rgb)
    lights = [
        (( 0.0, -11.0,  8.0), 1450, 11, (1.00, 0.98, 0.95)),  # frontal key
        ((-9.0,  -7.5,  5.5),  620, 17, (0.94, 0.97, 1.00)),  # soft fill
        (( 7.5,   9.0,  6.0),  330, 15, (0.86, 0.91, 1.00)),  # rim / back
        (( 0.0,  -5.0, -6.0),  190, 13, (1.00, 0.96, 0.90)),  # weak bounce
    ]
    for loc, energy, size, col in lights:
        bpy.ops.object.light_add(type='AREA', location=loc)
        obj = bpy.context.active_object
        direction = Vector((0, 0, 0)) - Vector(loc)
        obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        obj.data.energy = energy
        obj.data.size   = size
        obj.data.color  = col


# ══════════════════════════════════════════════════════════════════════════════
#  WORLD BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════

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
    bg.inputs['Strength'].default_value = 0.85  # dark world, slightly stronger ambient fill
    links.new(bg.outputs['Background'], out.inputs['Surface'])


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

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
    scene.render.image_settings.color_mode  = 'RGBA'  # preserve alpha
    scene.render.film_transparent           = True    # transparent background

    scene.render.filepath = output_path

    # High glossy bounces keep the gold mirrors looking properly metallic
    scene.cycles.max_bounces          = 12
    scene.cycles.diffuse_bounces      = 4
    scene.cycles.glossy_bounces       = 8
    scene.cycles.transmission_bounces = 4
    scene.cycles.volume_bounces       = 2   # required for cavity glow

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


# ══════════════════════════════════════════════════════════════════════════════
#  OPTICAL ANNOTATIONS (additive only) — 400 nm beam, Rayleigh scattering, labels
# ══════════════════════════════════════════════════════════════════════════════

def create_emission_material(name, color, strength=4.0):
    """Plain emissive surface shader — used only by the new beam/ray/text objects."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out  = nodes.new('ShaderNodeOutputMaterial')
    emit = nodes.new('ShaderNodeEmission')
    emit.inputs['Color'].default_value    = (*color, 1.0)
    emit.inputs['Strength'].default_value = strength
    links.new(emit.outputs['Emission'], out.inputs['Surface'])
    return mat


def create_transparent_emission_material(name, color, strength=2.0, alpha=0.25):
    """Semi-transparent emissive shader — used only for the incident beam's
    outer glow sleeve, kept local to that one object so nothing else is affected."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out    = nodes.new('ShaderNodeOutputMaterial')
    emit   = nodes.new('ShaderNodeEmission')
    transp = nodes.new('ShaderNodeBsdfTransparent')
    mix    = nodes.new('ShaderNodeMixShader')
    emit.inputs['Color'].default_value    = (*color, 1.0)
    emit.inputs['Strength'].default_value = strength
    mix.inputs['Fac'].default_value = alpha
    links.new(transp.outputs['BSDF'], mix.inputs[1])
    links.new(emit.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], out.inputs['Surface'])
    return mat


def create_cylinder_between(start, end, radius, material, vertices=20, name='Cylinder'):
    """Thin cylindrical shaft from `start` to `end`. Blender cylinders are built
    along local Z, so local Z is track-rotated onto the segment direction."""
    start_v = Vector(start)
    end_v   = Vector(end)
    direction = end_v - start_v
    length = direction.length
    if length < 1e-6:
        return None
    midpoint = (start_v + end_v) / 2.0
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=length, vertices=vertices, location=midpoint)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    for poly in obj.data.polygons:
        poly.use_smooth = True
    obj.data.materials.append(material)
    return obj


def create_arrow_between(start, end, shaft_radius, material, arrow_length, arrow_radius,
                         vertices=20, name='Arrow'):
    """Shaft (cylinder) + small cone arrowhead, apex exactly at `end`."""
    start_v = Vector(start)
    end_v   = Vector(end)
    direction = end_v - start_v
    total_length = direction.length
    if total_length < 1e-6:
        return None, None
    dir_norm  = direction.normalized()
    head_len  = min(arrow_length, total_length * 0.6)
    shaft_end = end_v - dir_norm * head_len

    shaft = create_cylinder_between(
        start_v, shaft_end, shaft_radius, material, vertices=vertices, name=f'{name}_Shaft')

    base_center = end_v - dir_norm * head_len
    mid = (end_v + base_center) / 2.0
    bpy.ops.mesh.primitive_cone_add(
        radius1=arrow_radius, radius2=0.0, depth=head_len,
        vertices=max(vertices, 24), location=mid)
    head = bpy.context.active_object
    head.name = f'{name}_Head'
    head.rotation_euler = dir_norm.to_track_quat('Z', 'Y').to_euler()
    for poly in head.data.polygons:
        poly.use_smooth = True
    head.data.materials.append(material)

    return shaft, head


def flip_normals(mesh):
    """Reverse face winding (and thus normal direction) of a mesh in place.
    Used only on the enlarged outline shells — see create_outlined_arrow_between."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()


def create_outline_material(name, color, strength=1.6):
    """Flat dark emissive shell material with backface culling enabled, so
    once paired with an enlarged, normal-flipped shell it only shows at the
    silhouette rim (classic inverted-hull outline) and lets the core show through."""
    mat = create_emission_material(name, color, strength)
    mat.use_backface_culling = True
    return mat


def create_outlined_arrow_between(start, end, core_radius, core_material, arrow_length, arrow_radius,
                                  outline_material, outline_radius_scale, vertices=20, name='Arrow'):
    """An arrow (shaft + arrowhead) plus a slightly larger, inverted-normal,
    backface-culled outline shell around it. The shell is invisible except at
    the rim where it extends past the core, producing a dark stroke around the
    colored beam/ray without hiding it."""
    outline_shaft, outline_head = create_arrow_between(
        start, end, core_radius * outline_radius_scale, outline_material,
        arrow_length, arrow_radius * outline_radius_scale,
        vertices=vertices, name=f'{name}_Outline')
    for shell_obj in (outline_shaft, outline_head):
        if shell_obj is not None:
            flip_normals(shell_obj.data)

    core_shaft, core_head = create_arrow_between(
        start, end, core_radius, core_material, arrow_length, arrow_radius,
        vertices=vertices, name=name)

    return outline_shaft, outline_head, core_shaft, core_head


def add_incident_beam(start, end, core_radius, arrow_length, arrow_radius, color, emission_strength,
                      glow_radius_scale=2.4, glow_emission_strength=1.8, glow_alpha=0.22,
                      outline_color=(0.15, 0.18, 0.28), outline_emission_strength=1.6,
                      outline_radius_scale=1.7):
    """Thin blue-violet beam + arrowhead, wrapped in a dark outline shell for
    visibility on white, plus a soft semi-transparent outer glow sleeve
    (this object's own materials only — no scene-wide bloom)."""
    mat = create_emission_material('IncidentBeamMat', color, emission_strength)
    outline_mat = create_outline_material('IncidentBeamOutlineMat', outline_color, outline_emission_strength)
    outline_shaft, outline_head, shaft, head = create_outlined_arrow_between(
        start, end, core_radius, mat, arrow_length, arrow_radius,
        outline_mat, outline_radius_scale, name='IncidentBeam')

    glow_mat = create_transparent_emission_material(
        'IncidentBeamGlowMat', color, strength=glow_emission_strength, alpha=glow_alpha)
    dir_norm = (Vector(end) - Vector(start)).normalized()
    glow_end = Vector(end) - dir_norm * (arrow_length * 0.5)   # stop short of the arrow tip
    glow = create_cylinder_between(
        start, glow_end, core_radius * glow_radius_scale, glow_mat,
        vertices=16, name='IncidentBeamGlow')

    return shaft, head, glow, outline_shaft, outline_head


def add_scattering_rays(origin, ray_count, angle_min_deg, angle_max_deg,
                        length_min, length_max, core_radius, arrow_length, arrow_radius,
                        color, emission_strength, seed,
                        outline_color=(0.15, 0.18, 0.28), outline_emission_strength=1.6,
                        outline_radius_scale=1.7):
    """Small fan of thin cyan-blue rays radiating from the cluster surface,
    each wrapped in a dark outline shell for visibility on white. Uses its own
    RNG (seeded independently) so molecule placement is untouched."""
    rng = random.Random(seed)
    origin_v = Vector(origin)
    mat = create_emission_material('ScatterRayMat', color, emission_strength)
    outline_mat = create_outline_material('ScatterRayOutlineMat', outline_color, outline_emission_strength)

    if ray_count <= 1:
        base_angles = [(angle_min_deg + angle_max_deg) / 2.0]
        angle_step  = angle_max_deg - angle_min_deg
    else:
        angle_step  = (angle_max_deg - angle_min_deg) / (ray_count - 1)
        base_angles = [angle_min_deg + i * angle_step for i in range(ray_count)]
    jitter = abs(angle_step) * 0.2

    rays = []
    for i, base_angle in enumerate(base_angles):
        angle_rad = math.radians(base_angle + rng.uniform(-jitter, jitter))
        length    = rng.uniform(length_min, length_max)
        y_offset  = rng.uniform(-0.05, 0.05)   # slight out-of-plane variation
        dir_v = Vector((math.cos(angle_rad), y_offset, math.sin(angle_rad))).normalized()
        end_v = origin_v + dir_v * length
        rays.append(create_outlined_arrow_between(
            origin_v, end_v, core_radius, mat, arrow_length, arrow_radius,
            outline_mat, outline_radius_scale, name=f'ScatterRay_{i}'))
    return rays


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(MASTER_SEED)

    print('── Clearing scene ─────────────────────────────────────────')
    clear_scene()

    print('── Building materials ─────────────────────────────────────')
    mats = build_material_palette()

    central_center = (0.0, 0.0, 0.0)   # exact cavity centre

    # ── Central red cluster ────────────────────────────────────────────────
    print('── Central cluster ────────────────────────────────────────')
    rng_central = random.Random(MASTER_SEED)
    create_cluster(
        center         = central_center,
        num_spheres    = CENTRAL_SPHERE_COUNT,
        base_radius    = CENTRAL_BASE_RADIUS,
        radius_var     = 0.20,
        material       = mats['red'],
        cluster_radius = CENTRAL_CLUSTER_R,
        subdivisions   = 2,      # smooth — in-focus hero objects
        rng            = rng_central,
    )

    # ── Surrounding clusters — sparse, diffuse, far from beam ─────────────
    print('── Surrounding clusters ───────────────────────────────────')
    rng_surr = random.Random(MASTER_SEED + 7)
    for (n, color_key, ox, oy, oz) in SURROUNDING_CLUSTERS:
        center = (central_center[0] + ox,
                  central_center[1] + oy,
                  central_center[2] + oz)
        br = rng_surr.uniform(0.16, 0.22)     # smaller than central cluster
        cr = br * rng_surr.uniform(5.0, 8.0)  # larger relative spread → diffuse
        create_cluster(
            center         = center,
            num_spheres    = n,
            base_radius    = br,
            radius_var     = 0.24,
            material       = mats[color_key],
            cluster_radius = cr,
            subdivisions   = 2,
            rng            = rng_surr,
        )

    # ── Isolated monomers ─────────────────────────────────────────────────
    print('── Isolated particles ─────────────────────────────────────')
    rng_mono  = random.Random(MASTER_SEED + 13)
    mono_keys = ['red_weak', 'pale_blue', 'light_gray']
    for _ in range(ISOLATED_COUNT):
        ox = rng_mono.uniform(-5.5,  5.5)
        oy = rng_mono.uniform(-4.5,  4.5)
        oz = rng_mono.uniform(-3.5,  3.5)
        r  = rng_mono.uniform(0.12,  0.26)
        mk = rng_mono.choice(mono_keys)
        create_sphere(
            (central_center[0]+ox, central_center[1]+oy, central_center[2]+oz),
            r, mats[mk], subdivisions=2,
        )

    # ── Cavity mirrors ─────────────────────────────────────────────────────
    print('── Cavity mirrors ─────────────────────────────────────────')
    create_cavity_mirrors(
        gold_mat    = mats['gold_mirror'],
        half_height = CAVITY_HALF_HEIGHT,
        width       = MIRROR_WIDTH,
        depth       = MIRROR_DEPTH,
        thickness   = MIRROR_THICKNESS,
        bevel       = MIRROR_BEVEL,
    )

    # ── Cavity mode glow ───────────────────────────────────────────────────
    if CAVITY_GLOW:
        print('── Cavity glow ────────────────────────────────────────────')
        glow_height = (CAVITY_HALF_HEIGHT - MIRROR_THICKNESS * 0.5) * 1.95
        create_cavity_glow(
            center    = central_center,
            radius_xy = GLOW_RADIUS,
            height    = glow_height,
            color     = GLOW_COLOR,
            strength  = GLOW_STRENGTH,
            density   = GLOW_DENSITY,
        )

    # ── Camera ─────────────────────────────────────────────────────────────
    print('── Camera ─────────────────────────────────────────────────')
    setup_camera(
        cam_loc  = CAM_LOCATION,
        target   = CAM_TARGET,
        focal_mm = CAM_FOCAL_MM,
        fstop    = CAM_FSTOP,
    )

    # ── Lighting ───────────────────────────────────────────────────────────
    print('── Lighting ───────────────────────────────────────────────')
    setup_lighting()

    # ── World ──────────────────────────────────────────────────────────────
    print('── World ──────────────────────────────────────────────────')
    setup_world()

    # ── Optical annotations (additive only) ─────────────────────────────────
    if SHOW_OPTICAL_ANNOTATIONS:
        print('── Optical annotations ──────────────────────────────────')

        add_incident_beam(
            start                    = BEAM_START,
            end                      = BEAM_END,
            core_radius              = BEAM_RADIUS,
            arrow_length             = BEAM_ARROW_LENGTH,
            arrow_radius             = BEAM_ARROW_RADIUS,
            color                    = BEAM_COLOR,
            emission_strength        = BEAM_EMISSION_STRENGTH,
            glow_radius_scale        = BEAM_GLOW_RADIUS_SCALE,
            glow_emission_strength   = BEAM_GLOW_EMISSION_STRENGTH,
            glow_alpha               = BEAM_GLOW_ALPHA,
            outline_color            = OUTLINE_COLOR,
            outline_emission_strength = OUTLINE_EMISSION_STRENGTH,
            outline_radius_scale     = BEAM_OUTLINE_RADIUS_SCALE,
        )

        add_scattering_rays(
            origin             = SCATTER_ORIGIN,
            ray_count          = SCATTER_RAY_COUNT,
            angle_min_deg      = SCATTER_ANGLE_MIN_DEG,
            angle_max_deg      = SCATTER_ANGLE_MAX_DEG,
            length_min         = SCATTER_LENGTH_MIN,
            length_max         = SCATTER_LENGTH_MAX,
            core_radius        = SCATTER_RADIUS,
            arrow_length       = SCATTER_ARROW_LENGTH,
            arrow_radius       = SCATTER_ARROW_RADIUS,
            color              = SCATTER_COLOR,
            emission_strength  = SCATTER_EMISSION_STRENGTH,
            seed               = SCATTER_SEED,
            outline_color            = OUTLINE_COLOR,
            outline_emission_strength = OUTLINE_EMISSION_STRENGTH,
            outline_radius_scale     = SCATTER_OUTLINE_RADIUS_SCALE,
        )

    # ── Render ─────────────────────────────────────────────────────────────
    print('── Render settings ────────────────────────────────────────')
    setup_render(OUTPUT_PATH, RENDER_QUALITY)

    print(f'── Rendering [{RENDER_QUALITY}] ────────────────────────────────')
    bpy.ops.render.render(write_still=True)
    print(f'\n  Saved: {OUTPUT_PATH}')


main()
