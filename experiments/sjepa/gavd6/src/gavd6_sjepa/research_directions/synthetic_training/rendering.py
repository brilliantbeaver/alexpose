"""Textured, camera-labelled SMPL-H images for the synthetic teaching study.

The renderer needs licensed compatible UV/body assets and actual texture and
background images. It never substitutes the preservation study's coloured mesh
for a photographic training pipeline. Appearance realism still needs empirical
validation on the real early-reference panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


KEYPOINT_NAMES = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
COCO_INDICES = np.arange(5, 17, dtype=np.int64)
SMPL_INDICES = np.array([16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8])


@dataclass(frozen=True)
class RenderRecipe:
    """A camera and corruption setting, held constant throughout one clip."""

    name: str
    azimuth_deg: float | None = None
    person_height_fraction: float = 0.65
    blur_px: float = 0.0
    occlusion_fraction: float = 0.0


LESSON_RECIPES = (
    RenderRecipe("front", 0.0),
    RenderRecipe("side", 90.0),
    RenderRecipe("oblique", 45.0),
    RenderRecipe("low_resolution", None, 0.25),
    RenderRecipe("blur", None, 0.65, 1.2),
    RenderRecipe("occlusion", None, 0.65, 0.0, 0.20),
    RenderRecipe("low_resolution_occlusion", None, 0.25, 0.0, 0.20),
    RenderRecipe("balanced", None, 0.45, 0.6, 0.10),
)


def source_domains() -> tuple[RenderRecipe, ...]:
    """Twenty-four shared settings, with separately held people for validation.

    These defaults do not claim an additional holdout of domain combinations.
    """
    return tuple(
        RenderRecipe(f"{view}_{resolution}_{occlusion}_{blur}", angle, fraction, sigma, cover)
        for view, angle in (("front", 0.0), ("oblique", 45.0), ("side", 90.0))
        for resolution, fraction in (("low", 0.25), ("high", 0.65))
        for occlusion, cover in (("clear", 0.0), ("occluded", 0.20))
        for blur, sigma in (("sharp", 0.0), ("blur", 1.2))
    )


def camera_pose(joints: np.ndarray, azimuth_deg: float, distance: float) -> np.ndarray:
    """Return OpenGL camera-to-world pose, with +Y up and view along local -Z.

    Azimuth is relative to the initial pelvis lateral axis, not an arbitrary
    AMASS world heading. Perspective projection below uses the identical pose.
    """
    target = np.median(joints[:, 0], axis=0).astype(float)
    right = np.asarray(joints[0, 2] - joints[0, 1], dtype=float)
    right[1] = 0
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("Cannot determine body heading from the pelvis")
    right /= np.linalg.norm(right)
    # SMPL-H's neutral subject-left axis is +X and forward is +Z in this
    # +Y-up frame. Subject-right cross up would reverse front and back.
    forward = np.cross([0.0, 1.0, 0.0], right)
    angle = np.deg2rad(azimuth_deg)
    backward = np.cos(angle) * forward + np.sin(angle) * right
    camera_right = np.cross([0.0, 1.0, 0.0], backward)
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(backward, camera_right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([camera_right, camera_up, backward])
    pose[:3, 3] = target + distance * backward
    return pose


def project_points(points, pose, width, height, yfov):
    """Project with the rendered camera and top-left pixel center at (0, 0).

    OpenGL's first raster sample is at window coordinate (0.5, 0.5), so the
    array-coordinate optical center is ((width-1)/2, (height-1)/2).
    """
    camera = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    depth = -camera[..., 2]
    focal = height / (2 * np.tan(yfov / 2))
    xy = np.stack([
        focal * camera[..., 0] / np.maximum(depth, 1e-8) + (width - 1) / 2,
        -focal * camera[..., 1] / np.maximum(depth, 1e-8) + (height - 1) / 2,
    ], axis=-1)
    return xy.astype(np.float32), depth.astype(np.float32)


def load_uv_topology(path, faces):
    """Read a seam-preserving UV asset and reject a different mesh topology.

    NPZ fields are ``uv[Nuv,2], face_uv[F,3], faces[F,3]``. The common SMPL
    spellings ``vt, ft, f`` are also accepted. A triangular OBJ with v/vt/f
    indices can be read directly without reindexing its mesh. UV coordinates
    follow OpenGL's bottom-left convention. Assets are user-supplied.
    """
    path = Path(path).expanduser()
    if path.suffix.lower() == ".obj":
        coords, uv_faces, vertex_faces = [], [], []
        vertex_count = 0
        for line in path.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertex_count += 1
            elif fields[0] == "vt":
                coords.append([float(x) for x in fields[1:3]])
            elif fields[0] == "f":
                if len(fields) != 4 or any("/" not in value or not value.split("/")[1] for value in fields[1:]):
                    raise ValueError("UV template OBJ requires triangular faces with explicit texture indices")
                indices = [[int(part) for part in value.split("/")[:2]] for value in fields[1:]]
                vertex_faces.append([v - 1 if v > 0 else vertex_count + v for v, _ in indices])
                uv_faces.append([u - 1 if u > 0 else len(coords) + u for _, u in indices])
        uv, face_uv, template_faces = np.asarray(coords), np.asarray(uv_faces), np.asarray(vertex_faces)
    else:
        with np.load(path, allow_pickle=False) as asset:
            keys = ("uv", "face_uv", "faces") if "uv" in asset else ("vt", "ft", "f")
            if not set(keys) <= set(asset.files):
                raise ValueError("UV NPZ needs uv/face_uv/faces or vt/ft/f arrays")
            uv, face_uv, template_faces = (np.asarray(asset[key]) for key in keys)
    if not np.array_equal(template_faces, faces):
        raise ValueError("UV template faces must exactly match the licensed SMPL-H model")
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError("UV coordinates must be finite [N,2]")
    if (face_uv.shape != faces.shape or not np.issubdtype(face_uv.dtype, np.integer)
            or face_uv.min() < 0 or face_uv.max() >= len(uv)):
        raise ValueError("Invalid UV triangle indices")
    return uv.astype(np.float32), face_uv.astype(np.int64)


def _image_assets(folder):
    path = Path(folder).expanduser()
    images = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise FileNotFoundError(f"Real texture/background images are required in {path}")
    return images


class TexturedBodyRenderer:
    """Render actual full-body geometry with explicit UV textures and backgrounds.

    Set ``PYOPENGL_PLATFORM=egl`` before Python starts on a headless GPU node.
    Joint-center depth tolerance is only a synthetic visibility approximation.
    It must not be presented as an independent visibility label on real video.
    """

    def __init__(self, uv_path, texture_dir, background_dir, width=512, height=384):
        if not Path(uv_path).expanduser().is_file():
            raise FileNotFoundError(f"A licensed compatible UV topology NPZ or OBJ is required: {uv_path}")
        if min(width, height) < 64:
            raise ValueError("Use a source render resolution of at least 64 pixels per axis")
        self.uv_path = uv_path
        self.textures = _image_assets(texture_dir)
        self.backgrounds = _image_assets(background_dir)
        self.width, self.height = int(width), int(height)
        self._renderer = None

    def close(self):
        """Release the offscreen OpenGL context after a preparation stage."""
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None

    def render(self, body, recipe: RenderRecipe, seed=17):
        """Return RGB, twelve projected joints, visibility, boxes and asset metadata.

        Low resolution is produced by increasing camera distance in the source
        image. Person crops are resized only later by the common estimator path.
        No perfect high-resolution crop is downsampled as a hidden substitute.
        """
        import pyrender
        import trimesh
        from PIL import Image, ImageFilter, ImageOps

        if body.coordinate_system != "y_up":
            raise ValueError("Expected the existing AMASS +Y-up body coordinates")
        rng = np.random.default_rng(seed)
        texture_path = self.textures[int(rng.integers(len(self.textures)))]
        background_path = self.backgrounds[int(rng.integers(len(self.backgrounds)))]
        with Image.open(texture_path) as image:
            texture = image.convert("RGB").copy()
        with Image.open(background_path) as image:
            background = np.asarray(ImageOps.fit(image.convert("RGB"), (self.width, self.height))).copy()
        uv, face_uv = load_uv_topology(self.uv_path, body.faces)
        azimuth = float(recipe.azimuth_deg if recipe.azimuth_deg is not None else rng.choice([0, 45, 90, 135, 180]))
        yfov = np.deg2rad(50.0)
        body_height = float(np.ptp(body.vertices[..., 1]))
        distance = max(2.0, body_height / (2 * np.tan(yfov / 2) * recipe.person_height_fraction))
        pose = camera_pose(body.joints, azimuth, distance)
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=self.width / self.height, znear=0.05)
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35, 0.35, 0.35])
        scene.add(camera, pose=pose)
        light_pose = pose.copy()
        light_pose[:3, 3] += np.array([1.5, 2.0, 0.0])
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=light_pose)
        if self._renderer is None:
            self._renderer = pyrender.OffscreenRenderer(self.width, self.height)
        triangles = np.arange(body.faces.size).reshape(-1, 3)
        expanded_uv = uv[face_uv.reshape(-1)]
        joints, joint_depth = project_points(body.joints[:, SMPL_INDICES], pose, self.width, self.height, yfov)
        projected_vertices, _ = project_points(body.vertices, pose, self.width, self.height, yfov)
        blocker = np.zeros((self.height, self.width), dtype=bool)
        if recipe.occlusion_fraction:
            center_x = int(self.width / 2)
            projected_width = float(np.ptp(projected_vertices[..., 0]))
            half_width = max(1, int(projected_width * recipe.occlusion_fraction))
            top = int(self.height * 0.50)
            blocker[top:, max(0, center_x - half_width):min(self.width, center_x + half_width)] = True
        rgbs, visible, boxes = [], [], []
        for frame, vertices in enumerate(body.vertices):
            base = trimesh.Trimesh(vertices=vertices, faces=body.faces, process=False)
            mesh = trimesh.Trimesh(
                vertices=vertices[body.faces.reshape(-1)], faces=triangles,
                vertex_normals=base.vertex_normals[body.faces.reshape(-1)], process=False,
                visual=trimesh.visual.texture.TextureVisuals(uv=expanded_uv, image=texture),
            )
            node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
            rgba, depth = self._renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            scene.remove_node(node)
            foreground = depth > 0
            if not foreground.any():
                raise ValueError("Camera produced no body pixels; inspect the rendering assets")
            pixels = background.copy()
            pixels[foreground] = rgba[..., :3][foreground]
            ys, xs = np.where(foreground)
            bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
            if recipe.occlusion_fraction:
                # A fixed image-space obstacle occludes the lower central body.
                # It hides pixels, not the supervised 3D geometry used to project labels.
                # Borrow background material, instead of making a label-coloured rectangle.
                pixels[blocker] = np.roll(background, self.width // 3, axis=1)[blocker]
            flags = np.zeros(12, dtype=bool)
            for joint, (x, y) in enumerate(joints[frame]):
                ix, iy = int(round(float(x))), int(round(float(y)))
                if 0 <= ix < self.width and 0 <= iy < self.height and not blocker[iy, ix]:
                    near = depth[max(0, iy - 1):iy + 2, max(0, ix - 1):ix + 2]
                    near = near[near > 0]
                    flags[joint] = bool(len(near) and abs(float(joint_depth[frame, joint] - near.min())) < 0.16)
            if recipe.blur_px:
                pixels = np.asarray(Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(recipe.blur_px)))
            rgbs.append(pixels)
            visible.append(flags)
            boxes.append(bbox)
        return dict(
            images=np.stack(rgbs), keypoints=joints, visible=np.stack(visible), boxes=np.stack(boxes),
            texture_path=str(texture_path), background_path=str(background_path),
            camera_pose=pose, azimuth_deg=azimuth, recipe=recipe.name,
            landmark_convention="projected_smplh_joint_centers_approximate_coco_body12",
            visibility_reference="synthetic_depth_proxy_0.16m_not_real_annotation",
        )
