"""Small CPU mesh renderer with known material transport for the first pilot.

This rasterizes the actual SMPL-H triangles. It is deliberately dependency-light
and favors interpretable reference flow over cinematic rendering. Measure its
throughput before scaling; render independent clips on separate CPU workers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .body_geometry import BodySequence


@dataclass
class Camera:
    eye: np.ndarray
    rotation: np.ndarray
    width: int = 128
    height: int = 128
    focal_px: float = 150.0

    @classmethod
    def look_at(cls, target=(0, 0.9, 0), eye=None, width=128, height=128,
                focal_px=None, yaw_degrees=25, distance=3.5, elevation_degrees=5):
        target = np.asarray(target, dtype=np.float64)
        if eye is None:
            yaw, elevation = np.deg2rad([yaw_degrees, elevation_degrees])
            eye = target + distance * np.array([np.sin(yaw)*np.cos(elevation),
                                                 np.sin(elevation),np.cos(yaw)*np.cos(elevation)])
        eye = np.asarray(eye, dtype=np.float64)
        forward = target-eye; forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0,1,0]); right /= np.linalg.norm(right)
        up = np.cross(right,forward)
        return cls(eye,np.stack([right,up,forward]),int(width),int(height),
                   float(focal_px if focal_px is not None else 1.15*width))

    def project(self, points):
        points = (np.asarray(points)-self.eye) @ self.rotation.T
        depth = points[...,2]
        denominator = np.where(np.abs(depth)>1e-8,depth,np.nan)
        xy = self.focal_px*points[...,:2]/denominator[...,None]
        xy[...,1] *= -1
        xy += [(self.width-1)/2,(self.height-1)/2]
        return xy.astype(np.float32),depth.astype(np.float32)

    def as_dict(self):
        return dict(eye=self.eye.tolist(),rotation=self.rotation.tolist(),width=self.width,
                    height=self.height,focal_px=self.focal_px)


@dataclass
class RenderedSequence:
    rgb: np.ndarray
    depth: np.ndarray
    joints2d: np.ndarray
    joint_visible: np.ndarray
    flow: np.ndarray
    flow_valid: np.ndarray
    foreground: np.ndarray
    camera: Camera
    metadata: dict


def _rasterize(vertices, faces, camera):
    """Perspective-correct material weights and nearest-triangle depth."""
    xy,z = camera.project(vertices)
    height,width = camera.height,camera.width
    depth = np.full((height,width),np.inf,np.float32)
    triangle_ids = np.full((height,width),-1,np.int32)
    barycentric = np.zeros((height,width,3),np.float32)
    for index,face in enumerate(faces):
        uv, zz = xy[face], z[face]
        if not np.isfinite(uv).all() or np.min(zz)<=0.01:
            continue
        lo=np.maximum(np.ceil(uv.min(axis=0)).astype(int),[0,0])
        hi=np.minimum(np.floor(uv.max(axis=0)).astype(int),[width-1,height-1])
        if np.any(lo>hi):
            continue
        ax,ay=uv[0]; bx,by=uv[1]; cx,cy=uv[2]
        denominator=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(denominator)<1e-8:
            continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        w0=((by-cy)*(xx-cx)+(cx-bx)*(yy-cy))/denominator
        w1=((cy-ay)*(xx-cx)+(ax-cx)*(yy-cy))/denominator
        weights=np.stack([w0,w1,1-w0-w1],axis=-1)
        inside=np.min(weights,axis=-1)>=-1e-6
        perspective=weights/zz
        inverse_depth=np.sum(perspective,axis=-1)
        candidate=1/np.maximum(inverse_depth,1e-12)
        ys,xs=slice(lo[1],hi[1]+1),slice(lo[0],hi[0]+1)
        nearer=inside & (candidate<depth[ys,xs])
        depth[ys,xs][nearer]=candidate[nearer]
        triangle_ids[ys,xs][nearer]=index
        barycentric[ys,xs][nearer]=(perspective/np.maximum(inverse_depth[...,None],1e-12))[nearer]
    return depth,triangle_ids,barycentric


def _vertex_colors(vertices, appearance, seed):
    if appearance=="textureless":
        return np.full((len(vertices),3),[.57,.65,.74],dtype=np.float32)
    # Material colors attach to vertex identities and are shared by paired clips.
    rng=np.random.default_rng(seed)
    direction=rng.normal(size=(3,3))
    phase=vertices @ direction.T * 26 + rng.uniform(0,2*np.pi,size=3)
    return (.50+.27*np.sin(phase)).astype(np.float32)


def _transport_visible(points, depths, vertices, faces, camera, triangle_ids):
    """Conservative subpixel visibility against nearby rasterized triangles.

    Compare depth at the transported point itself. A centimeter-scale tolerance
    against the nearest pixel can incorrectly accept a different, occluding limb.
    At unresolved triangle boundaries, exclude the correspondence from scoring.
    """
    result = np.zeros(len(points), bool)
    inside = (np.isfinite(points).all(axis=1) & np.isfinite(depths) & (depths > .01)
              & (points[:, 0] >= 0) & (points[:, 0] <= camera.width-1)
              & (points[:, 1] >= 0) & (points[:, 1] <= camera.height-1))
    take = np.flatnonzero(inside)
    if not len(take):
        return result
    xy = points[take]
    lower = np.floor(xy).astype(int)
    offsets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    neighbors = np.minimum(lower[:, None] + offsets, [camera.width-1, camera.height-1])
    candidate_ids = triangle_ids[neighbors[..., 1], neighbors[..., 0]]
    projected, vertex_depth = camera.project(vertices)
    candidate_faces = faces[np.maximum(candidate_ids, 0)]
    uv, z = projected[candidate_faces], vertex_depth[candidate_faces]
    a, b, c = uv[:, :, 0], uv[:, :, 1], uv[:, :, 2]
    denominator = (b[..., 1]-c[..., 1])*(a[..., 0]-c[..., 0]) + (c[..., 0]-b[..., 0])*(a[..., 1]-c[..., 1])
    safe = np.where(np.abs(denominator) > 1e-8, denominator, np.nan)
    x, y = xy[:, None, 0], xy[:, None, 1]
    w0 = ((b[..., 1]-c[..., 1])*(x-c[..., 0])+(c[..., 0]-b[..., 0])*(y-c[..., 1])) / safe
    w1 = ((c[..., 1]-a[..., 1])*(x-c[..., 0])+(a[..., 0]-c[..., 0])*(y-c[..., 1])) / safe
    weights = np.stack([w0, w1, 1-w0-w1], axis=-1)
    covers = (candidate_ids >= 0) & (weights.min(axis=-1) >= -1e-6) & (z.min(axis=-1) > .01)
    with np.errstate(divide="ignore", invalid="ignore"):
        surface_depth = 1 / np.sum(weights/z, axis=-1)
    nearest = np.min(np.where(covers, surface_depth, np.inf), axis=1)
    rounded = np.rint(xy).astype(int)
    foreground = triangle_ids[rounded[:, 1], rounded[:, 0]] >= 0
    result[take] = foreground & np.isfinite(nearest) & (np.abs(nearest-depths[take]) < 1e-4)
    return result


def render_sequence(body: BodySequence, camera=None, appearance="textured",
                    occlusion=None, seed=17) -> RenderedSequence:
    """Render RGB, projected joints, and exact mesh material-flow references.

    ``occlusion=(left,top,right,bottom)`` is a fixed image-fraction rectangle in
    front of the person. It changes pixels and visibility, never the event label.
    ``appearance='moving_shadow'`` changes illumination independently of geometric
    motion, providing a flow-confusion stress test. Use identical scene settings
    for the real-event and tracking-failure members of each matched pair.

    Flow scoring retains surface points that pass a conservative raster-based
    visibility check in both frames. Anatomical-joint visibility is an approximate
    depth-neighborhood proxy, not a surface correspondence or pose-confidence
    reference.
    """
    if body.coordinate_system != "y_up":
        raise ValueError("Renderer expects the common +Y-up world frame")
    if appearance not in {"textured","textureless","moving_shadow"}:
        raise ValueError(f"Unknown appearance: {appearance}")
    if camera is None:
        target=np.median(body.joints[:,0],axis=0)
        camera=Camera.look_at(target=target)
    t=len(body.joints); h,w=camera.height,camera.width
    colors=_vertex_colors(body.vertices[0],appearance,seed)
    yy,xx=np.mgrid[:h,:w]
    background=np.stack([.12+.08*yy/max(h-1,1),.17+.05*xx/max(w-1,1),np.full((h,w),.23)],axis=-1)
    rgb=np.empty((t,h,w,3),np.uint8)
    depths,ids,barys=[],[],[]
    blocker=np.zeros((h,w),bool)
    if occlusion is not None:
        left,top,right,bottom=np.asarray(occlusion,dtype=float)
        if not (0<=left<right<=1 and 0<=top<bottom<=1):
            raise ValueError("Occlusion must be (left,top,right,bottom) in [0,1]")
        blocker[int(top*h):int(bottom*h),int(left*w):int(right*w)]=True
    for frame in range(t):
        depth,face_index,bary=_rasterize(body.vertices[frame],body.faces,camera)
        visible=face_index>=0
        pixels=background.copy()
        indices=body.faces[face_index[visible]]
        pixels[visible]=np.sum(colors[indices]*bary[visible,...,None],axis=1)
        if appearance=="moving_shadow":
            shadow=.35+.65*(.5+.5*np.sin(xx*.21-frame*.75))
            pixels[visible]*=shadow[visible,None]
        pixels[blocker]=[.28,.29,.30]
        depth[blocker]=0.1
        face_index[blocker]=-1
        bary[blocker]=0
        rgb[frame]=np.clip(pixels*255,0,255).astype(np.uint8)
        depths.append(depth);ids.append(face_index);barys.append(bary)
    depth=np.stack(depths)
    foreground=np.stack(ids)>=0
    flow=np.zeros((max(t-1,0),h,w,2),np.float32)
    valid=np.zeros((max(t-1,0),h,w),bool)
    for frame in range(t-1):
        source=ids[frame]>=0
        ys,xs=np.where(source)
        faces=body.faces[ids[frame][source]]
        transported=np.sum(body.vertices[frame+1,faces]*barys[frame][source,...,None],axis=1)
        target_xy,target_z=camera.project(transported)
        visible = _transport_visible(target_xy, target_z, body.vertices[frame+1],
                                     body.faces, camera, ids[frame+1])
        flow[frame,ys,xs]=target_xy-np.stack([xs,ys],axis=-1)
        valid[frame,ys,xs]=visible
    q,z=camera.project(body.joints)
    joint_visible=np.zeros(q.shape[:2],bool)
    # Joint centers are inside a limb, so this is intentionally a loose proxy.
    for frame in range(t):
        finite = np.isfinite(q[frame]).all(axis=-1) & (z[frame] > .01)
        coords=np.rint(np.where(finite[:, None], q[frame], -1)).astype(int)
        for joint,(x,y) in enumerate(coords):
            if 0<=x<w and 0<=y<h and not blocker[y,x]:
                neighborhood=depth[frame,max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)]
                d=np.min(neighborhood)
                joint_visible[frame,joint]=np.isfinite(d) and abs(float(z[frame,joint]-d))<.16
    metadata=dict(renderer="perspective_mesh_cpu",reference="model_derived_mesh_material_transport",
                  appearance=appearance,seed=int(seed),occlusion=occlusion,
                  joint_visibility="approximate_anatomical_depth_proxy",
                  flow_visibility="conservative_subpixel_raster_depth",camera=camera.as_dict())
    return RenderedSequence(rgb,depth,q,joint_visible,flow,valid,foreground,camera,metadata)
