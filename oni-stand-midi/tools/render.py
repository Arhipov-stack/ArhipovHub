"""Tiny software renderer for previews: z-buffered point splats with clay shading."""
import numpy as np
from PIL import Image
from scipy import ndimage


def _samples(m, color):
    v = m.vertices
    n = m.vertex_normals
    f = m.faces
    # vertices + face centroids + edge midpoints: dense enough for marching-cubes meshes
    pts = [v, v[f].mean(1), (v[f[:, 0]] + v[f[:, 1]]) / 2, (v[f[:, 1]] + v[f[:, 2]]) / 2,
           (v[f[:, 2]] + v[f[:, 0]]) / 2]
    nrm = [n, n[f].mean(1), (n[f[:, 0]] + n[f[:, 1]]) / 2, (n[f[:, 1]] + n[f[:, 2]]) / 2,
           (n[f[:, 2]] + n[f[:, 0]]) / 2]
    P = np.vstack(pts)
    N = np.vstack(nrm)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    col = np.broadcast_to(np.asarray(color, np.float32), (len(P), 3))
    return P, N, col


def render(parts, yaw=30, pitch=15, px=0.35, size=None, path=None, bg=(250, 250, 250)):
    """parts: list of (trimesh, rgb 0..1). yaw around z (deg), pitch looks down."""
    P, N, Cc = zip(*[_samples(m, c) for m, c in parts])
    P, N, Cc = np.vstack(P), np.vstack(N), np.vstack(Cc)
    a, b = np.deg2rad(yaw), np.deg2rad(pitch)
    # camera looks along -y' after rotation; screen x = x', screen up = z'
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), np.sin(b)], [0, -np.sin(b), np.cos(b)]])
    R = Rx @ Rz
    Q = P @ R.T
    Nq = N @ R.T
    lo, hi = Q.min(0), Q.max(0)
    W = int((hi[0] - lo[0]) / px) + 40
    H = int((hi[2] - lo[2]) / px) + 40
    # camera sits at +y' looking back, so its right-hand side is -x'
    u = ((hi[0] - Q[:, 0]) / px + 20).astype(int)
    v = (H - 1 - ((Q[:, 2] - lo[2]) / px + 20)).astype(int)
    depth = Q[:, 1]                       # larger y' = closer to camera
    order = np.argsort(depth)             # far first, near overwrite
    zbuf = np.full((H, W), -np.inf)
    img = np.zeros((H, W, 3), np.float32)
    nb = np.zeros((H, W, 3), np.float32)
    zbuf[v[order], u[order]] = depth[order]
    img[v[order], u[order]] = Cc[order]
    nb[v[order], u[order]] = Nq[order]
    filled = np.isfinite(zbuf)
    # close pin-holes
    hole = ~filled & ndimage.binary_closing(filled, iterations=2)
    if hole.any():
        idx = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
        img[hole] = img[idx[0][hole], idx[1][hole]]
        nb[hole] = nb[idx[0][hole], idx[1][hole]]
        filled = filled | hole
    L1 = np.array([0.45, 0.75, 0.5]); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.6, 0.3, 0.2]); L2 /= np.linalg.norm(L2)
    nn = nb / (np.linalg.norm(nb, axis=2, keepdims=True) + 1e-9)
    nn = np.where(nn[..., 1:2] < 0, -nn, nn)  # backfaces seen through openings
    diff = 0.62 * np.clip(nn @ L1, 0, 1) + 0.18 * np.clip(nn @ L2, 0, 1)
    rim = 0.12 * (1 - np.clip(nn[..., 1], 0, 1)) ** 2
    shade = 0.28 + diff + rim
    out = np.clip(img * shade[..., None], 0, 1) * 255
    bgc = np.array(bg, np.float32)
    out[~filled] = bgc
    im = Image.fromarray(out.astype(np.uint8))
    if size:
        im.thumbnail((size, size), Image.LANCZOS)
    if path:
        im.save(path)
    return im


def sheet(images, path, cols=None, pad=10, bg=(250, 250, 250)):
    cols = cols or len(images)
    rows = (len(images) + cols - 1) // cols
    w = max(i.width for i in images)
    h = max(i.height for i in images)
    S = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        S.paste(im, (pad + c * (w + pad) + (w - im.width) // 2, pad + r * (h + pad) + (h - im.height) // 2))
    S.save(path)
    return S
