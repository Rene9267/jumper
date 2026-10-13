"""Build Calappa-style visual-only overlays in the ORIGINAL mesh frames of the
visual STLs (no compiled/re-exported meshes -> no double transform)."""
import numpy as np, trimesh, re, os, sys
J = sys.argv[1]  # assets/jumper dir
V = os.path.join(J, 'urdf/jumper/meshes/visual')
OUT = os.path.join(J, 'meshes_calappa'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

def top_z(mesh, xy, zstart=1.0):
    o = np.c_[xy, np.full(len(xy), zstart)]; d = np.tile([0, 0, -1.0], (len(xy), 1))
    loc, idx, _ = mesh.ray.intersects_location(o, d, multiple_hits=False)
    z = np.full(len(xy), np.nan); z[idx] = loc[:, 2]; return z

def sheet(Z, X, Y, thick):
    """closed thin shell from a height grid (nan = outside)."""
    ny, nx = Z.shape; ok = ~np.isnan(Z)
    vid = -np.ones((ny, nx), int); vs = []
    for j in range(ny):
        for i in range(nx):
            if ok[j, i]: vid[j, i] = len(vs); vs.append([X[j, i], Y[j, i], Z[j, i]])
    vs = np.array(vs); n = len(vs); faces = []
    for j in range(ny-1):
        for i in range(nx-1):
            a,b,c,d = vid[j,i],vid[j,i+1],vid[j+1,i+1],vid[j+1,i]
            if min(a,b,c,d) >= 0: faces += [[a,b,c],[a,c,d]]
    faces = np.array(faces)
    top = trimesh.Trimesh(vs, faces, process=False)
    bot = vs.copy(); bot[:, 2] -= thick
    allv = np.vstack([vs, bot]); f = [faces, faces[:, ::-1] + n]
    # stitch boundary
    e = trimesh.Trimesh(vs, faces, process=False).edges_sorted
    u, c = np.unique(e, axis=0, return_counts=True)
    for a, b in u[c == 1]:
        f.append(np.array([[a, b, b+n], [a, b+n, a+n]]))
    m = trimesh.Trimesh(allv, np.vstack(f), process=True); m.fix_normals(); return m

def bumps(X, Y, n, r, h, mask_fn):
    cx = rng.uniform(X.min(), X.max(), n*3); cy = rng.uniform(Y.min(), Y.max(), n*3)
    keep = mask_fn(cx, cy); cx, cy = cx[keep][:n], cy[keep][:n]
    B = np.zeros_like(X)
    for a, b in zip(cx, cy):
        rr = r * rng.uniform(0.7, 1.3)
        B = np.maximum(B, h*rng.uniform(.6,1)*np.exp(-((X-a)**2+(Y-b)**2)/(2*rr**2)))
    return B

# ---------- carapace dome + lateral wings (v2, polar grid -> smooth rim) ----------
shell = trimesh.load(os.path.join(V, 'upper_shell_link.stl'))
def sstep(x): x=np.clip(x,0,1); return x*x*(3-2*x)
cx0, a, b = -0.021, 0.080, 0.084
RIM = 1.10
WING_AMP = 0.0  # set 1.0 to restore
def wingf(x, y):
    w = np.zeros_like(x)
    for s in (1, -1):
        t = np.clip(((x+0.045)/0.050)**2 + ((s*y-0.090)/0.031)**2, 0, 1)
        w = np.maximum(w, sstep(1-t))
    return WING_AMP * w * (np.abs(y) > 0.05)
def inside(x, y):
    return (((x-cx0)/a)**2 + (y/b)**2 <= RIM) | (wingf(x, y) > 0.02)
# boundary radius per angle (fine ray march), then lightly smoothed
NT, NR = 360, 70
th = np.linspace(0, 2*np.pi, NT, endpoint=False)
rr_ = np.linspace(0, 0.16, 1600)
bnd = np.array([rr_[np.nonzero(inside(cx0+rr_*np.cos(t), rr_*np.sin(t)))[0].max()] for t in th])
for _ in range(3): bnd = (np.roll(bnd,1)+2*bnd+np.roll(bnd,-1))/4
rho = np.linspace(0, 1, NR)[1:]**0.85
X = cx0 + np.outer(rho, bnd*np.cos(th)); Y = np.outer(rho, bnd*np.sin(th))
zs = top_z(shell, np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
r = ((X-cx0)/a)**2 + (Y/b)**2
dome = 0.026 + 0.022*np.sqrt(np.clip(1-r, 0, 1))
base_ = np.where(np.isnan(zs), 0.0, zs)
Z = np.where(r <= 1, np.fmax(base_ + 0.0016, dome),
             base_ + 0.0008 + (0.026 - base_ - 0.0008)*sstep((RIM - r)/(RIM-1)))
wz = 0.019 + 0.015*wingf(X, Y)
Z = np.where(wingf(X, Y) > 0.02, np.fmax(Z, wz), Z)
# rounded lip: last ~12% of each ray eases down 3 mm
Z = Z - 0.003*sstep((rho[:, None]-0.88)/0.12)
Z = Z + bumps(X, Y, 190, 0.0047, 0.0032, lambda x, y: ((x-cx0)/a)**2+(y/b)**2 < 0.9)
def polar_sheet(Z, X, Y, thick, zc):
    """seamless polar shell: rings wrap modulo NT, centre is one fan vertex."""
    nr, nt = Z.shape
    top = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    top = np.vstack([top, [cx0, 0.0, zc]]); C = len(top) - 1
    idx = lambda i, j: i * nt + (j % nt)
    f = []
    for j in range(nt):
        f.append([C, idx(0, j), idx(0, j+1)])
        for i in range(nr-1):
            a, b, c, d = idx(i, j), idx(i, j+1), idx(i+1, j+1), idx(i+1, j)
            f += [[a, d, c], [a, c, b]]
    f = np.array(f); n = len(top)
    bot = top.copy(); bot[:, 2] -= thick
    F = [f, f[:, ::-1] + n]
    for j in range(nt):
        a, b = idx(nr-1, j), idx(nr-1, j+1)
        F.append(np.array([[a, b+n, b], [a, a+n, b+n]]))
    m = trimesh.Trimesh(np.vstack([top, bot]), np.vstack(F), process=False); m.fix_normals(); return m
zc = float(Z[0].mean())
carapace = polar_sheet(Z, X, Y, 0.0015, zc)
carapace.export(os.path.join(OUT, 'calappa_carapace.stl'))
# dorsal-plane texcoords from the repo skin atlas so the round spots continue on the overlay
sys.path[:0] = [os.path.join(J, 'tools'), os.path.join(J, '../../rl'), os.path.join(J, '../..')]
import mujoco, build_skin as bs
_bm = mujoco.MjModel.from_xml_path(os.path.join(J, 'jumper_crab.xml'))
_sv, _ = bs._shell_stl(_bm); _lo, _, _uvs = bs.shell_frame(_sv)
_P = carapace.vertices.copy()
for k in (0, 1):   # overhang (wings) -> mirror back into the painted cell so spots continue
    lo_, hi_ = _sv[:, k].min() + 0.004, _sv[:, k].max() - 0.004
    _P[:, k] = np.where(_P[:, k] > hi_, 2*hi_ - _P[:, k], _P[:, k])
    _P[:, k] = np.where(_P[:, k] < lo_, 2*lo_ - _P[:, k], _P[:, k])
CUV = bs._atlas_uv(_P, np.full(len(carapace.vertices), 2), _lo, _uvs)
# ---------- claw-top granules (inside palm footprint only) ----------
for side in ('LF', 'RF'):
    palm = trimesh.load(os.path.join(V, f'{side}_palm_link.stl'))
    lo, hi = palm.bounds; sgn = 1 if side == 'LF' else -1
    parts = []
    pts = []
    while len(pts) < 22:
        z = rng.uniform(lo[2]+0.009, hi[2]-0.009); y = sgn*rng.uniform(0.034, 0.111)
        loc, idx, _ = palm.ray.intersects_location([[1.0, y, z]], [[-1.0, 0, 0]], multiple_hits=False)
        if len(loc): pts.append(loc[0])
    for x, y, z in pts:
        rr = rng.uniform(0.0020, 0.0030)
        s_ = trimesh.creation.icosphere(1, radius=rr); s_.apply_scale([0.7, 1, 1])
        s_.apply_translation([x - rr*0.55, y, z]); parts.append(s_)
    g = trimesh.util.concatenate(parts)
    g.export(os.path.join(OUT, f'{side}_claw_granules.stl'))
    gb = g.bounds
    print(side, 'palm xy', lo[:2].round(4), hi[:2].round(4), 'granules xy', gb[0][:2].round(4), gb[1][:2].round(4),
          'inside:', bool((gb[0][1:] >= lo[1:]).all() and (gb[1][1:] <= hi[1:]).all()))

# ---------- variant XML ----------
src = open(os.path.join(J, 'jumper_crab.xml')).read()
assets = '''    <mesh name="LF_claw_granules" content_type="model/stl" file="meshes_calappa/LF_claw_granules.stl"/>
    <mesh name="RF_claw_granules" content_type="model/stl" file="meshes_calappa/RF_claw_granules.stl"/>
'''
inl = '    <mesh name="calappa_carapace" vertex="%s" face="%s" texcoord="%s"/>\n' % (' '.join('%.6g'%v for v in carapace.vertices.ravel()), ' '.join(str(int(f)) for f in carapace.faces.ravel()), ' '.join('%.6g'%v for v in CUV.ravel()))
src = src.replace('  </asset>', inl + assets + '  </asset>', 1)
def add_after(geomname, new):
    global src
    m = re.search(r'(\s*)(<geom name="%s"[^>]*/>)' % geomname, src); assert m, geomname
    src = src[:m.end()] + m.group(1) + new + src[m.end():]
add_after('upper_shell_link_visual', '<geom name="calappa_carapace_visual" class="visual" type="mesh" contype="0" conaffinity="0" density="0" mass="0" material="crab_shell" rgba="1 1 1 1" mesh="calappa_carapace"/>')
for s in ('LF', 'RF'):
    add_after(f'{s}_palm_link_visual', f'<geom name="{s}_claw_granules_visual" class="visual" type="mesh" contype="0" conaffinity="0" density="0" mass="0" material="crab_claw" rgba="1 0.96 0.97 1" mesh="{s}_claw_granules"/>')
open(os.path.join(J, 'jumper_crab_mesh.xml'), 'w').write(src)
print('carapace faces', len(carapace.faces), 'watertight', carapace.is_watertight, 'bounds', carapace.bounds.round(4).tolist())
