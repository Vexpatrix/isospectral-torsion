"""Self-contained verifier: two connected isospectral polygons with different torsional rigidity.

Requires Python 3 and the accompanying compressed JSON witness of integer nodal numerators.
Every verification step uses the standard library and integer or exact rational arithmetic.

Tile T: triangle with vertices V0 = (0,0), V1 = (1,0), V2 = (3/10, 4/5).
Mirror sides: a = V1V2, b = V2V0, c = V0V1.  Copy j of the tile is placed by the isometry reached
from copy 1 along the gluing tree; crossing mirror x multiplies by the reflection of T in side x.
Gluing involutions on copies 1..7 (cycles; fixed copies keep that side on the boundary):
    twin A:  a = (1 4)(3 6),  b = (2 4)(3 5),  c = (2 3)(6 7)
    twin B:  a = (1 4)(3 6),  b = (2 4)(3 5),  c = (1 3)(5 7)

Checks
 1. Exact geometry: each twin is a union of seven triangles with pairwise disjoint interiors, two
    triangles share a segment exactly when they are glued, and the union is connected.  The
    boundary polygon is printed with exact rational vertices.
 2. Transplantation: integer 7x7 matrices T_D, T_N with T X^B_x = X^A_x T for all mirrors x, where
    X_x has 1 at glued pairs and -1 (Dirichlet) or +1 (Neumann) at fixed copies; det != 0.
    By the transplantation theorem (Buser-Conway-Doyle-Semmler) the twins are Dirichlet- and
    Neumann-isospectral.
 3. Torsional rigidity tau = int v, -Lap v = 1 in the polygon, v = 0 on its boundary:
       lower  tau >= 2 int u - int |grad u|^2        any u in H^1_0 (here continuous P1)
       upper  tau <= int |sigma|^2                    any sigma with div sigma = -1
              sigma = -(x - c)/2 + curl psi, psi continuous P1 (normal component continuous)
    evaluated exactly; prints a certified separation of tau(A) and tau(B).

Usage: python verify_torsion_twins.py [K] [--witness PATH]
The mesh has K^2 sub-triangles per tile. The default witness uses K = 64.
"""
import argparse
import gzip
import itertools
import json
from collections import Counter, defaultdict
from fractions import Fraction as Fr
from pathlib import Path

V = [(Fr(0), Fr(0)), (Fr(1), Fr(0)), (Fr(3, 10), Fr(4, 5))]
SIDES = [(1, 2), (2, 0), (0, 1)]                                  # mirror x = side opposite V_x
FLUX_CENTRE = (Fr(1, 2), Fr(1, 2))
DENOMINATOR_POWER = 50


def perm(cycles):
    p = list(range(7))
    for i, j in cycles:
        p[i - 1], p[j - 1] = j - 1, i - 1
    return p


TWINS = {
    "A": [perm([(1, 4), (3, 6)]), perm([(2, 4), (3, 5)]), perm([(2, 3), (6, 7)])],
    "B": [perm([(1, 4), (3, 6)]), perm([(2, 4), (3, 5)]), perm([(1, 3), (5, 7)])],
}

DIRICHLET_TRANSPLANT = [
    [0, 1, 0, -1, 0, 1, 0],
    [1, 0, 0, -1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1],
    [-1, -1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, -1],
    [1, 0, 0, 0, 0, 1, -1],
    [0, 0, 1, 0, -1, -1, 0],
]


# ------------------------------------------------------------------ exact plane geometry
def reflection(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    n2 = dx * dx + dy * dy
    R = ((2 * dx * dx / n2 - 1, 2 * dx * dy / n2), (2 * dx * dy / n2, 2 * dy * dy / n2 - 1))
    return R, (p[0] - R[0][0] * p[0] - R[0][1] * p[1], p[1] - R[1][0] * p[0] - R[1][1] * p[1])


def act(g, p):
    R, t = g
    return (R[0][0] * p[0] + R[0][1] * p[1] + t[0], R[1][0] * p[0] + R[1][1] * p[1] + t[1])


def compose(g, h):
    R, _ = g
    S, s = h
    return (((R[0][0] * S[0][0] + R[0][1] * S[1][0], R[0][0] * S[0][1] + R[0][1] * S[1][1]),
             (R[1][0] * S[0][0] + R[1][1] * S[1][0], R[1][0] * S[0][1] + R[1][1] * S[1][1])), act(g, s))


def placements(perms):
    refl = [reflection(V[a], V[b]) for a, b in SIDES]
    iso = {0: (((Fr(1), Fr(0)), (Fr(0), Fr(1))), (Fr(0), Fr(0)))}
    stack = [0]
    while stack:
        i = stack.pop()
        for x, p in enumerate(perms):
            j = p[i]
            if j != i and j not in iso:
                iso[j] = compose(iso[i], refl[x])
                stack.append(j)
    assert len(iso) == 7, "gluing graph must be connected"
    for x, p in enumerate(perms):
        for i, j in enumerate(p):
            if i != j:
                assert iso[j] == compose(iso[i], refl[x]), "inconsistent glued placement"
    return iso


def separated(A, B):
    """Use SAT with touching projections allowed, so only the interiors must be disjoint."""
    for P in (A, B):
        for e in range(3):
            n = (P[e][1] - P[(e + 1) % 3][1], P[(e + 1) % 3][0] - P[e][0])
            pa = [q[0] * n[0] + q[1] * n[1] for q in A]
            pb = [q[0] * n[0] + q[1] * n[1] for q in B]
            if max(pa) <= min(pb) or max(pb) <= min(pa):
                return True
    return False


def shared_segment(A, B):
    for e, f in itertools.product(range(3), range(3)):
        a0, a1, b0, b1 = A[e], A[(e + 1) % 3], B[f], B[(f + 1) % 3]
        d = (a1[0] - a0[0], a1[1] - a0[1])
        if any(d[0] * (q[1] - a0[1]) - d[1] * (q[0] - a0[0]) for q in (b0, b1)):
            continue
        L = d[0] ** 2 + d[1] ** 2
        s = sorted(((q[0] - a0[0]) * d[0] + (q[1] - a0[1]) * d[1]) / L for q in (b0, b1))
        if min(Fr(1), s[1]) > max(Fr(0), s[0]):
            return True
    return False


def cross(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def on_segment(p, q, r):
    return cross(p, q, r) == 0 and all(min(a, b) <= c <= max(a, b) for a, b, c in zip(p, q, r))


def check_edge_intersections(edges):
    """Check that distinct edges meet only at common endpoints."""
    for (p, q), (r, s) in itertools.combinations(edges, 2):
        common = {p, q} & {r, s}
        for point in (r, s):
            assert not on_segment(p, q, point) or point in common, "overlapping edges or T-junction"
        for point in (p, q):
            assert not on_segment(r, s, point) or point in common, "overlapping edges or T-junction"
        assert not (cross(p, q, r) * cross(p, q, s) < 0 and
                    cross(r, s, p) * cross(r, s, q) < 0), "crossing edges"


def check_corner_multiplicities(triangles, boundary):
    corners = defaultdict(list)
    for triangle in triangles:
        for label, vertex in enumerate(triangle):
            corners[vertex].append(label)
    assert set(corners) == set(boundary), "unexpected interior or missing corner"
    multiplicities = [[], [], []]
    for labels in corners.values():
        label = labels[0]
        assert all(other == label for other in labels), "inconsistent seed corner labels"
        multiplicities[label].append(len(labels))
    assert [sorted(group) for group in multiplicities] == [[1, 2, 4], [1, 2, 4], [1, 3, 3]]


def check_geometry(perms, iso):
    T = [[act(iso[i], v) for v in V] for i in range(7)]
    glued = {frozenset((i, p[i])) for p in perms for i in range(7) if p[i] != i}
    for i, j in itertools.combinations(range(7), 2):
        assert separated(T[i], T[j]), f"triangles {i + 1},{j + 1} overlap"
        assert shared_segment(T[i], T[j]) == (frozenset((i, j)) in glued), f"bad contact {i + 1},{j + 1}"
    tile_edges = {tuple(sorted(edge)) for triangle in T for edge in itertools.combinations(triangle, 2)}
    assert len(tile_edges) == 15, "unexpected number of tile edges"
    check_edge_intersections(tile_edges)
    # boundary: sides of fixed copies; chain them into a closed polygon
    edges = []
    for i in range(7):
        for x, (a, b) in enumerate(SIDES):
            if perms[x][i] == i:
                edges.append((T[i][a], T[i][b]))
    nxt = {}
    for p, q in edges:
        nxt.setdefault(p, []).append(q)
        nxt.setdefault(q, []).append(p)
    assert all(len(v) == 2 for v in nxt.values()), "boundary is not a single simple closed curve"
    start = edges[0][0]
    poly, prev, cur = [start], start, edges[0][1]
    while cur != start:
        assert cur not in poly, "boundary repeats a vertex"
        poly.append(cur)
        a, b = nxt[cur]
        prev, cur = cur, (b if a == prev else a)
    assert len(poly) == len(edges) == 9, "boundary is not a single nine-edge cycle"
    # merge collinear consecutive vertices
    out = []
    for i, p in enumerate(poly):
        q0, q1 = poly[i - 1], poly[(i + 1) % len(poly)]
        if cross(q0, p, q1) != 0:
            out.append(p)
    assert len(out) == 9, "boundary has fewer than nine corners"
    twice_area = abs(sum(p[0] * q[1] - p[1] * q[0] for p, q in zip(out, out[1:] + out[:1])))
    assert twice_area == Fr(28, 5), "unexpected polygon area"
    assert [sum(i == j for i, j in enumerate(p)) for p in perms] == [3, 3, 3]
    check_corner_multiplicities(T, out)
    return T, out


# ------------------------------------------------------------------ exact transplantation
def gluing_matrix(p, s):
    return [[(s if i == j else 1) if p[i] == j else 0 for j in range(7)] for i in range(7)]


def det(M):
    M = [[Fr(v) for v in r] for r in M]
    d = Fr(1)
    for c in range(len(M)):
        q = next((i for i in range(c, len(M)) if M[i][c] != 0), None)
        if q is None:
            return Fr(0)
        if q != c:
            M[c], M[q] = M[q], M[c]
            d = -d
        d *= M[c][c]
        for i in range(c + 1, len(M)):
            f = M[i][c] / M[c][c]
            M[i] = [a - f * b for a, b in zip(M[i], M[c])]
    return d


def matmul(P, Q):
    return [[sum(P[i][k] * Q[k][j] for k in range(7)) for j in range(7)] for i in range(7)]


def verify_transplant(T, s):
    XA = [gluing_matrix(p, s) for p in TWINS["A"]]
    XB = [gluing_matrix(p, s) for p in TWINS["B"]]
    assert all(matmul(T, B) == matmul(A, T) for A, B in zip(XA, XB)), "intertwining identity fails"
    determinant = det(T)
    assert determinant == -24, "unexpected transplant determinant"
    return determinant


# ------------------------------------------------------------------ torsion certificate
def mesh(perms, iso, k):
    key, X, bnd, el = {}, [], [], []
    expected_boundary = set()
    for i in range(7):
        fixed = [perms[x][i] == i for x in range(3)]
        loc = {}
        for a in range(k + 1):
            for b in range(k + 1 - a):
                p = act(iso[i], (V[0][0] + Fr(a, k) * (V[1][0] - V[0][0]) + Fr(b, k) * (V[2][0] - V[0][0]),
                                 V[0][1] + Fr(a, k) * (V[1][1] - V[0][1]) + Fr(b, k) * (V[2][1] - V[0][1])))
                on = (a + b == k, a == 0, b == 0)
                isb = any(on[x] and fixed[x] for x in range(3))
                if p not in key:
                    key[p] = len(X)
                    X.append(p)
                    bnd.append(isb)
                else:
                    bnd[key[p]] = bnd[key[p]] or isb
                loc[(a, b)] = key[p]
        sides = ([loc[(k - j, j)] for j in range(k + 1)],
                 [loc[(0, j)] for j in range(k + 1)],
                 [loc[(j, 0)] for j in range(k + 1)])
        for is_fixed, side in zip(fixed, sides):
            if is_fixed:
                expected_boundary.update(tuple(sorted(edge)) for edge in zip(side, side[1:]))
        for a in range(k):
            for b in range(k - a):
                el.append((loc[(a, b)], loc[(a + 1, b)], loc[(a, b + 1)]))
                if a + b < k - 1:
                    el.append((loc[(a + 1, b)], loc[(a + 1, b + 1)], loc[(a, b + 1)]))
    assert len(el) == 7 * k * k, "wrong number of elements"
    counts = Counter(tuple(sorted(edge)) for element in el for edge in itertools.combinations(element, 2))
    assert set(counts.values()) <= {1, 2}, "an edge occurs in more than two elements"
    mesh_boundary = {edge for edge, count in counts.items() if count == 1}
    assert mesh_boundary == expected_boundary and len(mesh_boundary) == 9 * k, "mesh has cracks or unexpected edges"
    assert {j for edge in mesh_boundary for j in edge} == {j for j, on_boundary in enumerate(bnd) if on_boundary}
    return X, bnd, el


def witness_path(k):
    return Path(__file__).with_name(f"torsion-witness-k{k}.json.gz")


def read_witness(path, k):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or set(data) != {"k", "denominator_power", "twins"}:
        raise ValueError("wrong witness fields")
    if type(data["k"]) is not int or data["k"] != k:
        raise ValueError(f"witness must use k = {k}")
    if type(data["denominator_power"]) is not int or data["denominator_power"] != DENOMINATOR_POWER:
        raise ValueError(f"witness denominator must be 2^{DENOMINATOR_POWER}")
    twins = data["twins"]
    if not isinstance(twins, dict) or set(twins) != set(TWINS):
        raise ValueError("witness must give vectors for A and B")
    return twins


def read_coefficients(data, boundary, name):
    if not isinstance(data, dict) or set(data) != {"u", "psi"}:
        raise ValueError(f"witness for {name} must provide u and psi")
    for label in ("u", "psi"):
        values = data[label]
        if not isinstance(values, list) or len(values) != len(boundary):
            raise ValueError(f"{name}.{label} must have {len(boundary)} nodal values")
        if any(type(value) is not int for value in values):
            raise ValueError(f"{name}.{label} must have integer numerators")
    if any(is_boundary and value != 0 for is_boundary, value in zip(boundary, data["u"])):
        raise ValueError(f"{name}.u must be zero at every boundary vertex")
    denominator = 1 << DENOMINATOR_POWER
    return ([Fr(value, denominator) for value in data["u"]],
            [Fr(value, denominator) for value in data["psi"]])


def certify(X, el, U, S):
    c0 = FLUX_CENTRE
    lo = up = Fr(0)
    for e in el:
        p = [X[j] for j in e]
        b = (p[1][1] - p[2][1], p[2][1] - p[0][1], p[0][1] - p[1][1])
        c = (p[2][0] - p[1][0], p[0][0] - p[2][0], p[1][0] - p[0][0])
        s = b[0] * c[1] - b[1] * c[0]
        A = abs(s) / 2
        uu, ss = [U[j] for j in e], [S[j] for j in e]
        gx = sum(uu[t] * b[t] for t in range(3)) / s
        gy = sum(uu[t] * c[t] for t in range(3)) / s
        lo += 2 * A / 3 * sum(uu) - A * (gx * gx + gy * gy)
        cx = sum(ss[t] * c[t] for t in range(3)) / s             # curl psi = (psi_y, -psi_x)
        cy = -sum(ss[t] * b[t] for t in range(3)) / s
        bx = sum(q[0] for q in p) / 3 - c0[0]
        by = sum(q[1] for q in p) / 3 - c0[1]
        L2 = sum((p[(t + 1) % 3][0] - p[t][0]) ** 2 + (p[(t + 1) % 3][1] - p[t][1]) ** 2 for t in range(3))
        up += (A * (bx * bx + by * by) + A * L2 / 36) / 4 - A * (cx * bx + cy * by) + A * (cx * cx + cy * cy)
    return lo, up


def decimal_bound(value, *, upper=False):
    """Format a rational bound with nine decimal places rounded outward."""
    scale = 10 ** 9
    numerator = value.numerator * scale
    denominator = value.denominator
    units = -((-numerator) // denominator) if upper else numerator // denominator
    integer, fractional = divmod(abs(units), scale)
    return f"{'-' if units < 0 else ''}{integer}.{fractional:09d}"


def main():
    if not __debug__:
        raise RuntimeError("run without -O so that verification checks are enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("k", nargs="?", type=int, default=64)
    parser.add_argument("--witness", type=Path, help="read this gzip-compressed JSON witness")
    args = parser.parse_args()
    k = args.k
    if k < 1:
        parser.error("the mesh subdivision count must be positive")
    try:
        witness = read_witness(args.witness or witness_path(k), k)
    except (OSError, EOFError, ValueError) as error:
        parser.error(str(error))
    bounds = {}
    for name, perms in TWINS.items():
        iso = placements(perms)
        _, poly = check_geometry(perms, iso)
        print(f"twin {name}: seven triangles, disjoint interiors, contacts exactly along glued sides; "
              f"boundary is a simple {len(poly)}-gon of area 14/5:")
        print("   " + ", ".join(f"({p[0]}, {p[1]})" for p in poly))
        X, bnd, el = mesh(perms, iso, k)
        try:
            u, psi = read_coefficients(witness[name], bnd, name)
        except ValueError as error:
            parser.error(str(error))
        bounds[name] = certify(X, el, u, psi)
        lo, up = bounds[name]
        print(f"   {len(el)} mesh elements, {9 * k} boundary edges")
        print(f"   lower bound = {lo}\n   upper bound = {up}")
        print(f"   {decimal_bound(lo)} <= tau({name}) <= {decimal_bound(up, upper=True)}   (outward rounded)")
    D = DIRICHLET_TRANSPLANT
    N = [[abs(value) for value in row] for row in D]
    for label, T, s in (("Dirichlet", D, -1), ("Neumann", N, 1)):
        dT = verify_transplant(T, s)
        print(f"{label} transplantation: exact intertwining identities, det T = {dT}:")
        for row in T:
            print("   ", row)
    (la, ua), (lb, ub) = bounds["A"], bounds["B"]
    if k == 64:
        assert Fr("0.152472053") <= la <= ua <= Fr("0.152748988"), "A fails its claimed enclosure"
        assert Fr("0.152884186") <= lb <= ub <= Fr("0.153162566"), "B fails its claimed enclosure"
    if ua < lb or ub < la:
        first, second, upper, lower = ("A", "B", ua, lb) if ua < lb else ("B", "A", ub, la)
        print(f"CERTIFIED: tau({first}) < tau({second}); gap >= {decimal_bound(lower - upper)}")
    else:
        print(f"The two bounds overlap at k = {k}.")


if __name__ == "__main__":
    main()
