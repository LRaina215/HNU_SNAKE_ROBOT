"""Fixed-length release keyframes. Pure geometry, no robot controller or physics."""
from pathlib import Path
import math
import json
import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
L, R, P = .0645, .0915, .180
ALPHA = P/math.hypot(P, 2*math.pi*R)
BETA = math.sqrt(1-ALPHA**2)
INTAKE = 1.5*L
GAMMA = math.asin(ALPHA)
RHO = INTAKE/GAMMA
E = np.array([1.1-R, .20, 1.])
A = E-np.array([0., RHO*(1-math.cos(GAMMA)), RHO*math.sin(GAMMA)])
Q = np.array([.88, R, 1.])
UP = np.array([0., 0., 1.])
GRID = np.linspace(0., 1., 2401)

def source_curve(u):
    u = np.atleast_1d(u)
    phi = BETA*u/R
    return np.column_stack([Q[0]+ALPHA*u, R*np.cos(phi), 1.+R*np.sin(phi)])

def source_tangent(u):
    phi = BETA*u/R
    return np.array([ALPHA, -BETA*math.sin(phi), BETA*math.cos(phi)])

def receiving(v):
    v = np.atleast_1d(v)
    out = np.empty((len(v), 3))
    low, arc, high = v < -INTAKE, (v >= -INTAKE) & (v < 0), v >= 0
    out[low] = A+(v[low]+INTAKE)[:, None]*UP
    theta = (v[arc]+INTAKE)/RHO
    out[arc] = A+np.column_stack([np.zeros(len(theta)), RHO*(1-np.cos(theta)), RHO*np.sin(theta)])
    theta = math.pi-BETA*v[high]/R
    out[high] = np.column_stack([1.1+R*np.cos(theta), .20+ALPHA*v[high], 1.+R*np.sin(theta)])
    return out

def connector(start, tangent, length):
    """Hermite centerline with prescribed end tangents, arc-length reparameterized."""
    chord = np.linalg.norm(A-start)
    if chord > length+1e-8:
        raise ValueError('Connector chord exceeds available body length')
    if length-chord < 1e-8:
        # Final vertical route has zero slack, and both tangents are vertical.
        points = start+GRID[:, None]*(A-start)
        return points, GRID*length, 0.
    def raw(h):
        t = GRID[:, None]
        return ((2*t**3-3*t**2+1)*start+(t**3-2*t**2+t)*h*tangent
                +(-2*t**3+3*t**2)*A+(t**3-t**2)*h*UP)
    def arc(h):
        return np.linalg.norm(np.diff(raw(h), axis=0), axis=1).sum()
    h = brentq(lambda h: arc(h)-length, .000001, 4.)
    points = raw(h)
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return points, cumulative, h

def make_peel(q):
    # q releases source material from its outlet backwards toward the tail.
    cut = -6*L-q*6*L
    source_u = cut+6*L
    start = source_curve([source_u])[0]
    free_length = -cut
    points, cumulative, handle = connector(start, source_tangent(source_u), free_length-INTAKE)
    def curve(v):
        v = np.atleast_1d(v)
        out = np.empty((len(v), 3))
        kept, span, entry = v < cut, (v >= cut) & (v < -INTAKE), v >= -INTAKE
        out[kept] = source_curve(v[kept]+6*L)
        out[span] = np.column_stack([np.interp(v[span]-cut, cumulative, points[:, k]) for k in range(3)])
        out[entry] = receiving(v[entry])
        return out
    metadata = dict(q=q, source_rods_retained=6*(1-q), released_source_rods=6*q,
        fixed_target_rods=9, free_rear_length_mm=free_length*1000,
        connector_length_mm=float(cumulative[-1]*1000), intake_length_mm=INTAKE*1000,
        retained_source_length_mm=6*L*(1-q)*1000, cut_point_mm=(start*1000).tolist(),
        minimum_connector_height_mm=float(points[:,2].min()*1000), handle_mm=handle*1000)
    return curve, metadata

def make_unfold(q):
    # Tail first exits radially and goes around the source underside. A diagonal
    # from the old tail straight to the side gate would cut across the source pipe.
    start0 = source_curve([-6*L])[0]
    end = A-(12*L-INTAKE)*UP
    gate = np.array([.91, .15, .48])
    theta0 = math.atan2(start0[2]-1.,start0[1])
    smooth = lambda z: z*z*(3-2*z)
    if q <= .15:
        radius = R+smooth(q/.15)*(.130-R)
        start = np.array([start0[0],radius*math.cos(theta0),1.+radius*math.sin(theta0)])
    elif q <= .40:
        blend = smooth((q-.15)/.25)
        theta = theta0+blend*(1.5*math.pi-theta0)
        start = np.array([start0[0]+blend*(.80-start0[0]),.130*math.cos(theta),1.+.130*math.sin(theta)])
    elif q <= .70:
        blend = smooth((q-.40)/.30)
        start = np.array([.80,0.,.87])+blend*(gate-np.array([.80,0.,.87]))
    else:
        blend = smooth((q-.70)/.30)
        start = gate+blend*(end-gate)
    tangent0 = source_tangent(-6*L)
    tangent = (1-q)*tangent0+q*UP
    tangent /= np.linalg.norm(tangent)
    points, cumulative, handle = connector(start, tangent, 12*L-INTAKE)
    bow = 0.
    if .40 < q < 1.:
        # Keep the free-tail bend broad: trade the long Hermite handles for a
        # lateral bow below the pipes, instead of making a tiny fold at the tail.
        blend = smooth(min(1.,(q-.40)/.15))
        handle += blend*(min(handle,np.linalg.norm(A-start))-handle)
        t = GRID[:,None]
        base = ((2*t**3-3*t**2+1)*start+(t**3-2*t**2+t)*handle*tangent
                +(-2*t**3+3*t**2)*A+(t**3-t**2)*handle*UP)
        weight = 16*t*t*(1-t)*(1-t)
        def bent(b):
            return base+weight*b*np.array([-1.,0.,0.])
        def arc(b):
            return np.linalg.norm(np.diff(bent(b),axis=0),axis=1).sum()
        if arc(0.) < 12*L-INTAKE-1e-10:
            bow = brentq(lambda b:arc(b)-(12*L-INTAKE),0.,2.)
        points = bent(bow)
        cumulative = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    def curve(v):
        v = np.atleast_1d(v)
        out = receiving(v)
        span = v < -INTAKE
        out[span] = np.column_stack([np.interp(v[span]+12*L, cumulative, points[:, k]) for k in range(3)])
        return out
    metadata = dict(q=q, source_rods_retained=0, fixed_target_rods=9,
        free_rear_length_mm=12*L*1000, connector_length_mm=float(cumulative[-1]*1000),
        intake_length_mm=INTAKE*1000, tail_point_mm=(start*1000).tolist(),
        minimum_connector_height_mm=float(points[:,2].min()*1000), handle_mm=handle*1000,
        lateral_bow_mm=bow*1000)
    return curve, metadata

initial_curve, INITIAL_METADATA = make_peel(0.)

