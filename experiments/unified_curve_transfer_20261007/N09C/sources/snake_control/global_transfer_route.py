"""Fixed, manually specified L-pipe route, independent of measured body poses.

Coordinates increase from tail towards head.  Source polar angle is about +X;
target atan2(z, x) is about -Y, hence its derivative has the opposite sign.
Only the initial supported source supplies radius, pitch and handedness.
"""
import math

import numpy as np
from scipy.interpolate import CubicSpline, PPoly

LINK = .0645
CORNER = np.array([1.1, 0., 1.])


class FrozenRouteSpec(dict):
    """Numeric plan shared safely by transactional local-curve snapshots."""
    def __init__(self, values):
        contents = dict(values)
        for key in ['spline_x', 'spline_c', 'bridge_controls']:
            contents[key] = np.asarray(contents[key]).copy()
            contents[key].flags.writeable = False
        dict.__init__(self, contents)

    def __deepcopy__(self, memo):
        return self

    def _immutable(self, *args, **kwargs):
        raise TypeError('The manual global route is immutable during execution')

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable


def make_manual_route(initial_nodes, source_node=13, target_node=8):
    p = np.asarray(initial_nodes)
    u0 = (21-np.arange(22))*LINK
    us, ut = float(u0[source_node]), float(u0[target_node])
    ids = np.arange(source_node, 22)
    radial = p[ids, 1:3]-[0., 1.]
    phi = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
    handedness = float(np.sign(np.polyfit(p[ids, 0], phi, 1)[0]))
    radius = float(np.median(np.linalg.norm(radial, axis=1)))
    pitch = float(np.clip(np.polyfit(u0[ids], p[ids, 0], 1)[0], .12, .55))
    rate = handedness*math.sqrt(1-pitch*pitch)/radius
    source = p[source_node].copy()
    source[1:3] = [0., 1.]+radius*radial[0]/np.linalg.norm(radial[0])
    # The manual target endpoint uses known geometry and the initial occupied
    # target sector. It does not inherit the target's initially opposite twist.
    target = p[target_node].copy()
    rr = target[[0, 2]]-CORNER[[0, 2]]
    target[[0, 2]] = CORNER[[0, 2]]+radius*rr/np.linalg.norm(rr)
    target[1] = max(.09, float(target[1]))
    a0 = math.atan2(source[2]-1., source[1])
    a1 = math.atan2(target[2]-1., target[0]-1.1)
    ts = np.array([pitch, -radius*rate*math.sin(a0), radius*rate*math.cos(a0)])
    tt = np.array([-radius*(-rate)*math.sin(a1), pitch, radius*(-rate)*math.cos(a1)])
    # Four manually fixed control points form a tangent-continuous outer
    # bridge. Its shape is frozen once chosen, rather than rebuilt from p(t).
    controls = np.vstack([source, source+.16*ts, target-.16*tt, target])
    source_u = np.arange(-2.2, us, .0025)
    bridge_u = np.linspace(us, ut, 161)
    target_u = np.arange(ut+.0025, u0[0]+3.8, .0025)
    d0, d1 = source_u-us, target_u-ut
    sp = np.column_stack([source[0]+pitch*d0,
        radius*np.cos(a0+rate*d0), 1.+radius*np.sin(a0+rate*d0)])
    t = ((bridge_u-us)/(ut-us))[:, None]
    bp = (1-t)**3*controls[0]+3*(1-t)**2*t*controls[1]+3*(1-t)*t*t*controls[2]+t**3*controls[3]
    tp = np.column_stack([1.1+radius*np.cos(a1-rate*d1),
        target[1]+pitch*d1, 1.+radius*np.sin(a1-rate*d1)])
    u = np.r_[source_u, bridge_u, target_u]
    xyz = np.vstack([sp, bp, tp])
    spline = CubicSpline(u, xyz, axis=0)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    return dict(version=1, origin='manual fixed route from CLOSE1B source winding and known pipe geometry',
        source_bound=us, target_bound=ut, initial_head_parameter=float(u0[0]),
        initial_head_arc=float(np.interp(u0[0], u, arc)),
        spline_x=spline.x.tolist(), spline_c=spline.c.tolist(),
        radius=radius, axial_sine=pitch, source_polar_rate=rate, target_polar_rate=-rate,
        source_right_hand_dphi_daxis=rate/pitch, target_right_hand_dphi_daxis=rate/pitch,
        bridge_controls=controls.tolist(), frozen=True)


def sample_route(spec, u, phase=None, local_pipe_motion=None, source_release=None):
    u = np.asarray(u)
    native=PPoly(np.asarray(spec['spline_c']), np.asarray(spec['spline_x']))
    p=native(u)
    released_controls=None
    if source_release is not None and source_release.get('amplitude',0.)>0.:
        # Keep the posterior helix, progressively flatten circumferential
        # tangent toward +X near its outlet, and integrate from a fixed rear
        # anchor. The outlet and its bridge control tangent travel together.
        strength=source_release['amplitude'];end=spec['source_bound']
        start=end-source_release['length_m']
        grid=np.linspace(start,end,max(33,int(np.ceil((end-start)/.002))+1))
        original=native(grid);derivative=native.derivative()(grid);speed=np.linalg.norm(derivative,axis=1)
        weight=(grid-start)/(end-start);weight=weight*weight*(3-2*weight)
        radial=original[:,1:]-[0.,1.];radius=np.linalg.norm(radial,axis=1)
        radius_rate=np.sum(radial*derivative[:,1:],axis=1)/np.maximum(radius,1e-12)
        angle_rate=(radial[:,0]*derivative[:,2]-radial[:,1]*derivative[:,1])/np.maximum(radius**2,1e-12)
        released_rate=angle_rate*(1-strength*weight)
        angle=np.arctan2(radial[0,1],radial[0,0])+np.r_[0.,np.cumsum(.5*(released_rate[:-1]+released_rate[1:])*np.diff(grid))]
        axial_rate=np.sqrt(np.maximum(1e-12,speed**2-radius_rate**2-(radius*released_rate)**2))
        axial=original[0,0]+np.r_[0.,np.cumsum(.5*(axial_rate[:-1]+axial_rate[1:])*np.diff(grid))]
        points=np.column_stack([axial,radius*np.cos(angle),1.+radius*np.sin(angle)])
        tangential=np.column_stack([axial_rate,
            radius_rate*np.cos(angle)-radius*released_rate*np.sin(angle),
            radius_rate*np.sin(angle)+radius*released_rate*np.cos(angle)])
        released=CubicSpline(grid,points,bc_type=((1,tangential[0]),(1,tangential[-1])))
        inside=(u>=start)&(u<=end)
        p=np.where(inside[...,None],released(u),p)
        released_controls=np.asarray(spec['bridge_controls']).copy()
        captured_control= released_controls[1]-released_controls[0]
        allowance=np.linalg.norm(captured_control)
        released_controls[0]=points[-1]
        released_controls[1]=points[-1]+(1-strength)*captured_control+strength*allowance*tangential[-1]/np.linalg.norm(tangential[-1])
    transported_bridge = spec.get('bridge_phase_transport',False)
    if (transported_bridge or local_pipe_motion is not None) and phase is None:
        phase=np.zeros(2)
    if phase is None:
        return p
    phase=np.asarray(phase).copy()
    axial=np.zeros(2)
    if local_pipe_motion is not None:
        phase+=np.asarray(local_pipe_motion['phase_rad'])
        axial=np.asarray(local_pipe_motion['axis_extension_m'])
    f = np.clip((u-spec['source_bound'])/(spec['target_bound']-spec['source_bound']), 0., 1.)
    f = f*f*(3-2*f)
    values = []
    for side, xy, center in [(0, [1, 2], [0., 1.]), (1, [0, 2], [1.1, 1.])]:
        a = np.asarray(phase)[..., side]
        radial = p[..., xy]-center
        rotated = np.stack([np.cos(a)*radial[..., 0]-np.sin(a)*radial[..., 1],
                            np.sin(a)*radial[..., 0]+np.cos(a)*radial[..., 1]], axis=-1)
        out = p.copy(); out[..., xy] = np.asarray(center)+rotated
        out[..., side]+=axial[side]
        values.append(out)
    result=values[0]*(1-f)[..., None]+values[1]*f[..., None]
    if transported_bridge:
        controls=np.asarray(spec['bridge_controls']).copy() if released_controls is None else released_controls
        for side,axes,center,ids in [(0,[1,2],[0.,1.],[0,1]),(1,[0,2],[1.1,1.],[2,3])]:
            angle=np.asarray(phase)[side];radial=controls[np.ix_(ids,axes)]-center
            controls[np.ix_(ids,axes)]=np.asarray(center)+np.column_stack([
                np.cos(angle)*radial[:,0]-np.sin(angle)*radial[:,1],
                np.sin(angle)*radial[:,0]+np.cos(angle)*radial[:,1]])
            controls[ids,side]+=axial[side]
        linear=np.clip((u-spec['source_bound'])/(spec['target_bound']-spec['source_bound']),0.,1.)
        t=linear[...,None]
        bridge=(1-t)**3*controls[0]+3*(1-t)**2*t*controls[1]+3*(1-t)*t*t*controls[2]+t**3*controls[3]
        bridge+=np.sin(np.pi*linear)[...,None]**2*np.asarray(spec['bridge_outward_offset'])
        inside=(u>=spec['source_bound'])&(u<=spec['target_bound'])
        result=np.where(inside[...,None],bridge,result)
    return result


def route_body(spec, shift, phase=None, bridge_height=0., spatial_offsets=None, local_pipe_motion=None, source_release=None):
    u = np.asarray(spec['spline_x'])
    xyz = sample_route(spec, u, phase,local_pipe_motion,source_release)
    if spatial_offsets is not None:
        grid, offsets = spatial_offsets
        xyz += np.column_stack([np.interp(u,grid,np.asarray(offsets)[:,j]) for j in range(3)])
    if bridge_height:
        f = np.clip((u-spec['source_bound'])/(spec['target_bound']-spec['source_bound']), 0., 1.)
        window = np.sin(np.pi*f)**2
        xyz[:, 2] += window*bridge_height
        xyz[:, 0] -= window*.004*min(1., bridge_height/.020)
        xyz[:, 1] -= window*.004*min(1., bridge_height/.020)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    head = float(np.interp(spec['initial_head_parameter'], u, arc))+shift
    def point(a):
        return np.array([np.interp(a, arc, xyz[:, j]) for j in range(3)])
    positions, aa = [point(head)], [head]
    if spec.get('material_window') == 'uniform_arclength':
        aa = head-np.arange(22)*LINK
        positions = np.asarray([point(a) for a in aa])
        residual = np.asarray(spec.get('entry_residual',np.zeros((22,3))))
        if phase is not None:
            residual_phase=np.asarray(phase)
            if local_pipe_motion is not None:residual_phase=residual_phase+np.asarray(local_pipe_motion['phase_rad'])
            rotated = []
            for side,axes in [(0,[1,2]),(1,[0,2])]:
                angle = residual_phase[side]
                r = residual[:,axes]
                values = residual.copy()
                values[:,axes] = np.column_stack([np.cos(angle)*r[:,0]-np.sin(angle)*r[:,1],
                                                  np.sin(angle)*r[:,0]+np.cos(angle)*r[:,1]])
                rotated.append(values)
            f = np.asarray(spec.get('entry_target_weights',np.zeros(22)))[:,None]
            residual = (1-f)*rotated[0]+f*rotated[1]
        fade = np.clip(shift/LINK,0.,1.)
        fade = 1-fade*fade*(3-2*fade)
        return positions+fade*residual,np.interp(aa,arc,u)
    for _ in range(21):
        right, left = aa[-1], aa[-1]-1.35*LINK
        while np.linalg.norm(point(left)-positions[-1]) < LINK:
            left -= .25*LINK
            if left < arc[0]:
                raise ValueError('Fixed global route extension exhausted')
        for _ in range(22):
            mid = .5*(left+right)
            if np.linalg.norm(point(mid)-positions[-1]) >= LINK:
                left = mid
            else:
                right = mid
        aa.append(.5*(left+right)); positions.append(point(aa[-1]))
    return np.asarray(positions), np.interp(aa, arc, u)
