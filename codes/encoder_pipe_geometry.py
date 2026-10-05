"""Two-cylinder geometry hypothesis from encoder-reconstructed robot points.

Scope: intersecting right-angle pipes of comparable radius. No pipe model,
contact identity, measured global pose or manually selected joint boundary is
an input. The input point coordinates may use any rigid frame. This estimates
geometry only: a fitted coil is NOT proof of contact or sustainable support.
"""
import numpy as np
from scipy.optimize import least_squares


def cylinder(points, nominal_radius, axis_hint=None):
    centers=(points[:-1]+points[1:])/2
    centered=centers-centers.mean(axis=0)
    links=np.diff(points,axis=0)
    tangents=links/np.linalg.norm(links,axis=1)[:,None]
    _,vectors=np.linalg.eigh(np.cov(tangents.T,bias=True))
    candidates=[]
    seeds=vectors.T if axis_hint is None else [np.asarray(axis_hint)/np.linalg.norm(axis_hint)]
    angular_bound=5. if axis_hint is None else .2
    for seed in seeds:
        h=np.eye(3)[np.argmin(abs(seed))];h-=h@seed*seed;h/=np.linalg.norm(h)
        v=np.cross(seed,h)
        def decode(x):
            axis=seed+x[0]*h+x[1]*v;axis/=np.linalg.norm(axis)
            e=h-(h@axis)*axis;e/=np.linalg.norm(e)
            center=x[2]*e+x[3]*np.cross(axis,e)
            return axis,center
        def residual(x):
            axis,center=decode(x)
            return np.linalg.norm(np.cross(centered-center,axis),axis=1)-x[4]
        result=least_squares(residual,[0.,0.,0.,0.,nominal_radius],
            bounds=([-angular_bound,-angular_bound,-3*nominal_radius,-3*nominal_radius,.6*nominal_radius],
                    [angular_bound,angular_bound,3*nominal_radius,3*nominal_radius,1.8*nominal_radius]),
            x_scale=[.4,.4,nominal_radius,nominal_radius,nominal_radius],
            loss='soft_l1',f_scale=5.,max_nfev=300)
        axis,center=decode(result.x)
        candidates.append(dict(axis=axis,axis_point_mm=center+centers.mean(axis=0),
            radius_mm=float(result.x[4]),radial_rms_mm=float(np.sqrt(np.mean(residual(result.x)**2))),
            objective=float(result.cost),solver_success=bool(result.success)))
    return min(candidates,key=lambda row:row['objective'])


def branch(centers, source):
    a=source['axis'];p=source['axis_point_mm'];r=source['radius_mm']
    h=np.eye(3)[np.argmin(abs(a))];h-=h@a*a;h/=np.linalg.norm(h)
    v=np.cross(a,h);z0=float(np.mean((centers-p)@a))
    def decode(x):
        return np.cos(x[0])*h+np.sin(x[0])*v,p+x[1]*a
    def residual(x):
        d,c=decode(x)
        radial=np.linalg.norm(np.cross(centers-c,d),axis=1)-x[2]
        # A soft comparable-radius prior, not the true pipe diameter.
        return np.r_[radial,.5*(x[2]-r)]
    solutions=[]
    for phi in np.linspace(-np.pi,np.pi,12,endpoint=False):
        fitted=least_squares(residual,[phi,z0,r],
            bounds=([-10*np.pi,z0-3*r,.6*r],[10*np.pi,z0+3*r,1.8*r]),
            x_scale=[1.,r,r],loss='soft_l1',f_scale=5,max_nfev=200)
        d,c=decode(fitted.x);axial=(centers-c)@d
        if np.mean(axial)<0:d=-d;axial=-axial
        radial=centers-c-axial[:,None]*d
        e=radial[0]/np.linalg.norm(radial[0]);f=np.cross(d,e)
        angles=np.unwrap(np.arctan2(radial@f,radial@e))
        solutions.append(dict(axis=d,corner_mm=c,radius_mm=float(fitted.x[2]),
            rms_mm=float(np.sqrt(np.mean((np.linalg.norm(radial,axis=1)-fitted.x[2])**2))),
            objective=float(fitted.cost),wrap=float(abs(angles[-1]-angles[0])/(2*np.pi)),
            axial_mm=axial,solver_success=bool(fitted.success)))
    return min(solutions,key=lambda row:row['objective'])


def right_angle_hypothesis(points, nominal_radius, shell_radius=29., source_fit=None):
    """Fixed observation windows, NOT fixed control segment boundaries.

    The last ten-joint span estimates the established original coil; first
    seven module centres test a different direction. This initial-drape
    hypothesis is invalid once the rear has already left its original pipe.
    Temporal confirmation and support-response evidence are separate steps.
    """
    points=np.asarray(points,float);centers=(points[:-1]+points[1:])/2
    rear=cylinder(points[-12:],nominal_radius) if source_fit is None else dict(source_fit)
    if rear['axis']@(points[-8]-points[-1])<0:rear['axis']=-rear['axis']
    front=branch(centers[:7],rear)
    source_error=np.linalg.norm(np.cross(centers[:7]-rear['axis_point_mm'],rear['axis']),axis=1)-rear['radius_mm']
    single_rms=float(np.sqrt(np.mean(source_error**2)))
    quality=dict(rear_fit=rear['radial_rms_mm']<5.,front_fit=front['rms_mm']<12.,
        distinct_direction=single_rms-front['rms_mm']>15.,wrap=front['wrap']>.35,
        branch_forward=float(np.min(front['axial_mm']))>-nominal_radius/2)
    return dict(source=rear['axis'],target=front['axis'],corner_mm=front['corner_mm'],
        pipe_radius_mm=rear['radius_mm']-shell_radius,rear=rear,front=front,
        single_axis_front_rms_mm=single_rms,geometric_candidate=all(quality.values()),quality=quality)
