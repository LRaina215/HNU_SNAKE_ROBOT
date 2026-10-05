"""Causal calibration API: encoder/command response -> virtual pipe model.

No scene name, physical root pose, velocity, gravity direction or contacts are
accepted. The explicit environment prior is intersecting right-angle pipes
with gravity along an axis or their plane normal. The robot/friction model is
assumed known. This research calibrator has not been qualified for hardware.
"""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import least_squares
from feedback_takeover_dynamic_observer_probe import virtual_model,predict


def decode_geometry(initial,x):
    s=np.asarray(initial['source']);t=np.asarray(initial['target']);normal=np.cross(s,t)
    source=s+x[3]*t+x[4]*normal;source/=np.linalg.norm(source)
    perpendicular=t-(t@source)*source;perpendicular/=np.linalg.norm(perpendicular)
    target=np.cos(x[5])*perpendicular+np.sin(x[5])*np.cross(source,perpendicular)
    return dict(initial,corner_mm=np.asarray(initial['corner_mm'])+x[:3],source=source,target=target,
        pipe_radius_mm=initial['pipe_radius_mm']+x[6])


def junction_seed(robot,robot_xml,initial,records):
    """Coarse ambiguity along the original pipe, scored only by probe response.

    A partially wrapped front can fit a corner near the pipe surface rather
    than its centre. Cover one inferred radius along the source and +/-0.3rad
    around it before the existing local fit; no real corner is consulted.
    """
    rows=[];measured=np.array([r['q'] for r in records[:41]])
    source=np.asarray(initial['source']);target=np.asarray(initial['target'])
    normal=np.cross(source,target);radius=float(initial['pipe_radius_mm'])
    for shift in np.linspace(-radius,radius,5):
        for angle in (-.3,0.,.3):
            estimate=dict(initial,corner_mm=np.asarray(initial['corner_mm'])+shift*source,
                target=np.cos(angle)*target+np.sin(angle)*normal)
            axes=(source,estimate['target'],np.cross(source,estimate['target']))
            for axis in axes:
                for sign in (-1,1):
                    prediction=predict(virtual_model(robot,estimate,9.81*sign*axis,robot_xml),records[:41])
                    rms=float(np.sqrt(np.mean(np.degrees(np.asarray(prediction['predicted'])-measured)**2)))
                    rows.append(dict(shift_mm=float(shift),angle_rad=angle,rms_deg=rms,estimate=estimate))
    best=min(rows,key=lambda r:r['rms_deg'])
    return best['estimate'],dict(selected=best,ranked=sorted(rows,key=lambda r:r['rms_deg'])[:10])


def calibrate(robot,robot_xml,initial,records,junction_search=False,extension=None):
    coarse=None;unrefined=initial
    seed_bases=[initial]
    if junction_search:
        # Do not rank raw collision residuals before local refinement: a wrong
        # radius can hide a useful junction seed beneath source-contact error.
        # Each of three geometry hypotheses gets the same gravity/local fit.
        seed_bases += [dict(initial,corner_mm=np.asarray(initial['corner_mm'])+sign*initial['pipe_radius_mm']*np.asarray(initial['source']))
            for sign in (-1,1)]
        coarse=dict(method='three locally refined corner starts',source_offsets_in_pipe_radii=[0,-1,1])
    measured=np.array([r['q'] for r in records]);train=records[:41];history=[];begin=time.perf_counter()
    axes={name:np.asarray(initial[name]) for name in ('source','target')}
    axes['normal']=np.cross(axes['source'],axes['target'])
    geometry_low=np.r_[np.full(3,-20.),np.full(3,-.35),-5.]
    if junction_search:geometry_low[5]=-.7
    geometry_scale=np.r_[np.full(3,3.),np.full(3,.05),1.]

    def fit(start,decode,steps,low,high,scale,error_scale,central,velocity):
        calls=0
        def residual(x):
            nonlocal calls
            estimate,g,v=decode(x)
            prediction=predict(virtual_model(robot,estimate,g,robot_xml,extension),train,v)
            error=np.degrees(np.asarray(prediction['predicted'])-measured[:41])
            prior=np.r_[x[:3]/5.,x[3:6]/.1,x[6]/2.]
            prior=np.r_[prior,x[7:10]/.03,x[10:13]/.3] if velocity else np.r_[prior,(np.linalg.norm(g)-9.81)/.2]
            calls+=1
            return np.r_[error.reshape(-1)/error_scale,prior]
        def jacobian(x):
            base=residual(x);columns=[]
            for i,h in enumerate(steps):
                plus=x.copy();plus[i]+=h
                if central:
                    minus=x.copy();minus[i]-=h;columns.append((residual(plus)-residual(minus))/(2*h))
                else:columns.append((residual(plus)-base)/h)
            return np.column_stack(columns)
        fitted=least_squares(residual,start,jac=jacobian,bounds=(low,high),x_scale=scale,
            max_nfev=60,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        return fitted,calls

    # First solve unconstrained gravity from all six starts; this gives the
    # geometry optimizer a response-derived start rather than a saved answer.
    free=[]
    for seed_index,seed_base in enumerate(seed_bases):
        def free_decode(x):return decode_geometry(seed_base,x),x[7:10],None
        for name,axis in axes.items():
            for sign in (-1,1):
                start=np.r_[np.zeros(7),9.81*sign*axis]
                result,calls=fit(start,free_decode,np.r_[np.full(3,.1),np.full(3,.001),.05,np.full(3,.1)],
                    np.r_[geometry_low,np.full(3,-10.)],np.r_[-geometry_low,np.full(3,10.)],
                    np.r_[geometry_scale,np.full(3,3.)],.2,False,False)
                free.append((result,seed_base));history.append(dict(stage='free',seed_index=seed_index,axis=name,sign=sign,cost=float(result.cost),parameters=result.x,calls=calls))
                print('CALIBRATE free',seed_index,name,sign,'cost',round(result.cost,4),'wall',round(time.perf_counter()-begin,1),flush=True)
    best_free,initial=min(free,key=lambda row:row[0].cost)
    aligned=[]
    for name in axes:
        for sign in (-1,1):
            def aligned_decode(x):
                estimate=decode_geometry(initial,x)
                axis=np.cross(estimate['source'],estimate['target']) if name=='normal' else estimate[name]
                return estimate,9.81*sign*axis,x[7:13]
            start=np.r_[best_free.x[:7],np.zeros(6)]
            result,calls=fit(start,aligned_decode,np.r_[np.full(3,.01),np.full(3,.0001),.005,np.full(3,.0001),np.full(3,.001)],
                np.r_[geometry_low,np.full(3,-.1),np.full(3,-1.)],np.r_[-geometry_low,np.full(3,.1),np.full(3,1.)],
                np.r_[geometry_scale,np.full(3,.02),np.full(3,.2)],.02,True,True)
            estimate,g,v=aligned_decode(result.x)
            aligned.append(dict(cost=float(result.cost),estimate=estimate,gravity=g,initial_velocity=v,parameters=result.x,axis=name,sign=sign))
            history.append(dict(stage='aligned',axis=name,sign=sign,cost=float(result.cost),parameters=result.x,calls=calls))
            print('CALIBRATE aligned',name,sign,'cost',round(result.cost,4),'wall',round(time.perf_counter()-begin,1),flush=True)
    best=min(aligned,key=lambda r:r['cost'])
    model=virtual_model(robot,best['estimate'],best['gravity'],robot_xml,extension)
    prediction=predict(model,records,best['initial_velocity'])
    error=np.degrees(np.asarray(prediction['predicted'])-measured)
    return dict(best,history=history,prediction=prediction,initial=initial,unrefined=unrefined,junction_search=coarse,extension=extension,
        training_rms_deg=float(np.sqrt(np.mean(error[:41]**2))),held_out_rms_deg=float(np.sqrt(np.mean(error[41:]**2))),
        wall_seconds=time.perf_counter()-begin,
        limitation='Calibration from collected angles/commands only. Explicit right-angle and horizontal/vertical gravity prior; exact nominal dynamics. No true geometry/pose/contacts/velocity in objective or model selection.')


def calibrate_network(robot,robot_xml,initial,records,junction_search=True):
    """Compare L and both T hypotheses, without reading the physical layout.

    All candidates receive the same measurements and train on the first 0.8s.
    The final 0.4s remains a gate. These expensive fits are research-time work.
    """
    begin=time.perf_counter()
    # Each hypothesis creates its own model/data; inputs are read-only. These
    # independent fits can share wall time without changing the objective,
    # start set, training/held-out split or deterministic selection order.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(calibrate,robot,robot_xml,initial,records,junction_search,extension)
                 for extension in (None,'source','target')]
        fits=[future.result() for future in futures]
    best=min(fits,key=lambda row:row['training_rms_deg'])
    return dict(best,network_wall_seconds=time.perf_counter()-begin,
        network_hypotheses=[dict(extension=row['extension'],training_rms_deg=row['training_rms_deg'],
        held_out_rms_deg=row['held_out_rms_deg'],estimate=row['estimate'],gravity=row['gravity'],history=row['history']) for row in fits])
