"""Same-material tangent/twist diagnostics at key saved states; no steps."""
from physics import *
from scipy.spatial.transform import Rotation
import csv,json
rows=[];summary=[]
for name in ['T_FROZEN','T_UPDATE','P_FROZEN','P_UPDATE']:
    case=BASE/name;z=dict(np.load(case/'trace.npz'));m=mujoco.MjModel.from_xml_path(str(case/'model.xml'));d=mujoco.MjData(m);h=m.body('head').id;last=None
    for sec in [0.,15.,30.,75.,150.,234.9]:
        i=int(np.argmin(abs(z['t']-sec)));d.qpos[:]=z['qpos'][i];mujoco.mj_kinematics(m,d)
        field=z['curvature_field'][i];bounds=z['curvature_bounds_mm'][i];material=z['material_arc_nodes_mm'][i]
        vec=np.column_stack([np.zeros(len(field)),-field[:,1],field[:,0]])
        steps=Rotation.from_rotvec(vec*np.diff(bounds)[:,None]).as_matrix();frames=[z['reference_frame'][i]]
        for step in steps:frames.append(frames[-1]@step)
        frames=np.array(frames);mid=.5*(material[:-1]+material[1:]);bins=np.clip(np.searchsorted(bounds,mid,side='right')-1,0,len(field)-1)
        partial=Rotation.from_rotvec(vec[bins]*(mid-bounds[bins])[:,None]).as_matrix();reference=frames[bins]@partial
        body=d.xmat[h:h+21].reshape(21,3,3);actual_tangent=-body[:,:,2]
        # Native XML alternates a fixed pi rotation about bodyZ at each link.
        # q_even=-integral(kB), q_odd=-integral(kN), physical axes X,-Y
        # correspond to material N=-(-1)^i*bodyX, T=-bodyZ.
        actual_normal=-((-1.)**np.arange(21))[:,None]*body[:,:,0]
        target_tangent=reference[:,:,0];cross=np.cross(actual_tangent,target_tangent);sine=np.linalg.norm(cross,axis=1);cosine=np.clip(np.sum(actual_tangent*target_tangent,axis=1),-1,1);axis=cross/np.maximum(sine[:,None],1e-12)
        aligned=actual_normal*cosine[:,None]+np.cross(axis,actual_normal)*sine[:,None]+axis*np.sum(axis*actual_normal,axis=1)[:,None]*(1-cosine[:,None])
        aligned[sine<1e-12]=actual_normal[sine<1e-12]
        phase=np.degrees(np.arctan2(np.sum(aligned*reference[:,:,2],axis=1),np.sum(aligned*reference[:,:,1],axis=1)));heading=np.degrees(np.arccos(cosine))
        for n in range(21):rows.append(dict(case=name,t=float(z['t'][i]),material=n,tangent_angle_deg=float(heading[n]),twist_difference_after_minimal_tangent_alignment_deg=float(phase[n]),comparison_defined=bool(cosine[n]>-1+1e-8)))
        last=dict(t=float(z['t'][i]),tangent_angle_rms_deg=float(np.sqrt(np.mean(heading**2))),wrapped_material_twist_rms_deg=float(np.sqrt(np.mean(phase**2))))
    summary.append(dict(case=name,last=last,world_reference_reregistered=False,scope='Coordinate-frame comparison only; swing/twist decomposition, not actual self-spin or commonchi subtraction'))
with (BASE/'MATERIAL_FRAME_DIAGNOSTICS.csv').open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(BASE/'MATERIAL_FRAME_DIAGNOSTICS.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
