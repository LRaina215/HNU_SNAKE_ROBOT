"""One full-source given coil, forward/render only. No dynamic integration."""
from pathlib import Path
import os,json,csv
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
from physics import *
import cv2
def forbidden(*a,**k):raise RuntimeError('Independent start audit pending; no dynamics')
for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,forbidden)
from evaluation import snapshot
from from_source_curve import FromSourceCurve
from geometry import Geometry
from static_equilibrium import balance
old=BASE.parent/'STEP2_20261014';original=ROOT/'experiments/cmu_native_underup_20261012/U03/initial.npz'
m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(original)
d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.ctrl[:]=z['ctrl'];mujoco.mj_forward(m,d)
before=nodes(d);translation=.90-float(np.max(before[:,0]));d.qpos[0]+=translation;mujoco.mj_forward(m,d)
equilibrium=balance(m,d)
row=snapshot(m,d);p=row['points'];c=row['centers'];rho=np.linalg.norm(c[:,1:]-[0,1],axis=1)
all_source=(c[:,0]>-.2)&(c[:,0]<.95)&(rho>=.065)&(rho<=.13)
normal=np.zeros(21);world=np.zeros((21,3));target_normal=np.zeros(21);other=[];head=m.body('head').id
for k,ct in enumerate(d.contact):
    gs=[ct.geom1,ct.geom2]
    for j,g in enumerate(gs):
        material=int(m.geom_bodyid[gs[1-j]])-head;name=m.geom(g).name or ''
        if not 0<=material<21 or not name.startswith('pipe_'):continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f)
        if name=='pipe_straight_01':normal[material]+=max(0,f[0]);world[material]+=(1 if 1-j==1 else -1)*ct.frame.reshape(3,3).T@f[:3]
        else:target_normal[material]+=max(0,f[0]);other.append(dict(material=material,geometry=name,normal_N=float(f[0])))
state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez_compressed(BASE/'SOURCE_INITIAL.npz',state=state,qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,S=0.,chi=0.,receiving_delta_m=0.)
constants=json.loads((old/'source_constants.json').read_text());constants['x0']+=translation
(BASE/'source_constants.json').write_text(json.dumps(constants,indent=2))
reg=json.loads((old/'REGISTERED.json').read_text());conditions=reg['conditions']
curves={}
for condition,prior in conditions.items():
    ref=FromSourceCurve(constants,Geometry.from_dict(prior));ref.configure_initial(d.ctrl);q,info=ref.command(0.)
    curves[condition]=dict(initial_raw_vs_ctrl_max_deg=float(np.max(abs(q-d.ctrl))*180/np.pi),S=float(info['S']),chi=float(info['chi']),roll_phase=float(info['roll_phase']),grow=float(info['uncoil_fraction']))
weight=float(m.body_mass[head:head+21].sum()*np.linalg.norm(m.opt.gravity))
proof=dict(single_prescribed_initial=True,source='U03 t0 coil, prescribed X placement plus static compliant equilibrium; no time integration or later state',translation_x_m=translation,static_equilibrium=equilibrium,
    geometric_rule='Set foremost backbone node X=.90m, 50mm behind known finite source endpoint; do not choose a later state',
    joint_count=20,rod_count=21,all21_source_envelope=bool(all_source.all()),source_envelope_count=int(all_source.sum()),
    source_center_radius_m=rho.tolist(),body_center_x_m=c[:,0].tolist(),backbone_x_range_m=[float(p[:,0].min()),float(p[:,0].max())],
    source_straight_normal_N=normal.tolist(),source_straight_vertical_N=world[:,2].tolist(),source_straight_vertical_total_N=float(world[:,2].sum()),robot_weight_N=weight,
    source_straight_vertical_fraction=float(world[:,2].sum())/weight,non_source_pipe_normal_total_N=float(target_normal.sum()),non_source_pipe_contacts=other,
    target_pre_wrap=False,target_pre_bearing=False,new_dynamics_starts=0,integration_calls=0,qvel_norm=float(np.linalg.norm(d.qvel)),
    instantaneous_qacc_norm=float(np.linalg.norm(d.qacc)),static_forward_is_not_dynamic_stability_proof=True,initial_curve_audit=curves,
    exact_static_equilibrium=False,requires_logged_full_source_stabilization=True,
    model_file=str(MODEL),physics_changed=False)
(BASE/'SOURCE_INITIAL_PROOF.json').write_text(json.dumps(proof,indent=2))
with (BASE/'SOURCE_INITIAL_21_RODS.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['material','x_m','source_radius_m','source_envelope','source_normal_N','source_vertical_N','non_source_normal_N'])
    for j in range(21):writer.writerow([j,c[j,0],rho[j],bool(all_source[j]),normal[j],world[j,2],target_normal[j]])
ren=mujoco.Renderer(m,width=850,height=650);cam=mujoco.MjvCamera();cam.distance=1.9;cam.lookat[:]=[.70,.13,1.];ims=[]
for az,el,label in [(90,-8,'SOURCE SIDE'),(140,-26,'OBLIQUE')]:
    cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam)
    for k in range(ren.scene.ngeom):
        g=ren.scene.geoms[k]
        if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
            j=int(m.geom_bodyid[g.objid])-head
            if 0<=j<21:g.rgba[:3]=[.95,.72,.15] if j<8 else [.2,.60,.95]
    im=cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR)
    lines=[f'UNIQUE t=0 FULL SOURCE COIL / {label}',f'21/21 source envelope; source Fz={world[:,2].sum():.2f}N; weight={weight:.2f}N',
        f'Target / elbow contact Fn={target_normal.sum():.6f}N; S=0 chi=0; original20 joints',
        'Private forward / no dynamics; front M0..7 yellow, rear M8..20 blue']
    for j,line in enumerate(lines):cv2.putText(im,line,(12,27+j*23),0,.45,(255,255,255),1)
    ims.append(im)
cv2.imwrite(str(BASE/'SOURCE_INITIAL_DUAL.png'),np.concatenate(ims,axis=1));ren.close()
assert all_source.all() and target_normal.sum()==0.
assert all(v['initial_raw_vs_ctrl_max_deg']<1e-8 for v in curves.values())
print(json.dumps({k:proof[k] for k in ['all21_source_envelope','translation_x_m','source_straight_vertical_total_N','source_straight_vertical_fraction','non_source_pipe_normal_total_N','instantaneous_qacc_norm','initial_curve_audit']},indent=2))
