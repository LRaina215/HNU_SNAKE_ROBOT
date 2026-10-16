"""No integration: post-capture route/material/native-command numerical audit."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import argparse,json
from physics import *
from geometry import Geometry,normalize
from rough_planner import RoughPlanner
from post_capture_curve import PostCaptureCurve

def run(folder):
    P=Path(folder).resolve();reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(P/'CANDIDATE_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    geometry=Geometry([1.1,.2,1.],[0,1,0],.0565);plan=RoughPlanner(geometry,d.qpos,d.ctrl,reg['kinematic_solver']);path=np.load(P/'DIAGNOSTIC_TRUE_AXIS_T_WRAP.npz');G=path['G'][-1];d.qpos[:]=path['qpos'][-1];mujoco.mj_kinematics(m,d);R=d.xmat[m.body('head').id].reshape(3,3);T=normalize(G[1]-G[0]);N=normalize(R[:,0]-R[:,0]@T*T);frame=np.column_stack([T,N,np.cross(T,N)]);post=PostCaptureCurve(plan.given,geometry,G,frame,reg)
    rows=[]
    for S in np.linspace(0,27,55):
        q,i=post.command(float(S));field=plan.coordinates.field(q);rows.append(dict(S=float(S),native_joint_peak_deg=float(abs(q).max()*180/np.pi),native_minimum_norm_curvature_peak_per_m=float(np.linalg.norm(field,axis=1).max()*1000),G_material_length_error_mm=float(abs(np.linalg.norm(np.diff(i['global_reference_points'],axis=0),axis=1)-.0645).max()*1000),all_finite=bool(np.all(np.isfinite(q))),initial_G_departure_mm=float(np.linalg.norm(i['global_reference_points']-G,axis=1).max()*1000) if S==0 else None))
    report=dict(simulation_steps=0,role='Counterfactual post-curve numerics only; does not assert that capture occurred',maximum_native_joint_deg=max(v['native_joint_peak_deg'] for v in rows),maximum_curvature_per_m=max(v['native_minimum_norm_curvature_peak_per_m'] for v in rows),all_finite=all(v['all_finite'] for v in rows),rows=rows)
    (P/'POST_CURVE_NUMERICAL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder');run(ap.parse_args().folder)
