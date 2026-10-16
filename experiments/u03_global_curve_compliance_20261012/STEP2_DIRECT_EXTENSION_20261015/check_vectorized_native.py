"""Limited numerical equivalence/performance check, no dynamics integration."""
from pathlib import Path
import sys,json,time,copy,numpy as np
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V9_1_PRIOR_DOMAIN_BEARING';sys.path.insert(0,str(B))
from physics import *
from native_step1_reference import NativeStep1Reference
from geometry import Geometry
from native_mapping import NativeMapping
from native_mapping_vectorized import FastNativeMapping,fast_transport_normal
r=json.loads((B/'REGISTERED.json').read_text());m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(B/'GIVEN_INITIAL.npz');mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
f=NativeStep1Reference(nodes(d),d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]],Geometry.from_dict(r['conditions']['DIRECTION']));f.release_materials=12.;fast=copy.deepcopy(f);fast.transport_normal=fast_transport_normal;rows=[]
for S in [0.,9.,20.]:
    f.shape_progress=fast.shape_progress=6.+min(S,3.);start=time.perf_counter();q0,i0=NativeMapping.command(f,S);old_wall=time.perf_counter()-start
    start=time.perf_counter();q1,i1=FastNativeMapping.command(fast,S);fast_wall=time.perf_counter()-start
    row=dict(S=S,q_error_rad=float(abs(q0-q1).max()),field_error=float(abs(i0['curvature_field']-i1['curvature_field']).max()),C_error_m=float(abs(i0['nominal_curve_dense']-i1['nominal_curve_dense']).max()),original_wall_s=old_wall,vectorized_wall_s=fast_wall);rows.append(row)
    assert row['q_error_rad']<1e-10 and row['field_error']<1e-10 and row['C_error_m']<1e-9
out=dict(simulation_steps=0,selected_for_current_actual_run=False,actual_running_version=r['version'],purpose='Equivalent algebraic native integration speed only; not a new controller/physics test',rows=rows);(P/'NATIVE_VECTOR_EQUIVALENCE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
