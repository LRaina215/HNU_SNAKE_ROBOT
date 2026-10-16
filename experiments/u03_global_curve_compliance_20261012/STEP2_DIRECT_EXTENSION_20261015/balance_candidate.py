"""Static compliant adjustment of prescribed candidate, original physics only."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,mujoco
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'mujoco_model').is_dir());sys.path.insert(0,str(P.parent/'STEP2_FROM_SOURCE_20261015'))
from static_equilibrium import balance
for k in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,k,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No integration')))
m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'));d=mujoco.MjData(m);z=np.load(P/'CANDIDATE_INITIAL.npz')
mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);result=balance(m,d)
state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez_compressed(P/'BALANCED_CANDIDATE_INITIAL.npz',state=state,qpos=d.qpos.copy(),ctrl=d.ctrl.copy())
result.update(original_physics_enableflags=int(m.opt.enableflags),source_file='CANDIDATE_INITIAL.npz',output='BALANCED_CANDIDATE_INITIAL.npz',independent_initial_acceptance=False,
    scope='Given-pose static adjustment only, no dynamics; all-source and direct-reach proof must be repeated for final selected initial')
(P/'BALANCED_CANDIDATE_INITIAL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
