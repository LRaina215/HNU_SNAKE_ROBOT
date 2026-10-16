"""Visualize the saved real task/phase mismatch; no new solve or rollout."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np,ast,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;z=np.load(P/'CONTROLLER_V6_1_TRANSMITTED_C_STATE/T_UPDATE/trace.npz');t=z['t'];head_error=np.linalg.norm(z['points'][:,0]-z['global_reference_points'][:,0],axis=1)*1000;accepted=np.array([ast.literal_eval(str(m)).get('accepted',False) for m in z['planner_metrics']]);fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
axes[0].plot(t,z['alpha'],label='alpha phase');axes[0].set_ylabel('Phase alpha');axes[0].legend()
axes[1].plot(t,head_error,label='actual head - G head goal');axes[1].plot(t,np.linalg.norm(z['points'][:,0]-z['points'][0,0],axis=1)*1000,label='actual head departure from given');axes[1].set_ylabel('Distance(mm)');axes[1].legend()
axes[2].step(t,accepted.astype(int),where='post',label='C candidate accepted');axes[2].set_ylabel('Accepted C (0/1)');axes[2].set_xlabel('Actual simulation time(s)');axes[2].legend()
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('V6.1 actual failure: phase advances while geometric C is rejected\nSame original given/physics. No receiving contact or completed action.10Hz saved samples.')
fig.savefig(P/'V6_1_PHASE_TASK_STALL.png',dpi=150);plt.close(fig);print(str(P/'V6_1_PHASE_TASK_STALL.png'))
