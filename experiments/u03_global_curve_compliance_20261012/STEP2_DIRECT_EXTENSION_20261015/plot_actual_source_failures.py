"""Compare preserved source-hold failures, with no reruns or success claims."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True,layout='constrained')
for label,branch,color in [('V3.3','CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS','#4c78a8'),('V4.1','CONTROLLER_V4_1_ROBOT_MASS_HOLD','#f58518'),('V5','CONTROLLER_V5_SOURCE_CONTACT_DYNAMICS','#e45756'),('V5.1','CONTROLLER_V5_1_GRIP_PRELOAD_PROJECTION','#54a24b')]:
    z=np.load(P/branch/'T_UPDATE/trace.npz');t=z['t'];axes[0].plot(t,z['actual_source_motion_from_given_m']*1000,label=label,color=color);axes[1].plot(t,z['actual_source_boundary_X_motion_m']*1000,color=color);axes[2].plot(t,np.degrees(abs(z['ctrl'][:,8:]-z['ctrl'][0,8:]).max(1)),color=color)
    actual=z['sensor_force_by_estimated_role_N'][:,0,8:].sum(1);axes[3].plot(t,actual,color=color,label=label+' measured')
    if 'source_hold_allocated_normal_N' in z:
        allocated=z['source_hold_allocated_normal_N'].sum(1);valid=z['source_hold_active']&z['source_hold_qp_valid'];allocated[~valid]=np.nan;axes[3].plot(t,allocated,color=color,ls='--',label=label+' allocated(valid)')
axes[0].axhline(12,color='black',ls='--',lw=1,label='12mm stop guard');axes[0].set_ylabel('Max source motion\nfrom given pose (mm)');axes[0].legend(ncol=4)
axes[1].set_ylabel('Max source X motion\nfrom stable pose (mm)');axes[2].set_ylabel('Max rear ctrl change\nfrom given ctrl (deg)');axes[3].set_ylabel('Retained source\nnormal load (N)');axes[3].set_xlabel('Actual simulation time (s)');axes[3].legend(ncol=3,fontsize=8)
for ax in axes:ax.axvline(2.1,color='green',ls=':',lw=1);ax.grid(alpha=.2)
fig.suptitle('Actual source-hold development failures: all runs incomplete\nSame prescribed initial/physics; source controller revisions differ. Green: release after stable hold.\nNo receiver contact observed, geometry update or handoff. Curves end at last10Hz sample.')
fig.savefig(P/'ACTUAL_SOURCE_HOLD_FAILURES.png',dpi=150);plt.close(fig);print(str(P/'ACTUAL_SOURCE_HOLD_FAILURES.png'))
