"""Plot the three already-saved isolated witnesses; no new solve or integration."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_SOURCE_AXIS_BALANCE';sys.path.insert(0,str(B))
from rough_planner import RoughPlanner
from geometry import Geometry
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));initial=np.load(B/'GIVEN_INITIAL.npz');saved=np.load(B/'BALANCE_WITNESSES.npz');report=json.loads((B/'BALANCE_WITNESSES.json').read_text(encoding='utf-8'));fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');angles=np.linspace(0,2*np.pi,200)
for col,(alpha,qpos,row) in enumerate(zip(saved['alpha'],saved['qpos'],report['rows'])):
    plan=RoughPlanner(Geometry.from_dict(r['conditions']['DIRECTION']),initial['qpos'],initial['ctrl'],r['kinematic_solver']);G,goal,Tgoal,n=plan.guide(float(alpha),0.);C=plan.fk(qpos[plan.qa]);T=plan.d.xmat[plan.head].reshape(3,3)[:,2];com=np.average(plan.d.xipos[plan.head:plan.head+21],weights=plan.m.body_mass[plan.head:plan.head+21],axis=0)
    for ax,dims,labels in [(axes[0,col],[1,2],['Y(m)','Z(m)']),(axes[1,col],[0,1],['X(m)','Y(m)'])]:
        ax.plot(G[:,dims[0]],G[:,dims[1]],'--',c='#777777',lw=1.2,label='independent G')
        ax.plot(C[:9,dims[0]],C[:9,dims[1]],'o-',c='#e45756',ms=3,label='free fore C')
        ax.plot(C[8:,dims[0]],C[8:,dims[1]],'o-',c='#4c78a8',ms=3,label='retained source C')
        ax.scatter(*com[dims],marker='*',s=80,c='black',label='total robot COM')
        for vec,color in [(T,'#e45756'),(Tgoal,'#777777')]:ax.quiver(*C[0,dims],*vec[dims],angles='xy',scale_units='xy',scale=14,color=color,width=.006)
        ax.set_xlabel(labels[0]);ax.set_ylabel(labels[1]);ax.set_aspect('equal');ax.grid(alpha=.2)
        all_points=np.vstack([G,C,C[0]+T/14,C[0]+Tgoal/14,com]);lo=all_points[:,dims].min(0)-.025;hi=all_points[:,dims].max(0)+.025;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
    axes[0,col].plot(.0565*np.cos(angles),1+.0565*np.sin(angles),c='black',lw=1)
    axes[0,col].axvline(0,c='black',lw=.6,ls=':')
    axes[0,col].set_title(f'alpha={alpha:g}; moment={row["net_source_axis_moment_after_Nm"]:.4f}Nm\nhead error={row["head_error_mm"]:.2f}mm; tangent={row["tangent_error_deg"]:.1f}deg')
axes[1,0].legend(fontsize=8,loc='best');fig.suptitle('V6 isolated geometric load-reduction witnesses (no dynamics, no path proof)\nSource held in these private kinematic poses; arrows show achieved and goal head tangents. Large tangent errors remain.')
fig.savefig(B/'BALANCE_WITNESS_GEOMETRY.png',dpi=160);plt.close(fig);print(str(B/'BALANCE_WITNESS_GEOMETRY.png'))
