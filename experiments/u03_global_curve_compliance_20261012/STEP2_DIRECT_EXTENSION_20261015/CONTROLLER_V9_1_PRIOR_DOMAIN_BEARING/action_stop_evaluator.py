"""Evaluation-only full action stop. No outputs are fed to the controller."""
import numpy as np

class ActionStopEvaluator:
    def __init__(self):self.window=[]
    def update(self,row):
        c=row['centers'];p=row['points'];force=row['eval_pipe_normal_N']
        radius=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
        span=np.ptp(np.unwrap(np.arctan2(c[:,2]-1,c[:,0]-1.1)))
        full=bool(np.all((radius>.07)&(radius<.12)&(c[:,1]>.15)&(c[:,1]<1.2)) and np.all((p[:,1]>.15)&(p[:,1]<1.2)) and force[:2].sum()<.1 and force[2].sum()>.2 and span>=2*np.pi)
        if not full:self.window=[];return False
        self.window.append((row['t'],c[:,1].copy(),row['actual_axial_spin_rad_s'].copy()))
        while len(self.window)>2 and self.window[-1][0]-self.window[1][0]>=50.:self.window.pop(0)
        if self.window[-1][0]-self.window[0][0]<50.:return False
        tt=np.array([r[0] for r in self.window]);spin=np.degrees(np.trapz(np.array([r[2] for r in self.window]),tt,axis=0));advance=self.window[-1][1]-self.window[0][1]
        return bool((advance>.04).all() and (abs(spin)>90.).all() and ((spin>0).all() or (spin<0).all()))
