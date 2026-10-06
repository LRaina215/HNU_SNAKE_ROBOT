"""Repair only failed command sample phases in the same saved reference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json,shutil
import numpy as np
from scipy.interpolate import PchipInterpolator
from front_local_rolling_continue import NumericalFit,OUT
from front_local_rolling_geometry import ORIGINAL,write
from snake_control.front_local_rolling import FrontLocalRollingReference


def main():
    folder=OUT/'command_closure';folder.mkdir(exist_ok=False)
    for name in ['reference_partial.npz','full_review.json','progress.json']:
        shutil.copy2(OUT/name,folder/('before_'+name))
    fit=NumericalFit();progress=json.loads((OUT/'progress.json').read_text(encoding='utf8'));fit.used_s=progress['numerical_used_s']
    with np.load(OUT/'reference_partial.npz') as z:
        keys=z['phase_rad'].copy();values=np.column_stack((z['joint_rad'],z['axial_mm'],z['orbital_rad']))
    ip=PchipInterpolator(keys,values,axis=0);saved=dict(zip(map(float,keys),values));rows=[]
    cases=json.loads((OUT/'command_failure_diagnosis.json').read_text(encoding='utf8'))
    for case in cases:
        phi=case['phase_rad'];z,row=fit.solve_feasible(phi,ip(phi),True);rows.append(row)
        if not row['ready']:raise RuntimeError('Failed command point projection')
        saved[phi]=z
        newkeys=np.array(sorted(saved));vv=np.array([saved[k] for k in newkeys])
        np.savez_compressed(OUT/'reference_partial.npz',phase_rad=newkeys,joint_rad=vv[:,:12],axial_mm=vv[:,12],orbital_rad=vv[:,13])
        progress['rows'].append(row);progress['numerical_used_s']=fit.used_s;write(OUT/'progress.json',progress)
        write(folder/'progress.json',dict(physics_steps=0,rows=rows,numerical_used_s=fit.used_s))
        print(json.dumps({k:row[k] for k in ['phase_deg','ready','numerical_used_s']}),flush=True)
    write(folder/'result.json',dict(physics_steps=0,points=len(rows),status='REQUIRES_REVIEW',numerical_used_s=fit.used_s))


if __name__=='__main__':main()
