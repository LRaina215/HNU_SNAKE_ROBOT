"""Independent numerical checks for scoring/statistics, never simulation data."""
import math

import numpy as np

from pitch_adaptation_mechanism import partition_body
from pitch_adaptation_statistics import holm,summarize
from pitch_adaptation_experiment_v5 import SCENES


def checks():
    origin=np.zeros(3);axis=np.array([0.,0.,1.]);x=np.array([1.,0.,0.]);y=np.array([0.,1.,0.])
    theta=np.linspace(0.,4*np.pi,101);z=np.linspace(0.,2.,101)
    points=np.column_stack((np.cos(theta),np.sin(theta),z))
    frame=(origin,axis,x,y,np.array([.6,1.4]),['before','transition','after'])
    lengths,turns,_,_=partition_body(points,frame)
    assert np.allclose(turns,[.6,.8,.6],atol=1e-12)
    assert abs(lengths.sum()-np.linalg.norm(np.diff(points,axis=0),axis=1).sum())<1e-12
    # Traverse one revolution and then undo it. Absolute increments would
    # incorrectly claim two turns; signed winding must be zero.
    theta=np.r_[np.linspace(0,2*np.pi,51),np.linspace(2*np.pi,0,51)[1:]]
    reversed_points=np.column_stack((np.cos(theta),np.sin(theta),np.linspace(0,2,len(theta))))
    _,reversed_turns,_,_=partition_body(reversed_points,(origin,axis,x,y,np.array([]),['pipe']))
    assert abs(reversed_turns[0])<1e-12
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])
    rows=[dict(scene=scene,method=method,seed=seed,success=method=='D',outcome='synthetic_unit_fixture',
               speed_m_s=.01,hold_max_retreat_mm=0.,torque_rms_nm=1.)
          for scene in SCENES for method in 'ABCD' for seed in range(101,111)]
    result=summarize(rows)
    comparison=next(r for r in result['comparisons'] if r['scene']=='real_s2l' and r['right']=='B')
    assert comparison['wins']==10 and comparison['losses']==0
    assert math.isclose(comparison['exact_mcnemar_two_sided_p'],2/2**10)
    assert math.isclose(comparison['primary_holm_p'],6*2/2**10)
    assert result['stage3_released'] is False
    zero=next(r for r in result['methods'] if r['method']=='B')
    assert zero['success_interval95'][0]==0.
    assert math.isclose(zero['success_interval95'][1],1-.025**.1,rel_tol=1e-10)
    try:summarize(rows[:-1])
    except ValueError:pass
    else:raise AssertionError('Incomplete formal design was accepted')
    rows[0]['hold_max_retreat_mm']=None
    missing=summarize(rows)['comparisons'][0]['paired_metrics']['hold_max_retreat_mm']
    assert missing['n']==9 and missing['missing_seeds']==[101]
    print('PASS: region conservation, signed reversal, exact intervals/McNemar, Holm, paired missing data, incomplete-design refusal. Synthetic fixtures only.')


if __name__=='__main__':checks()
