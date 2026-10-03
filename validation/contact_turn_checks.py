"""Independent recorded-state audit and local-bending geometry checks."""
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'codes'),str(ROOT/'validation')]
from Adaptive_Pole_Climbing_Mujoco import HelixBackbone
from local_bend_backbone import LocalBendBackbone
from contact_turn_experiment import BASE,write_json,sha


def stage0_audit():
    m=mujoco.MjModel.from_xml_path(str(BASE/'models/horizontal_straight.xml'))
    d=mujoco.MjData(m)
    qadr=np.array([int(m.joint(f'joint{i}').qposadr[0]) for i in range(1,21)])
    vadr=np.array([int(m.joint(f'joint{i}').dofadr[0]) for i in range(1,21)])
    rows=[]
    for p in sorted((BASE/'stage0_round2_clearance').glob('phase*/trace.npz')):
        r=np.load(p)
        result=json.loads((p.parent/'result.json').read_text(encoding='utf8'))
        t=r['sim_time']
        assert np.allclose(np.diff(t),.01,rtol=0,atol=1e-10)
        assert np.array_equal(r['q'],r['qpos'][:,qadr])
        assert np.array_equal(r['velocity'],r['qvel'][:,vadr])
        hold=(t>result['hold_start_s']+1e-8)&(t<=result['hold_start_s']+10+1e-8)
        assert hold.sum()==1000
        formula=np.clip(127*(r['ctrl']-r['q']),-2.355,2.355)
        assert np.max(abs(formula-r['torque']))<1e-10
        assert np.max(abs(r['actuator_force']-r['torque']))<1e-10
        max_points=max_torque=0.
        initial_contacts=[]
        ground=False
        for i in range(len(t)):
            d.qpos[:]=r['qpos'][i];d.qvel[:]=r['qvel'][i];d.ctrl[:]=r['ctrl'][i]
            mujoco.mj_forward(m,d)
            points=np.vstack([d.site('s_head').xpos,*[d.joint(f'joint{j}').xanchor for j in range(1,21)],d.site('s_tail').xpos])
            max_points=max(max_points,float(abs(points-r['points'][i]).max()))
            max_torque=max(max_torque,float(abs(d.qfrc_actuator[vadr]-r['torque'][i]).max()))
            floor=m.geom('floor').id
            ground |= any((c.geom1==floor or c.geom2==floor) for c in d.contact)
            if i==0:
                initial_contacts=[dict(geom1=int(c.geom1),geom2=int(c.geom2),distance_m=float(c.dist)) for c in d.contact]
        assert max_points<1e-12 and max_torque<1e-10 and not ground
        commands=json.loads((p.parent/'commands.json').read_text(encoding='utf8'))
        assert all(x['applied'] for x in commands)
        assert all(x['computed_sim_time']==x['applied_sim_time'] for x in commands)
        assert [x['command_id'] for x in commands]==list(range(1,len(commands)+1))
        # Every negative-feedback stop must stop increasing the affected probe;
        # applied commands during return strictly move toward frozen targets.
        assert result['probe_recovery_passed']
        row=dict(case=p.parent.name,trace_sha256=sha(p),samples=len(t),hold_samples=int(hold.sum()),
                 physical_end_s=float(t[-1]),max_fk_difference_m=max_points,max_torque_difference_nm=max_torque,
                 initial_contacts=initial_contacts,commands=len(commands),probes=result['probes'])
        rows.append(row)
    assert len(rows)==3
    assert all(not row['initial_contacts'] for row in rows)
    report=dict(passed=True,cases=rows,raw_rate_hz=100,physics_dt_s=.001,
                note='Initial conditions are set once; no qpos resets after time zero. Contact forces are evaluator-only.')
    write_json(BASE/'stage0_independent_audit.json',report)
    print(json.dumps(dict(stage0_audit_passed=True,cases=3,initial_contacts=[len(x['initial_contacts']) for x in rows])))


def geometry_checks():
    helix=HelixBackbone.from_radius_pitch(81.5,220.)
    m=mujoco.MjModel.from_xml_path(str(BASE/'models/horizontal_straight.xml'))
    d=mujoco.MjData(m)
    links=np.full(21,64.5)
    zeros=[];rows=[]
    for phase in np.linspace(0,2*math.pi,25):
        b=LocalBendBackbone(helix,links)
        zeros.append(float(abs(b.joint_angles(phase)-helix.joint_angles(20,links,phase)).max()))
    for front in (4,6):
        for width in (64.5,129.):
            for degrees in (15.,30.,45.,60.,90.):
                max_angle=0.;self_contacts=0
                for direction in np.linspace(0,2*math.pi,8,endpoint=False):
                    for phase in (0.,2*math.pi/3,4*math.pi/3):
                        b=LocalBendBackbone(helix,links,mu_mm=front*64.5/2,width_mm=width,
                                            turn_rad=math.radians(degrees),direction_rad=direction)
                        q=b.joint_angles(phase)
                        max_angle=max(max_angle,float(np.degrees(abs(q)).max()))
                        d.qpos[:]=m.key_qpos[0];d.qpos[7:]=q;mujoco.mj_forward(m,d)
                        head=m.body('head').id
                        self_contacts+=int(sum(m.geom_bodyid[c.geom1]>=head and m.geom_bodyid[c.geom2]>=head and
                                           abs(m.geom_bodyid[c.geom1]-m.geom_bodyid[c.geom2])>1 for c in d.contact)
                                           )
                rows.append(dict(front=front,width_mm=width,turn_deg=degrees,
                                 max_target_deg=max_angle,nonadjacent_self_contacts=self_contacts,
                                 passed=max_angle<=90 and self_contacts==0))
    phase=.37
    reference=helix.joint_angles(20,links,phase)*.85
    b=LocalBendBackbone(helix,links,reference_angles=reference,reference_phase=phase)
    reference_error=float(abs(b.joint_angles(phase)-reference).max())
    b=LocalBendBackbone(helix,links,mu_mm=129.,width_mm=64.5,turn_rad=.5,direction_rad=.7)
    q=b.joint_angles(phase)
    independent=[]
    for j in range(20):
        component=1 if j%2==0 else 0
        independent.append(-quad(lambda s:float(b.field(s,phase)[component]),j*64.5,(j+2)*64.5,
                                 points=[(j+1)*64.5],epsabs=1e-11)[0])
    quadrature_error=float(abs(q-independent).max())
    sample=b.sample(phase)
    gram=np.stack([sample['normal1'],sample['normal2'],sample['tangent']],axis=2)
    frame_error=float(abs(np.transpose(gram,(0,2,1))@gram-np.eye(3)).max())
    arc_error=float(abs(np.linalg.norm(np.diff(sample['points_mm'],axis=0),axis=1).sum()-b.bounds[-1])/b.bounds[-1])
    changes=[]
    for knob in ('mu','width','turn','direction'):
        kwargs=dict(mu_mm=129.,width_mm=64.5,turn_rad=.5,direction_rad=.7)
        key={'mu':'mu_mm','width':'width_mm','turn':'turn_rad','direction':'direction_rad'}[knob]
        kwargs[key]+=1e-5
        changes.append(float(np.degrees(abs(LocalBendBackbone(helix,links,**kwargs).joint_angles(phase)-q)).max()))
    assert max(zeros)<1e-12 and reference_error<1e-10 and quadrature_error<1e-10 and frame_error<1e-7
    assert max(changes)<.001
    report=dict(zero_turn_error_rad=max(zeros),reference_shape_error_rad=reference_error,
                independent_integral_error_rad=quadrature_error,bishop_frame_error=frame_error,
                polyline_arc_length_relative_error=arc_error,small_parameter_change_max_deg=max(changes),
                scan=rows,source_sha256=sha(ROOT/'codes/local_bend_backbone.py'),
                geometry_scope='Model joint range and active nonadjacent collisions; no certification of real adjacent shell clearance')
    write_json(BASE/'local_bend_geometry_checks.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='scan'}))


if __name__=='__main__':
    stage0_audit()
    geometry_checks()
