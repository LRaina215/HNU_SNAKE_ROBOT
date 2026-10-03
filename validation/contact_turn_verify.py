"""Five held-out states for the frozen oracle-motion policy; no retuning."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'validation'))
from contact_turn_experiment import BASE,Simulation,write_json,sha
import contact_turn_stage1


class InitialOffsetSimulation(Simulation):
    initial_offset_m=0.
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.data.time==0 and not self.commands
        # This is the explicitly varied initial position, set before ANY
        # target or physical step. Never reset state during a probe/motion.
        self.data.qpos[:3]+=self.axis*self.initial_offset_m
        mujoco.mj_forward(self.model,self.data)
        if self.data.ncon:
            self.close()
            raise ValueError('Held-out initial offset collides with scene')
        self.initial_qpos=self.data.qpos.copy()
        self.records.clear();self.snapshot()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');args=p.parse_args()
    frozen_path=BASE/'stage1_parameter_freeze.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf8'))
    assert frozen['training_completed']==12
    for name,value in frozen['selected_manifest']['code_hash'].items():
        assert sha(ROOT/name)==value,'Frozen source changed: '+name
    if not args.run:
        print(json.dumps(frozen,ensure_ascii=False));return
    results=[]
    for i,state in enumerate(frozen['test_states']):
        InitialOffsetSimulation.initial_offset_m=state['source_position_offset_m']
        contact_turn_stage1.Simulation=InitialOffsetSimulation
        options=SimpleNamespace(front=frozen['front'],width=frozen['width_mm'],turn=frozen['turn_deg'],
                                phase=state['phase'],attempt=f'verification_v1_{i}',out=BASE,budget=60.)
        label=f'f{options.front}_w{options.width:g}_turn{options.turn:g}_p{options.phase:g}_{options.attempt}'
        directory=BASE/'stage1'/label
        if (directory/'result.json').exists():
            result=json.loads((directory/'result.json').read_text(encoding='utf8'))
        else:
            result=contact_turn_stage1.trial(options)
        write_json(directory/'held_out_state.json',dict(state=state,freeze_sha256=sha(frozen_path),
                     verification_entry_sha256=sha(__file__),initial_offset_applied_before_first_physics=True))
        result['directory']=str(directory.relative_to(ROOT));results.append(result)
        write_json(BASE/'stage1_verification_progress.json',dict(completed=len(results),required=5,results=results))
        if result['outcome']=='engineering_failure':
            raise RuntimeError('Verification engineering failure, not a motion verdict')
    success=sum(x['success'] for x in results)
    write_json(BASE/'stage1_release_gate.json',dict(stage=1,required_successes=4,total=5,successes=success,
                passed=success>=4,stage2_released=success>=4,stage3_released=False,
                results=results,freeze_sha256=sha(frozen_path),
                stop_reason=None if success>=4 else 'Known-direction motion did not pass 4/5 gate; perception and closed loop remain untested'))


if __name__=='__main__':main()
