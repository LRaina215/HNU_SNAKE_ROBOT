"""Head-first intake directly from an artificial all-source lower inner exit."""
import numpy as np,mujoco
from snake_control.head_first_inner_control import HeadFirstInnerControl
from snake_control.head_first_inner_route import HeadFirstInnerRoute
from snake_control.suspended_supply_transfer import WholeCurveMapper,nodes

class ReadyInnerEntry(HeadFirstInnerControl):
    def __init__(self,model,data,anchored_feed=False):
        super().__init__(model,data,anchored_feed=anchored_feed)
        p=nodes(data);self.route=HeadFirstInnerRoute(p,smooth_source=False)
        self.enter('head_first_inner_feed',data,0.)
        self.feed_phase=0.;self.feed_phase_base=0.;self.progress_head=p[0].copy();self.last_progress=0.
        self.before_feed_guide=self.route.window(0)[0];self.feed_mapper=WholeCurveMapper(model)
        self.guard=mujoco.MjData(model)
        self.events['artificial_ready_inner_start']=dict(t_s=0.,head_mm=p[0]*1000,entry_vector_mm=(p[0]-p[1])*1000,bridge_length_mm=self.route.bridge_length*1000,all_source=True,target_acquired=False,artificial=True)

    def command(self,data,t,forces,source_any):
        q,info=super().command(data,t,forces,source_any)
        # Evaluate only the current instruction; the actual root stays measured.
        self.guard.qpos[:]=data.qpos;self.guard.qpos[self.qa]=q;mujoco.mj_fwdPosition(self.m,self.guard)
        pipe=self_depth=0.
        head=self.m.body('head').id
        for con in self.guard.contact:
            b1,b2=map(int,self.m.geom_bodyid[[con.geom1,con.geom2]]);a=head<=b1<head+21;b=head<=b2<head+21
            if a!=b:
                g=con.geom2 if a else con.geom1
                if (self.m.geom(g).name or '').startswith('pipe_'):pipe=max(pipe,-float(con.dist))
            elif a and b and abs(b1-b2)>1:self_depth=max(self_depth,-float(con.dist))
        info['current_command_pipe_depth_mm']=pipe*1000;info['current_command_self_depth_mm']=self_depth*1000
        info['current_command_guard']='private FK at measured actual root; no future whole-route admission'
        if pipe>.003 or self_depth>.003 or info['projection']['predicted_pipe_depth_mm']>3:
            self.failure='unsafe_current_joint_command_before_dispatch'
        return q,info
