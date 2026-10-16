"""Privileged known-pipe material charts; cumulative history never resets."""
import numpy as np
from physics import nodes,mujoco


def cached_truth(m,d):
    p=nodes(d);c=.5*(p[:-1]+p[1:]);h=m.body('head').id
    f=np.zeros((2,21));depth=np.zeros((2,21));normal=np.zeros((2,21,3));point=np.zeros((2,21,3))
    for k,ct in enumerate(d.contact):
        geoms=[ct.geom1,ct.geom2]
        for j,g in enumerate(geoms):
            name=m.geom(g).name or '';body=int(m.geom_bodyid[geoms[1-j]])-h
            if not name.startswith('pipe_') or not 0<=body<21:continue
            role=0 if name in ['pipe_straight_01','pipe_sharp_in_02'] else 1
            force=np.zeros(6);mujoco.mj_contactForce(m,d,k,force);f[role,body]+=max(0.,force[0])
            if -ct.dist>depth[role,body]:
                depth[role,body]=-ct.dist;normal[role,body]=(1 if 1-j==1 else -1)*ct.frame[:3];point[role,body]=ct.pos
    qa=m.jnt_qposadr[m.actuator_trnid[:,0]]
    return dict(points=p,centers=c,forces=f,body_depth=depth,contact_normal=normal,contact_point=point,
        joint_actual=d.qpos[qa].copy(),joint_ctrl=d.ctrl.copy(),joint_torque=d.actuator_force.copy())


class MaterialObserver:
    def __init__(self,ref):
        self.ref=ref;self.previous=None;self.previous_chart=np.zeros(21,int)
        self.progress=np.zeros(21);self.duty=np.zeros((2,21));self.filtered_force=np.zeros((2,21))
        self.source_chart_seen=np.zeros(21,bool);self.target_chart_seen=np.zeros(21,bool)
        self.target_anchor_S=np.full(21,np.nan);self.target_anchor_p=np.zeros(21)
        self.target_expected_matched=np.zeros(21);self.target_observed_matched=np.zeros(21)
        self.target_expected_unobserved=np.zeros(21);self.previous_S=0.
        self.unobserved_time=np.zeros(21);self.unobserved_nominal_feed=np.zeros(21)
        self.front_age=0.;self.front_confirmed=False
        self.target_speed=0.;self.source_speed=0.;self.target_stall=0.;self.source_stall=0.

    def update(self,obs,S,dt):
        c=obs['centers'];f=obs['forces'];alpha=1-np.exp(-dt/.3)
        self.duty+=alpha*(np.clip(f/.5,0,1)-self.duty)
        self.filtered_force+=alpha*(f-self.filtered_force)
        radius=np.linalg.norm(c[:,[0,2]]-[1.1,1.],axis=1)
        src=(self.duty[0]>.25)|(f[0]>1.)
        geo=(radius>.065)&(radius<.13)&(c[:,1]>.15)&(c[:,1]<1.2)
        tgt=geo&((self.duty[1]>.25)|(f[1]>1.))
        bridge=(~geo)&((self.duty[1]>.25)|(f[1]>1.))
        source_geo=(np.linalg.norm(c[:,1:]-[0.,1.],axis=1)>.065)&(np.linalg.norm(c[:,1:]-[0.,1.],axis=1)<.13)&(c[:,0]>-.2)&(c[:,0]<.95)
        if self.previous is None:self.source_chart_seen|=source_geo
        self.source_chart_seen|=src;self.target_chart_seen|=tgt
        role=np.where(src,1,np.where(tgt,2,np.where(bridge,3,0)))
        role[src&tgt]=4
        source_chart=src|(source_geo&self.source_chart_seen)
        target_chart=geo&(self.target_chart_seen|self.front_confirmed)
        chart=np.zeros(21,int);chart[source_chart&~target_chart]=1;chart[target_chart&~source_chart]=2
        # Dual support retains the previous valid chart; never double counts.
        dual=source_chart&target_chart;chart[dual]=self.previous_chart[dual]
        role[(role==0)&(chart==2)]=5
        increment=np.zeros(21);valid=np.zeros(21,bool)
        if self.previous is not None:
            source_valid=(chart==1)&(self.previous_chart==1)
            target_valid=(chart==2)&(self.previous_chart==2)
            increment[source_valid]=(c[source_valid,0]-self.previous[source_valid,0])/self.ref.a
            increment[target_valid]=(c[target_valid,1]-self.previous[target_valid,1])/self.ref.target_a
            valid=source_valid|target_valid;self.progress+=increment
        speed=increment/dt
        span=np.ptp(np.unwrap(np.arctan2(c[:6,2]-1,c[:6,0]-1.1)))
        front=bool(np.all(geo[:6]) and f[1,:6].sum()>.2 and span>np.pi)
        self.front_age=self.front_age+dt if front else 0.
        if self.front_age>=2:self.front_confirmed=True
        had_anchor=~np.isnan(self.target_anchor_S)
        anchor=(self.front_confirmed&(chart==2))&~had_anchor
        self.target_anchor_S[anchor]=S;self.target_anchor_p[anchor]=self.progress[anchor]
        anchors=~np.isnan(self.target_anchor_S)
        matched=had_anchor&valid&(chart==2)
        expected_increment=self.ref.feed_distance(S)-self.ref.feed_distance(self.previous_S)
        self.unobserved_time[~valid]+=dt
        self.unobserved_nominal_feed[~valid]+=expected_increment
        self.target_expected_matched[matched]+=expected_increment
        self.target_observed_matched[matched]+=increment[matched]
        self.target_expected_unobserved[had_anchor&~matched]+=expected_increment
        lead=self.target_expected_matched-self.target_observed_matched
        expected=(self.ref.feed_early+(self.ref.feed_late-self.ref.feed_early)*self.ref.smooth((S-6)/2))*self.ref.feed_per_cycle*.02
        tm=valid&(chart==2)&(np.arange(21)<15)
        sm=valid&(chart==1)&(np.arange(21)>=17)
        rate_alpha=1-np.exp(-dt/1.)
        if tm.any():self.target_speed+=rate_alpha*(float(np.mean(speed[tm]))-self.target_speed)
        if sm.any():self.source_speed+=rate_alpha*(float(np.mean(speed[sm]))-self.source_speed)
        tr=self.target_speed/max(expected,1e-6);sr=self.source_speed/max(expected,1e-6)
        self.target_stall=self.target_stall+dt if self.front_confirmed and tm.any() and tr<.25 else 0.
        self.source_stall=self.source_stall+dt if self.front_confirmed and sm.any() and sr<.25 else 0.
        late=np.arange(21)>=17;loaded=np.where(late,self.filtered_force[0],0.);index=int(np.argmax(loaded))
        lm=anchors&(chart==2)&(np.arange(21)<15)
        lag=float(max(0.,np.median(lead[lm]))) if lm.any() else 0.
        tracking=float(np.max(abs(obs['joint_ctrl'][14:]-obs['joint_actual'][14:]))*180/np.pi)
        info=dict(actual_material_role=role,actual_source_support=src,actual_target_support=tgt,
            material_progress_observed_m=self.progress.copy(),material_progress_valid=valid,
            material_chart=chart,material_target_relative_lead_m=lead,
            material_target_expected_matched_m=self.target_expected_matched.copy(),
            material_target_observed_matched_m=self.target_observed_matched.copy(),
            material_target_expected_unobserved_m=self.target_expected_unobserved.copy(),
            material_unobserved_time_s=self.unobserved_time.copy(),
            material_unobserved_nominal_feed_m=self.unobserved_nominal_feed.copy(),
            new_target_prediction_anchors=anchor,target_progress_lead_m=lag,
            target_progress_rate_ratio=tr,source_progress_rate_ratio=sr,
            target_rate_valid=bool(tm.any()),source_rate_valid=bool(sm.any()),
            target_progress_rate_m_s=self.target_speed,source_progress_rate_m_s=self.source_speed,
            target_stall_age_s=self.target_stall,source_stall_age_s=self.source_stall,
            front_confirmed=self.front_confirmed,front_current_valid=front,
            actual_target_support_count=int(tgt.sum()),source_loaded_material=index,
            source_late_filtered_N=float(loaded[index]),source_late_depth_m=float(obs['body_depth'][0,index]),
            tail_tracking_deg=tracking,target_anchor_S=np.nan_to_num(self.target_anchor_S,nan=-1.).copy())
        self.previous=c.copy();self.previous_chart=chart.copy();self.previous_S=S
        return info


class SharedSpeed:
    def __init__(self):self.ratio=1.;self.pause_remaining=0.;self.cooldown=0.
    def update(self,info,dt,enabled):
        target=1.;reason=0
        if enabled and info['front_confirmed']:
            receiver=np.clip((info['target_progress_lead_m']-.04)/.10,0,1)*np.clip((.7-info['target_progress_rate_ratio'])/.7,0,1)
            source=np.clip((info['source_late_filtered_N']-18)/15,0,1)*np.clip((.6-info['source_progress_rate_ratio'])/.6,0,1)
            if not info['target_rate_valid']:receiver=0.
            if not info['source_rate_valid']:source=0.
            if info['tail_tracking_deg']<3:source=0.
            target=1-.45*max(receiver,source)
            reason=1 if receiver>=source and receiver>0 else 2 if source>0 else 0
            emergency=(info['source_late_filtered_N']>26 and info['source_late_depth_m']>.0016 and info['source_stall_age_s']>.4 and info['tail_tracking_deg']>3) or (info['target_progress_lead_m']>.14 and info['target_stall_age_s']>.6)
            if emergency and self.cooldown<=0 and self.pause_remaining<=0:self.pause_remaining=.8;self.cooldown=3.8
            if self.pause_remaining>0:target=0.;reason=3
        self.cooldown=max(0.,self.cooldown-dt)
        if not enabled:self.ratio=1.
        else:
            down=1. if reason==3 else .5
            self.ratio=float(np.clip(target,self.ratio-down*dt,self.ratio+.5*dt))
            if reason==3 and self.ratio<.01:self.pause_remaining=max(0.,self.pause_remaining-dt)
        return dict(shared_speed_ratio=self.ratio,shared_speed_requested_ratio=target,speed_feedback_reason=reason,
            shared_pause_remaining_s=self.pause_remaining)
