"""Persistent material charts relative to estimated target geometry."""
import numpy as np

class EstimatedObserver:
    def __init__(self,ref,source):
        self.ref=ref;self.source=source;self.previous=None;self.previous_axis=None;self.previous_chart=np.zeros(21,int)
        self.progress=np.zeros(21);self.duty=np.zeros((3,21));self.filtered_force=np.zeros((3,21))
        self.source_chart_seen=np.zeros(21,bool);self.target_chart_seen=np.zeros(21,bool)
        self.target_anchor_S=np.full(21,np.nan);self.target_expected_matched=np.zeros(21);self.target_observed_matched=np.zeros(21);self.target_expected_unobserved=np.zeros(21)
        self.previous_S=None;self.unobserved_time=np.zeros(21);self.unobserved_nominal_feed=np.zeros(21)
        self.front_age=0.;self.front_confirmed=False;self.target_speed=0.;self.source_speed=0.;self.target_stall=0.;self.source_stall=0.
    def update(self,obs,S,dt,geometry,geometry_reliable=False):
        c=obs['centers'];f=obs['forces'];alpha=1-np.exp(-dt/.3);self.duty+=alpha*(np.clip(f/.5,0,1)-self.duty);self.filtered_force+=alpha*(f-self.filtered_force)
        axial,rad,radius=geometry.coordinates(c);center_radius=geometry.radius+.036
        geo=(abs(radius-center_radius)<.031)&(axial>-.05)&(axial<1.)
        src=(self.duty[0]>.25)|(f[0]>1.);tgt=geo&((self.duty[1]>.25)|(f[1]>1.))
        bridge=((~geo)&((self.duty[1]>.25)|(f[1]>1.)))|(self.duty[2]>.25)|(f[2]>1.)
        source_r=np.linalg.norm(c[:,1:]-self.source['center_yz_m'],axis=1)
        source_geo=(abs(source_r-(self.source['radius_m']+.036))<.031)&(c[:,0]>self.source['axial_bounds_m'][0])&(c[:,0]<self.source['axial_bounds_m'][1])
        if self.previous is None:self.source_chart_seen|=source_geo
        self.source_chart_seen|=src;self.target_chart_seen|=tgt
        role=np.where(src,1,np.where(tgt,2,np.where(bridge,3,0)));role[src&tgt]=4
        sc=src|(source_geo&self.source_chart_seen);tc=geo&(self.target_chart_seen|self.front_confirmed)
        chart=np.zeros(21,int);chart[sc&~tc]=1;chart[tc&~sc]=2;dual=sc&tc;chart[dual]=self.previous_chart[dual];role[(role==0)&(chart==2)]=5
        inc=np.zeros(21);valid=np.zeros(21,bool)
        target_a=self.ref.pitch/np.sqrt((2*np.pi*(geometry.radius+.039))**2+self.ref.pitch**2)
        if self.previous is not None:
            sv=(chart==1)&(self.previous_chart==1);tv=(chart==2)&(self.previous_chart==2)
            inc[sv]=(c[sv,0]-self.previous[sv,0])/self.ref.a
            axis=geometry.axis+self.previous_axis;axis/=max(np.linalg.norm(axis),1e-12)
            # Project actual displacement; moving estimated origin cannot create progress.
            inc[tv]=((c[tv]-self.previous[tv])@axis)/target_a;valid=sv|tv;self.progress+=inc
        speed=inc/dt if dt>0 else np.zeros(21);e,b=geometry.frame();polar=np.unwrap(np.arctan2(rad[:6]@b,rad[:6]@e));span=np.ptp(polar)
        front=bool(geometry_reliable and np.all(geo[:6]) and f[1,:6].sum()>.2 and span>np.pi)
        self.front_age=self.front_age+dt if front else 0.
        if self.front_age>=2:self.front_confirmed=True
        had=~np.isnan(self.target_anchor_S);anchor=(self.front_confirmed&(chart==2))&~had;self.target_anchor_S[anchor]=S
        matched=had&valid&(chart==2);expected_inc=0. if self.previous_S is None else self.ref.feed_distance(S)-self.ref.feed_distance(self.previous_S)
        self.unobserved_time[~valid]+=dt;self.unobserved_nominal_feed[~valid]+=expected_inc
        self.target_expected_matched[matched]+=expected_inc;self.target_observed_matched[matched]+=inc[matched];self.target_expected_unobserved[had&~matched]+=expected_inc
        lead=self.target_expected_matched-self.target_observed_matched
        tm=valid&(chart==2)&(np.arange(21)<15);sm=valid&(chart==1)&(np.arange(21)>=17);ra=1-np.exp(-dt/1.)
        if tm.any():self.target_speed+=ra*(float(np.mean(speed[tm]))-self.target_speed)
        if sm.any():self.source_speed+=ra*(float(np.mean(speed[sm]))-self.source_speed)
        slope=(self.ref.feed_distance(S+1e-4)-self.ref.feed_distance(S))/1e-4
        expected=slope*self.ref.feed_per_cycle*self.ref.nominal_frequency_hz
        tr=self.target_speed/max(expected,1e-6);sr=self.source_speed/max(expected,1e-6)
        self.target_stall=self.target_stall+dt if self.front_confirmed and tm.any() and tr<.25 else 0.
        self.source_stall=self.source_stall+dt if self.front_confirmed and sm.any() and sr<.25 else 0.
        late=np.arange(21)>=17;loaded=np.where(late,self.filtered_force[0],0.);index=int(np.argmax(loaded));lm=had&(chart==2)&(np.arange(21)<15)
        info=dict(estimated_material_role=role,estimated_source_support=src,estimated_receiver_support=tgt,
            estimated_bridge_support=bridge,estimated_receiver_qualified=tgt|((chart==2)&self.front_confirmed),sensor_force_by_estimated_role_N=f.copy(),
            material_progress_observed_m=self.progress.copy(),material_progress_valid=valid,material_chart=chart,
            material_target_relative_lead_m=lead,material_target_expected_matched_m=self.target_expected_matched.copy(),material_target_observed_matched_m=self.target_observed_matched.copy(),material_target_expected_unobserved_m=self.target_expected_unobserved.copy(),
            material_unobserved_time_s=self.unobserved_time.copy(),material_unobserved_nominal_feed_m=self.unobserved_nominal_feed.copy(),new_target_prediction_anchors=anchor,
            target_progress_lead_m=float(max(0.,np.median(lead[lm]))) if lm.any() else 0.,target_progress_rate_ratio=tr,source_progress_rate_ratio=sr,target_rate_valid=bool(tm.any()),source_rate_valid=bool(sm.any()),
            target_progress_rate_m_s=self.target_speed,source_progress_rate_m_s=self.source_speed,target_stall_age_s=self.target_stall,source_stall_age_s=self.source_stall,
            front_confirmed=self.front_confirmed,front_current_valid=front,actual_target_support_count=int(tgt.sum()),source_loaded_material=index,source_late_filtered_N=float(loaded[index]),source_late_depth_m=float(obs['body_depth'][0,index]),
            tail_tracking_deg=float(np.max(abs(obs['joint_ctrl'][14:]-obs['joint_actual'][14:]))*180/np.pi),target_anchor_S=np.nan_to_num(self.target_anchor_S,nan=-1.).copy())
        self.previous=c.copy();self.previous_axis=geometry.axis.copy();self.previous_chart=chart.copy();self.previous_S=S
        return info

