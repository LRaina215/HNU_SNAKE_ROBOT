"""Preserved source support and front-force policy; no phase advancement."""
import numpy as np
from physics import mujoco
from contact_sensor import known_source,source_geometry

class SourcePolicy:
    def support(self,sensor,t,observed):
        source=self.reg['source_known'];retained=0.;all_source=0.;fixed_rear=0.;known_normal=0.;unknown_load=0.;self.current_supply_materials=set()
        c=sensor['centers'];rho=np.linalg.norm(c[:,1:]-source['center_yz_m'],axis=1)
        geo=source_geometry(c,source)
        self.source_chart_seen|=geo
        known_by_material=np.zeros(21)
        for ct in sensor['contacts']:
            if known_source(ct,source):known_by_material[ct['material']]+=ct['normal_N']
        membership=geo|(known_by_material>.02);components=[];start=0
        while start<21:
            if not membership[start]:start+=1;continue
            end=start
            while end+1<21 and membership[end+1]:end+=1
            angles=np.unwrap(np.arctan2(c[start:end+1,2]-source['center_yz_m'][1],c[start:end+1,1]-source['center_yz_m'][0]))
            components.append(dict(start=start,end=end,count=end-start+1,winding=float(np.ptp(angles)*180/np.pi) if end>start else 0.,normal=float(known_by_material[start:end+1].sum())))
            start=end+1
        bearing=max(components,key=lambda x:(x['normal']>=self.reg['support']['source_min_normal_N'],x['winding'],x['count'])) if components else dict(start=21,end=20,count=0,winding=0.,normal=0.)
        tail_start=21
        while tail_start>0 and geo[tail_start-1]:tail_start-=1
        tail_count=21-tail_start
        polar=np.unwrap(np.arctan2(c[tail_start:,2]-source['center_yz_m'][1],c[tail_start:,1]-source['center_yz_m'][0]))
        winding=float(np.ptp(polar)*180/np.pi) if tail_count>1 else 0.
        best={}
        for ct,role in zip(sensor['contacts'],observed['contact_estimated_classes']):
            j=ct['material']
            if known_source(ct,source) or role['receiver_candidate_consistent']:continue
            if j not in best or ct['normal_N']>best[j]['normal_N']:best[j]=ct
        for j,ct in best.items():
            old=self.unknown_candidates.get(j);continuous=old is not None and 0<=t-old['t']<=.05 and np.linalg.norm(ct['point']-old['point'])<.015 and ct['normal']@old['normal']>np.cos(np.radians(30))
            age=old['age']+max(0.,t-old['t']) if continuous else 0.
            self.unknown_candidates[j]=dict(t=t,age=age,point=ct['point'].copy(),normal=ct['normal'].copy())
        for ct,role in zip(sensor['contacts'],observed['contact_estimated_classes']):
            j=ct['material'];previous=self.support_tracks.get(j);known=known_source(ct,source)
            continuous=bool(previous is not None and t-previous['t']<=.10 and np.linalg.norm(ct['point']-previous['point'])<.015 and ct['normal']@previous['normal']>np.cos(np.radians(30)))
            adjacent=max(0,bearing['start']-self.reg['support']['bridge_adjacent_materials'])<=j<=min(20,bearing['end']+self.reg['support']['bridge_adjacent_materials'])
            candidate=self.unknown_candidates.get(j);persistent=candidate is not None and t-candidate['t']<.05 and candidate['age']>=self.reg['support']['new_bridge_persistence_s']
            eligible=known or (persistent and self.source_chart_seen[j] and adjacent and not role['receiver_candidate_consistent'])
            if eligible:
                self.support_tracks[j]=dict(t=t,point=ct['point'].copy(),normal=ct['normal'].copy())
                retained+=ct['world_force_N'][2]
                self.current_supply_materials.add(j)
                if not known:unknown_load+=ct['world_force_N'][2]
                if j>=8:fixed_rear+=ct['world_force_N'][2]
            if known:all_source+=ct['world_force_N'][2];known_normal+=ct['normal_N']
        return dict(retained_supply_N=float(retained),fixed_rear8to20_N=float(fixed_rear),known_source_N=float(all_source),all_source=bool(geo.all()),
            tail_chart_start=tail_start,tail_chart_count=tail_count,tail_source_winding_deg=winding,known_source_normal_N=float(known_normal),unknown_bridge_N=float(unknown_load),head_on_source=bool(geo[:2].all()),
            bearing_chart_start=bearing['start'],bearing_chart_end=bearing['end'],bearing_chart_count=bearing['count'],bearing_winding_deg=bearing['winding'],bearing_normal_N=bearing['normal'])
    def adequate_supply(self,support,weight):
        p=self.reg['support']
        return bool(support['retained_supply_N']>=p['rear_load_min_fraction_before_capture']*weight and
            support['bearing_chart_count']>=p['source_tail_min_materials'] and support['bearing_winding_deg']>=p['source_tail_min_winding_deg'] and support['bearing_normal_N']>=p['source_min_normal_N'])
    def local_front(self,d,sensor,dt):
        info=self.info;bounds=info['curvature_bounds_mm'];nodes=info['material_arc_nodes_mm'];mid=.5*(bounds[:-1]+bounds[1:]);v=np.clip(mid/nodes[8],0,1)
        w=np.sin(np.pi*v)**2;w[bounds[1:]>nodes[8]]=0.
        A=np.maximum(0,np.minimum(nodes[2:,None],bounds[None,1:])-np.maximum(nodes[:-2,None],bounds[None,:-1]));span=A@w/1000
        mapping=np.zeros((20,2));odd=np.arange(20)%2==1;mapping[odd,0]=-span[odd];mapping[~odd,1]=-span[~odd]
        desired=np.zeros(2);head=self.m.body('head').id;columns=self.m.jnt_dofadr[self.m.actuator_trnid[:,0]]
        supports=[ct for ct in sensor['contacts'] if ct['material']>=8 and ct['normal_N']>.5 and
            (known_source(ct,self.reg['source_known']) or ct['material'] in self.support_tracks)]
        if not supports:return
        anchor=max(supports,key=lambda ct:ct['normal_N'])['material']
        jpa=np.zeros((3,self.m.nv));jra=np.zeros((3,self.m.nv));mujoco.mj_jacBody(self.m,d,jpa,jra,head+anchor)
        J=np.vstack([jpa,jra]);root_mapping=-np.linalg.solve(J[:,:6],J[:,columns]@mapping)
        def normal_gradient(ct):
            jp=np.zeros((3,self.m.nv));jr=np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,d,jp,jr,ct['point'],head+ct['material'])
            return ct['normal']@(jp[:,columns]@mapping+jp[:,:6]@root_mapping)
        non_source={}
        if self.post is None:
            for ct in sensor['contacts']:
                if ct['material']<8 and not known_source(ct,self.reg['source_known']):non_source.setdefault(ct['material'],[]).append(ct)
        downward={i:cts for i,cts in non_source.items() if sum(ct['world_force_N'][2] for ct in cts)<0.}
        for ct in sensor['contacts']:
            if ct['material']>=8 or ct['normal_N']<=12:continue
            if ct['material'] in downward and not known_source(ct,self.reg['source_known']):continue
            gradient=normal_gradient(ct);n=np.linalg.norm(gradient)
            if n>1e-7:desired+=.012*(ct['normal_N']-12)*gradient/n
        threshold=self.reg['robot_weight_N']/21.
        for cts in downward.values():
            load=sum(ct['normal_N'] for ct in cts)
            if load<=threshold:continue
            gradient=sum(ct['normal_N']*normal_gradient(ct) for ct in cts);n=np.linalg.norm(gradient)
            if n>1e-7:desired+=.012*(load-threshold)*gradient/n
        n=np.linalg.norm(desired);desired*=min(1.,.30/max(n,1e-12))
        step=desired-self.front_relief;n=np.linalg.norm(step);step*=min(1.,.06*dt/max(n,1e-12));self.front_relief+=step
        self.ref.front_relief=self.front_relief.copy()
