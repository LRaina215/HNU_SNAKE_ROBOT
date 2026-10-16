"""Match existing relative total-command protection; preserve grip preload."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent;B=P/'CONTROLLER_V6_6_COMMAND_AWARE_CLEARANCE';old=P/'CONTROLLER_V6_5_DOWNWARD_CONTACT_RELIEF'
audit=B/'COMMAND_CLEARANCE_AUDIT.json'
if audit.exists():audit.rename(B/'REJECTED_ABSOLUTE_COMMAND_CLEARANCE_AUDIT.json')
s=(old/'rough_planner.py').read_text(encoding='utf-8')
a='self.coordinates=CurvatureCoordinates();self.q=self.initial_q.copy();self.snapshot=None;';b='self.coordinates=CurvatureCoordinates();self.q=self.initial_q.copy();self.command_local_offset=np.zeros(20);self.snapshot=None;';assert s.count(a)==1;s=s.replace(a,b)
a='self.last_geometric_targets=geometric_targets.copy();previous=self.q.copy();previous_points=self.fk(previous).copy();active=np.arange(n)'
b='''self.last_geometric_targets=geometric_targets.copy();previous=self.q.copy();offset=self.command_local_offset.copy();previous_command_points=self.fk(previous+offset).copy();previous_command_planes=[]
        for ct in self.contact_planes:
            body=self.head+ct['material'];point=self.d.xpos[body]+self.d.xmat[body].reshape(3,3)@ct['local'];previous_command_planes.append(-ct['depth']+ct['normal']@(point-ct['point']))
        previous_points=self.fk(previous).copy();active=np.arange(n)'''
assert s.count(a)==1;s=s.replace(a,b)
start=s.index('            cache.update(source_axis_balance_moment_Nm=')
extra='''            # Optimize against the SAME incremental compression rule used
            # by the final total-command filter. Existing preload can command
            # virtual overlap; its baseline is retained, never treated as new
            # physical penetration. All original geometric checks above stay.
            geometric_gap_count=len(gaps);constraint_labels=[('geometric',)+pair for pair in pairs]+[('geometric','measured_plane',ct['material']) for ct in self.contact_planes if ct['material']<n]
            command_points=self.fk(q+offset);jpa=np.zeros((3,self.m.nv));jra=np.zeros_like(jpa);mujoco.mj_jac(self.m,self.d,jpa,jra,self.snapshot['anchorP'],self.head+self.anchor);jpa=jpa[:,self.va[:n]];jra=jra[:,self.va[:n]]
            for index,ct in enumerate(self.contact_planes):
                i=ct['material']
                if i>=n:continue
                body=self.head+i;point=self.d.xpos[body]+self.d.xmat[body].reshape(3,3)@ct['local'];gap=-ct['depth']+ct['normal']@(point-ct['point']);limit=min(-.0008,previous_command_planes[index]-.0001);gradient=ct['normal']@self.point_J(point,body,jpa,jra,n)[0]
                gaps.append(gap);limits.append(limit);gradients.append(gradient);constraint_labels.append(('command','measured_plane',i));violation=min(0.,gap-limit)
                residual.append(np.array([violation/.001]));jac.append((gradient/.001)[None,:] if violation<0 else np.zeros((1,n)))
            if position_cap is not None:
                displacement=command_points[:n]-previous_command_points[:n];Jpoints=np.array([self.point_J(command_points[i],self.head+i,jpa,jra,n)[0] for i in range(n)])
                motion=np.r_[motion,1-np.sum(displacement**2,axis=1)/position_cap**2];motion_J=np.vstack([motion_J,-2*np.einsum('ij,ijk->ik',displacement,Jpoints)/position_cap**2])
'''
s=s[:start]+extra+s[start:];s=s.replace('cache.update(source_axis_balance_moment_Nm=','cache.update(constraint_labels=constraint_labels,geometric_gap_count=geometric_gap_count,source_axis_balance_moment_Nm=')
a="            most_active_clearance_pair=str(pairs[int(np.argmax(cache['limits'][:len(pairs)]-cache['gaps'][:len(pairs)]))]))"
b="            geometric_clearance_violation_m=float(np.max(cache['limits'][:cache['geometric_gap_count']]-cache['gaps'][:cache['geometric_gap_count']])),command_incremental_clearance_violation_m=float(np.max(cache['limits'][cache['geometric_gap_count']:]-cache['gaps'][cache['geometric_gap_count']:])),previous_local_offset_peak_rad=float(abs(offset).max()),most_active_clearance_pair=str(cache['constraint_labels'][int(np.argmax(cache['limits']-cache['gaps']))]))";assert s.count(a)==1;s=s.replace(a,b);(B/'rough_planner.py').write_text(s,encoding='utf-8')
r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));r['focused_corrections']['runtime_change']='DirectController sets previous sent total minus current geometric C as a fixed local offset. RoughPlanner retains all original geometric constraints and adds the SAME measured-plane relative compression constraint as the final total-command filter, plus2mm command node motion.';r['command_aware_clearance']['command_constraints']='Measured planes min(-0.8mm,previous_command_gap-0.1mm); extra2mm command-node motion. Original source-original collision remains in geometric checks and final common filter.';r['command_aware_clearance']['preload_baseline']='Existing virtual penetration of servo equilibrium command is preserved as the filter already does; no absolute command collision requirement is added.';(B/'REGISTERED.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Updated unfrozen V6.6; no integration')
