"""Let the existing source moment objective use one head-preserving C mode."""
from pathlib import Path
import json,shutil
P=Path(__file__).resolve().parent;old=P/'CONTROLLER_V6_7_CONTACT_ENTRY_CHART';B=P/'CONTROLLER_V6_8_HEAD_NULL_BALANCE_MODE';B.mkdir(exist_ok=False)
for f in list(old.glob('*.py'))+[old/n for n in ['GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json']]:shutil.copy2(f,B/f.name)
r=json.loads((old/'REGISTERED.json').read_text(encoding='utf-8'));r.update(version='DIRECT_EXTENSION_V6_8_HEAD_NULL_BALANCE_MODE',dynamics_authorized=False,independent_initial_and_chain_review_passed=False,current_code_independently_reviewed=False,new_dynamics_starts=0,prior_dynamics_starts_preserved=33,prior_version='V6.7 actual stops32.125s at original source12mm guard; probeclock5.075s/second direction but head-only, nofit/Gwrite/capture',revision_reason='Strong C body-guide fitting competes with the existing source-axis moment objective. Free one physically available gravity-changing body-shape direction that preserves current head position/tangent to first order; keep all hard constraints and remaining body-guide modes.')
r['initial_definition']=r['initial_definition'].replace('V6.7 preserves','V6.8 preserves');r['focused_corrections']=dict(baseline='V6.7 prescribed contact chart and unchanged physical chain',runtime_change='Only RoughPlanner C body-position objective during beta>0: at each optimization start compute a fixed head-null gravity-changing native-joint direction; project its body-point displacement direction out of the body-guide cost and its analytic Jacobian.',mode_inputs='Known robot gravity/masses, private C native Jacobians and existing measured bearing. No target truth or G write.',unchanged='Existing source net-moment residual/scale/activation, head/tangent/coiling terms, all geometric/command constraints, native common filter, source grip/guard/phase/probe/estimator/capture and physical settings.',limitations='One relaxed soft body mode is not a proof of static equilibrium, actual source retention or multi-material contact.')
r['head_null_body_balance']=dict(enabled_in_wrap_only=True,head_null='SVD nullspace of current C head position and unit-tangent Jacobian',native_direction='Projection of exact fore-robot gravity source-X moment gradient into that nullspace',body_projection='I-ddT/(dTd), d=body-node Jacobian times that native direction; fixed during each solve',rank_tolerance_relative=1e-8,all_original_hard_constraints_retained=True)
(B/'REGISTERED.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(B/'register_direct.py').write_text('"""Prepared by ../prepare_v6_8_balance_mode.py; never reset a frozen case."""\n',encoding='utf-8')
f=B/'rough_planner.py';s=f.read_text(encoding='utf-8');at=s.index('    def solve(self,alpha,beta=0.')
method='''    def body_balance_projection(self,q,n):
        """Fixed soft-cost projector; all physical/geometric constraints stay."""
        identity=np.eye(3*(n-1));points=self.fk(q);jpa=np.zeros((3,self.m.nv));jra=np.zeros_like(jpa)
        mujoco.mj_jac(self.m,self.d,jpa,jra,self.snapshot['anchorP'],self.head+self.anchor);jpa=jpa[:,self.va[:n]];jra=jra[:,self.va[:n]]
        jp,jr=self.point_J(points[0],self.head,jpa,jra,n);T=self.d.xmat[self.head].reshape(3,3)[:,2].copy();head_J=np.vstack([jp,-skew(T)@jr])
        _,singular,Vh=np.linalg.svd(head_J,full_matrices=True);rank=int(np.sum(singular>max(1e-12,float(singular[0])*1e-8)));null=Vh[rank:].T;gradient=np.zeros(n)
        for i in range(n):
            force=self.m.body_mass[self.head+i]*self.m.opt.gravity;Jp,_=self.point_J(self.d.xipos[self.head+i],self.head+i,jpa,jra,n);gradient+=force[2]*Jp[1]-force[1]*Jp[2]
        direction=null@(null.T@gradient);norm=float(np.linalg.norm(direction));metrics=dict(body_balance_mode_available=False,body_balance_head_rank=rank,body_balance_mode_gravity_gain_Nm_rad=0.,body_balance_mode_head_J_residual=0.)
        if norm<=1e-10:return identity,metrics
        direction/=norm;body_J=np.vstack([self.point_J(points[i],self.head+i,jpa,jra,n)[0] for i in range(1,n)]);displacement=body_J@direction;norm=float(np.linalg.norm(displacement))
        if norm<=1e-10:return identity,metrics
        displacement/=norm;metrics.update(body_balance_mode_available=True,body_balance_mode_gravity_gain_Nm_rad=float(gradient@direction),body_balance_mode_head_J_residual=float(np.linalg.norm(head_J@direction)))
        return identity-np.outer(displacement,displacement),metrics
'''
s=s[:at]+method+s[at:];a="        pairs=source_pairs+legacy_pairs+target_pairs+self_pairs\n        def evaluate(qfront):";b="        pairs=source_pairs+legacy_pairs+target_pairs+self_pairs\n        body_projection,body_mode_metrics=self.body_balance_projection(previous,n) if beta>0 and self.p.get('source_axis_balance',{}).get('enabled',False) else (np.eye(3*(n-1)),dict(body_balance_mode_available=False))\n        def evaluate(qfront):";assert s.count(a)==1;s=s.replace(a,b)
a="""            for i in range(1,n):
                Jp,_=self.point_J(points[i],self.head+i,jpa,jra,n);weight=self.p['wrap_body_position_weight_per_m'] if beta>0 else .02/LINK
                residual.append(weight*(points[i]-geometric_targets[i]));jac.append(weight*Jp)
"""
b="""            body_error=(points[1:n]-geometric_targets[1:n]).ravel();body_J=np.vstack([self.point_J(points[i],self.head+i,jpa,jra,n)[0] for i in range(1,n)]);weight=self.p['wrap_body_position_weight_per_m'] if beta>0 else .02/LINK
            residual.append(weight*(body_projection@body_error));jac.append(weight*(body_projection@body_J))
"""
assert s.count(a)==1;s=s.replace(a,b);a='self.last_metrics=dict(accepted=accepted,solution_kind=';b='self.last_metrics=dict(**body_mode_metrics,accepted=accepted,solution_kind=';assert s.count(a)==1;s=s.replace(a,b);f.write_text(s,encoding='utf-8');print(str(B))
