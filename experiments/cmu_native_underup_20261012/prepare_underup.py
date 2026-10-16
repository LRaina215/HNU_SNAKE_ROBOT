"""One directed reflection of N10, not a parameter grid or physical run."""
from native_run import *
m=mujoco.MjModel.from_xml_path(str(MODEL));A=1.2150592994538565;ws=.183889173151851;phase=2.7925268031909267;polar=.08234860970373083
up,s=placement(m,A,ws,phase,6,.95,polar);low,s2=placement(m,A,-ws,-phase,6,.95,-polar)
p=nodes(up);p_reflected=p.copy();p_reflected[:,2]=2-p[:,2]
reflection_error=np.max(np.linalg.norm(nodes(low)-p_reflected,axis=1))
init,pairs=snapshot(m,low);tangent=(init['points'][6]-init['points'][7])/LM
qerrors=[]
for t in [0,10,25,40,50,75,150,235]:
 q1,i=window(A,ws,phase,t,.02,6,.4,5,2.23,-1)
 q2,j=window(A,-ws,-phase,t,-.02,6,.4,5,2.23,-1)
 expected=q1.copy();expected[::2]*=-1
 qerrors.append(dict(t=t,maximum_joint_command_reflection_error=float(np.max(abs(q2-expected))),nt=i['nt'],L=i['L'],delta_N10=i['delta'],delta_U01=j['delta']))
out=dict(candidate='U01',N10_retained=True,preparation_only=True,physics_steps=0,initial_reflection_error_m=float(reflection_error),selected_tail_to_head=tangent.tolist(),saved_initial_depth_m=init['depth'].tolist(),initial_contacts=pairs,wave_reflection=qerrors,parameters=dict(A=A,ws=-ws,phase=-phase,polar=-polar,f=-.02,nb=6,seam=.95,rate_scale=2.23,delta_map=-1,low=.4,slope=5,slew=np.radians(20)),direction_hypothesis='Reflect N10 geometry and direct waves about Z=1. N10 free front rises over target before descending; reflected commanded geometry should approach from below and rise. Real gravity is unchanged, so acquisition/support must be verified by continuous dynamics.',unverified='Geometry/command reflection is exact; physical motion is not a reflected solution because gravity does not reverse.')
(BASE/'U01_PREPARATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
