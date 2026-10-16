"""Create a focused revision without modifying the preserved V3.2 case."""
from pathlib import Path
import shutil, json
P=Path(__file__).resolve().parent; old=P/'CONTROLLER_V3_POSITIVE_POLAR'; new=P/'CONTROLLER_V3_3_GRAVITY_SOURCE_CLASS'
if (new/'REVIEW_SNAPSHOT').exists() or (new/'T_UPDATE').exists():raise RuntimeError('Refuse to overwrite frozen or integrated revision')
new.mkdir(exist_ok=True)
for f in list(old.glob('*.py'))+[old/n for n in ['GIVEN_INITIAL.npz','KNOWN_SOURCE_COIL_CALIBRATION.npz','KNOWN_SOURCE_COIL_CALIBRATION.json']]:shutil.copy2(f,new/f.name)
f=new/'rough_planner.py';s=f.read_text(encoding='utf-8');before='            out[j]=-axis@moment';after='            # Fixed posterior bearing: increasing q rotates the free fore side negatively.\n            # Holding torque is dU/dq = +axis dot the fore-side gravity moment.\n            out[j]=axis@moment'
if s.count(before)!=1:raise RuntimeError('Unexpected gravity implementation')
f.write_text(s.replace(before,after),encoding='utf-8')
f=new/'contact_sensor.py';s=f.read_text(encoding='utf-8');before="    seam='straight_end_x_m' in source and abs(p[0]-source['straight_end_x_m'])<.005 and r<source['radius_m']+.005 and abs(n[0])>.65";after="    # At a known cylinder/cap edge the collision normal mixes both surface normals.\n    # Classify the known seam disk by its position, with the existing tolerances.\n    seam='straight_end_x_m' in source and abs(p[0]-source['straight_end_x_m'])<.005 and r<source['radius_m']+.005"
if s.count(before)!=1:raise RuntimeError('Unexpected source seam implementation')
f.write_text(s.replace(before,after),encoding='utf-8')
f=new/'register_direct.py';s=f.read_text(encoding='utf-8').replace("version='DIRECT_EXTENSION_V3_2_POSITIVE_POLAR'","version='DIRECT_EXTENSION_V3_3_GRAVITY_SOURCE_CLASS'")
s=s.replace('new_dynamics_starts=0,prior_dynamics_starts_preserved=20,','new_dynamics_starts=0,prior_dynamics_starts_preserved=21,prior_legacy_postwrap_starts_preserved=8,')
s=s.replace('Original source head initially points down','Original source head initially points down')
s=s.replace('current controller unreviewed.','current controller unreviewed. V3.2 one actual failed diagnostic preserved.')
s=s.replace("r['safety_stop']=", "r['focused_corrections']=dict(previous_actual_case='../CONTROLLER_V3_POSITIVE_POLAR/T_UPDATE',gravity='Holding torque equals fixed-bearing potential gradient; reverse erroneous gravity preload sign',source_class='Known straight-end seam disk uses existing 5mm geometric tolerance without excluding mixed edge normals',unchanged='Original initial, physics, gains, guards, probes, estimator and phase rules')\nr['safety_stop']=")
s=s.replace("revision_reason='Original given source pose retained. Receiving positive polar winding keeps source/receiver physical chirality and original rolling sign; restore source-native template radii, preserve phase-boundary C and actual probe evidence.'", "revision_reason='After one actual V3.2 source-drift failure, correct independently verified fixed-bearing gravity sign and known-source mixed-edge seam classification only.'")
f.write_text(s,encoding='utf-8')
design=(old/'CONTROLLER_DESIGN.md').read_text(encoding='utf-8');design=design.replace('DIRECT_EXTENSION_V3_2_POSITIVE_POLAR','DIRECT_EXTENSION_V3_3_GRAVITY_SOURCE_CLASS')
design=design.replace('当前版本为', '当前版本为',1)
design='''# V3.3 两项实现修正

前一版本 V3.2 已从物理 t=0 实际运行一次，5.075 秒触发源侧位置漂移保护，失败记录保留。当前 V3.3 尚未积分，仅申请单例 T_UPDATE 独立代码核对。

相对冻结 V3.2 仅修改：固定后侧承载端条件下，前部重力保持力矩符号改为重力势能对关节角的导数；已知 X=0.95 m 源管接缝按原有 5 mm 位置容差识别，不用端面法向阈值排除混合边缘法向。两项实际证据与只读核对见 ../V3_2_ACTUAL_DIAGNOSTIC.md 和 FOCUSED_CORRECTION_AUDIT.json。二者对实际漂移的个别贡献尚未隔离。

初态、原始物理、所有增益/速率/限值、探测、估计器、阶段切换规则与失败停止条件均不变；不利用真实接收几何识别在线接触。旧证据不构成本版动态成功。以下为继承的控制链说明，其中历史“未积分”仅针对其编写时版本。

'''+design
(new/'CONTROLLER_DESIGN.md').write_text(design,encoding='utf-8')
f=new/'prepare_review_snapshot.py';s=f.read_text(encoding='utf-8');start=s.index("    evidence=[");end=s.index("    limitations=",start)
s=s[:start]+"    evidence=['CONTROLLER_DESIGN.md','FOCUSED_CORRECTION_AUDIT.json','../V3_2_ACTUAL_DIAGNOSTIC.md',\n        '../CONTROLLER_V3_POSITIVE_POLAR/T_UPDATE/OUTCOME_SUMMARY.json','../CONTROLLER_V3_POSITIVE_POLAR/T_UPDATE/SOURCE_DRIFT_AND_GRAVITY_AUDIT.json'],\n"+s[end:]
f.write_text(s,encoding='utf-8')
print(str(new))
