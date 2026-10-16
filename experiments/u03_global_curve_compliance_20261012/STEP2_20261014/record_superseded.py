# -*- coding: utf-8 -*-
from pathlib import Path
import json
base=Path(__file__).resolve().parent;archive=base/'R1_SUPERSEDED_STARTS';rows=[]
for name in ['T_FROZEN','T_UPDATE','P_FROZEN','P_UPDATE']:
    result=json.loads((archive/name/'result.json').read_text());rows.append(dict(case=name,**result,excluded_from_final_R2_comparison=True,exclusion_reason='Stopped for supervisor-directed contact-role separation and safe proposal rejection; not selected by physical outcome'))
    assert result['stop']=='requested_stop'
archive.joinpath('EXCLUSIONS.json').write_text(json.dumps(dict(superseded_dynamics_starts=4,retained_all_results=True,results=rows),indent=2))
archive.joinpath('README.md').write_text('''# R1中止运行：完整保留，排除出最终R2对照

四次R1原定正式运行已启动后，监督指出接收载荷混入非源/接头不明接触，以及七次提议全异常时回退记录变量未初始化的问题。执行端通过stop_and_save.request正常停止，保存全部轨迹、视频、状态、命令、估计与更新日志及每次源码。

这是实现权限/拒绝保护的纠正，不依据物理成功或失败筛选，也不调整先验、初态、估计/更新增益、推进频率、物理、伺服或验收。R1材料和原R1_READY_FROZEN均保留。最终R2仍比较原定两例两版；总动力学启动数明确为8，其中4次监督纠正前中止、4次最终对照，不能只报告4次启动或把R1当作额外留出成功。
''',encoding='utf-8')
print('Retained all4 superseded requested-stop results:',[(x['case'],x['sim_time']) for x in rows])
