# -*- coding: utf-8 -*-
"""Record current ownership and the user's stricter sensor boundary."""
from pathlib import Path
import json
from datetime import datetime

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
note='''**2026-10-15 最新执行边界（覆盖下方旧监督安排）：** 用户停止执行会话，要求本对话直接开发 Step3；cmu 心跳已暂停，不再派发执行会话。Step3 必须从正常全源管包绕爬行开始，自行选择并调整出管位置，连续尝试搜索、接触定型与转移。用户明确控制器仅能使用舵机实际角度及电流/负载回读；不允许读取机体世界位姿、接触点/法向、目标标签或几何真值，也不把仿真初始位置传给控制器。仿真只模拟这些舵机通道并供独立评价，负载换算需与硬件标定分开。新增材料在 `experiments/u03_global_curve_compliance_20261012/STEP3_SERVO_ONLY_20261015/`；先前 `STEP3_ACTIVE_SEARCH_20261015/` 的真值接触估计仅保留为未采用原型，未启动动力学。原 Step2 工作不修改。保留用户周额度恢复即保存停止的边界，由本对话在开发节点查询；停止不等于完成。正式四次配对前先做必要开发诊断，尚未冻结或启动正式批次。\n\n'''
for rel in ['docs/实验记录/2026-10-12_U03曲线基线与局部顺应_执行任务.md','docs/实验记录/2026-10-14_全局曲线与局部顺应跨管_完整后续实验方案.md','docs/实验记录/统一顺应跨管_实验进度与接手说明.md','交接记录.md']:
 p=ROOT/rel
 if p.exists():
  text=p.read_text(encoding='utf-8-sig')
  if note.splitlines()[0] not in text:
   first,sep,body=text.partition('\n');p.write_text(first+'\n\n'+note+body,encoding='utf-8')
p=HERE.parent/'SUPERVISION/STEP3_RUN_CONTROL_20261015.json'
state=json.loads(p.read_text(encoding='utf-8-sig'))
state.update(owner='current root thread directly implements Step3',step3_state='executor_stopped_user_request',step3_cursor='85bfb2b0-ed71-451c-9ccf-e4af24e6de09:2',heartbeat_status='PAUSED',quota_monitor_active=False,quota_stop_boundary_active=True,monitoring_mode='direct_root_at_development_boundaries',stop_file=str(HERE/'STOP_REQUESTED.json'),active_directory=str(HERE),post_stop='Save and review completed evidence only; no automatic heartbeat restart',updated_at=datetime.now().astimezone().isoformat())
state['quota_target']['last_used_percent']=37
p.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
print('Ownership and servo-only boundary recorded.')
