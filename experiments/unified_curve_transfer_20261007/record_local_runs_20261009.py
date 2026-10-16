# -*- coding: utf-8 -*-
"""Local evidence notes from existing results; no dynamics or reconstruction."""
from pathlib import Path
import json

BASE=Path(__file__).resolve().parent
DETAILS={
    'U94D':'旧速度任务仅加入低权重Bishop角度提示，不能作为完整精确整身拟合主链；已撤回为主方案。',
    'U97K':'整身拟合的位置目标直接下发，但旧实测速度启动门槛约1秒后卡住shift；后续压力形变不代表持续shift动作。',
    'U98K':'持续shared shift并保持初始空间负载分布。后期精确FK参考穿入管实体，候选被拒绝；不是实测速度启动门槛。只读接触运动学诊断未推进动力学。',
    'U99K':'真实支撑材料点约束精确FK拟合中的虚拟姿态配准，材料标架与滚动相位独立积分；整组位置目标直接下发。',
    'U100K':'每侧两个分开承点、增强积分先验、加载法向跟随误差修改同一曲线；目标支撑丢失，无接管。这是摩擦对照的冻结控制基线。',
    'U101K':'移动柔顺桥与前/后侧相位场的备选，因最新优先摩擦对照而请求保存停止。代码完整保留在本轮sources，未混入后续对照；后群有所前进但前群回退，无接管，不能认定成功。',
    'U102F':'第一项独立对照：控制源码与U100K冻结控制逐字节一致，CLOSE1B完整初态一致；仅运行时蛇—管合成滑动摩擦从2降到1，扭转/滚动摩擦保留，原XML不改。',
    'U103F':'第二项独立对照：与U102F同摩擦1、同起点，仅移除拟合切向近似不滑残差，法向约束保留；无移动柔顺区和侧向相位场。',
}
lines=[]
for name,detail in DETAILS.items():
    case=BASE/name
    if not (case/'result.json').exists():
        continue
    r=json.loads((case/'result.json').read_text(encoding='utf8'))
    evidence=(f"实际 {r['physical_seconds']:.2f}s，参考shift {r['final_shift_mm']:.3f}mm；"
              f"前群M0–M6平均+Y {r['fixed_front_M0_M6_mean_y_advance_mm']:.3f}mm，"
              f"后群M14–M20平均+X {r['fixed_rear_M14_M20_mean_x_advance_mm']:.3f}mm。")
    (case/'BASIS.md').write_text(f"# {name} 实际连续试验\n\n从CLOSE1B完整状态起，sources冻结版本，原模型和驱动限制不变，只下发20关节ctrl。未完成整身转移。\n\n{detail}\n\n结果：{r['outcome']}。{evidence}新持续目标接触模块：{r['new_sustained_target_modules']}。参考shift不等于实际送身。\n\n原始轨迹、指令、完整物理末态、控制末态和结果分别保存；save_errors={r.get('save_errors',[])}。实际1x视频由本轮qpos逐帧重放，没有拼接其它试验。大文件仅留本地。\n",encoding='utf8')
    lines.append(f"- {name}: {r['outcome']}。{evidence}新持续接触：{r['new_sustained_target_modules']}。{detail}")
header="""# 2026-10-09 执行方本地记录

最终要求同一控制版本从CLOSE1B完整状态连续实际完成整身转移。当前尚未完成。监督方负责公共文档和Git；执行方仅修改源码、仿真与本地材料。原XML、杆长、质量和驱动限制不改；最新授权的U102F/U103F仅在运行时将有效滑动摩擦降到1，其它摩擦不变。不连接实机，不重做静止保持，大文件留本地。

主链：共同shift移动材料弧长取样窗口；固定初始材料标架与连续滚动相位产生整组积分先验；原模型20交替关节精确FK拟合；连续位置目标直接限幅/限速后下发；实际压力、跟随误差和负载修改同一曲线与推进速度。虚拟姿态仅用于拟合、按支撑配准，实际root只由原动力学演化。前部接续已有回勾，后部同步送身，外侧平缓桥传递长度，无预舒展或预存拱阶段。目标螺旋接续已有前部包绕，不要求源/目标同手性。

U88C–U91C的预存/整体调形前置已撤回。U92D/U93D的M8原始接触发生于支撑丢失附近，不能认定有效交接。U95K/U96K均从CLOSE1B连续独立起跑，退出保存时ndarray序列化失败，原始trace保留但commands、末态与结果缺失；各轮incomplete_record_summary.json仅提取既有轨迹，不补跑，不伪造缺失历史。U97K起修复JSON并将轨迹/物理状态/控制状态独立保存；小数据检查不推进动力学。

当前源码：snake_control/articulated_shift_transfer.py。每轮sources冻结版本与该轮原始结果一起保留。参考推进不记为成功；最终必须满足实际全身进入同一目标螺旋并持续支撑。

当前实际运行均已结束。最新要求的两个摩擦/切向约束对照已完成，未完成整身转移。FRICTION_COMPARISON_20261009/comparison.json、comparison.png、feedback_and_video_evidence.json汇总原始数据。U102F有效摩擦1与U100K有效摩擦2几乎同轨迹；U103F移除切向残差仍未改善前部卷入。对现控制，本段未支持滑动摩擦过高或该切向残差是唯一主因的判断；不能据此断言任意其它曲线/步态对摩擦不敏感。移动柔顺方案保留为备选。

"""
(BASE/'CURRENT_20261009.md').write_text(header+'\n'.join(lines)+'\n',encoding='utf8')
print('Updated local notes from saved results only.')
