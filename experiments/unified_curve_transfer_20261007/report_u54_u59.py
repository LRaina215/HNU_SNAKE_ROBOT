from pathlib import Path
import json

base = Path(__file__).resolve().parent
notes = {
54:'67.55秒原局部承点不足停止。3.45秒M18减载仍接触转feed，参考写入连续；3.45–5.90秒参考shift共增1.0375mm（减速尾段超过名义1mm0.0375mm，U55精确限制修正），固定前/后净−0.655502/+1.333451mm。同批源/目标实际轴向+1.176449/−0.464500mm，不是双侧送身成功。12.45秒真实M18离管单独记录。冻结审计ctrl误差0；5秒M1请求/求解+0.017915/+0.044482mm/秒，实际根+派发关节−0.334737，实际−0.315190；未执行自由根变量掩盖反送。',
55:'2.30秒no_supported_distinct_rear_release_after_regrip。全执行任务减去实测根运动，真实源/目标动作未形成双侧前进。但旧流程把未离管的M18持续接触误认重新抓回，提前转恢复/新释放检查；两秒结果不能单独判定速度层方案无效，后续U60贯通语义。',
56:'2.05秒no_supported_distinct_rear_release_after_regrip。coordinated_feed_active同时保持当前负载/形状曲率与外移调形，原8mm/s更新及12mm总限幅不变。0.95秒未离管M18被旧恢复流程误当返回，1.95秒确认后无新候选停止；不能单凭这一短轮判定并行调形或实测根求解无效。',
57:'1.90秒no_supported_distinct_rear_release_after_regrip。同一曲线任务用原模型50ms实际关节ctrl私有预测作有限修正，根由预测动力学自然演化；原实际支持恢复衔接仍错误。预测内的控制残差改善不是实际送身证据，未有双侧前进或新增接管。',
58:'0.50秒controller_exception。原模型预测加入原2源/3目标承点阈值和双侧实际轴向前进约束；极小所选关节允许区间在端点选择了零有限差分步，派发前SVD异常。失败提案未执行，U59从U54有效完整点重跑，没有用失败后部分已更新参考当接续依据。',
59:'2.00秒no_supported_distinct_rear_release_after_regrip。有限差分方向取有实际可用区间的一侧，数值异常解除；同一动力学支撑/前进约束保留。协调feed未离管持续接触仍被旧返回流程误结束，U60同时修复当前动作恢复/续送语义。'
}
for number,note in notes.items():
    case = base/f'U{number}'
    result = json.loads((case/'result.json').read_text(encoding='utf8'))
    plan = json.loads((case/'plan.json').read_text(encoding='utf8'))
    (case/'BASIS.txt').write_text(f'U{number}：完整实际状态分支：'+plan['initial']+'；'+str(plan.get('controller_state'))+'\n'+plan['change']+'\n',encoding='utf8')
    metrics=(f"实际{result['physical_seconds']:.2f}秒，{result['outcome']}，complete=False。固定前/后净"
        f"{result['fixed_front_M0_M6_mean_y_advance_mm']:+.6f}/{result['fixed_rear_M14_M20_mean_x_advance_mm']:+.6f}mm；"
        f"最大关节{result['max_actual_joint_deg']:.6f}°、力矩{result['max_actuator_torque_Nm']:.6f}Nm、管穿深{result['max_pipe_penetration_mm']:.6f}mm。\n")
    (case/'RESULT.txt').write_text(f'U{number}：'+note+'\n'+metrics+
        'M14重接触为历史已迁移材料；无M15以后新增接管，整身跨越未完成。只有正式派发关节ctrl后的原模型动力学计作真实证据；私有预测/恢复复放不算新实际试验。历史物理计量不重置，压力目标/驱动/形状上限保持。冻结源码与实际视频/轨迹/状态保存，大文件仅留本地。最终仍需同一最终版从CLOSE1B完整整身运行。\n',encoding='utf8')
headline='截至2026-10-08累计60轮实际动作已启动，U01–U59已结束，U60协调减载feed/动力学执行实际运行中；U41一次0秒启动失败不另计实际轮次。U46确认M20源抓回1.05秒，U47参考数值写入修复；U53保住M20并真正抬离M18。U54共同推进探测固定前/后−0.656/+1.333mm，仍非双侧送身。U55–U59旧恢复流程把未曾离管的持续接触误当返回而短停止，不能据此单独否定速度层/并行调形；U60保留当前动作/参考/进度恢复替代支撑再续送。动力学预测只用于实际ctrl修正，无新增材料前沿，整身跨越未完成。'
for name in ['STATUS.txt','RESULTS.txt']:
    path=base/name;lines=path.read_text(encoding='utf8').splitlines()
    index=next(i for i,line in enumerate(lines) if line.startswith('截至2026-10-08累计'))
    lines[index]=headline;path.write_text('\n'.join(lines)+'\n',encoding='utf8')
print('Wrote U54-U59 reports and running U60 status')
