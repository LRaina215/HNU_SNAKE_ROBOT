from pathlib import Path
import json

base = Path(__file__).resolve().parent
notes = {
49: '替代承点不足立即转自由尾恢复，102.60秒原90秒实际进度不足停止。多次M20恢复后重新让位又失去，未进入feed；仅修检查不能完成协调。',
50: '先按测得/下一派发速率估计支撑补偿，但仍保留曲线角度后覆盖，设计随后撤回。97.60秒原90秒实际进度不足停止；17.65秒M15减载仍接触准入，24帧正shift共0.4375mm。该值是参考推进，不代表实际送身。',
51: '所选后部关节下发支撑求解同一增量，取消曲线角度后覆盖。97.60秒原90秒实际进度不足停止；M20恢复循环仍存在。M15减载准入但实际后退刹车使shift0，未形成协调送身。',
52: '同一求解/下发增量；暂时失触承重段也用安全小步回靠，原1秒连续损失与交接分别计量，原12秒动作期限不延长。5.50秒替代承点不足1秒停止。冻结重建ctrl误差0，J18首让位求解/下发4.532104469°/秒一致；M20回靠守卫未删，但旧角度误差仍追赶。',
53: '后部让位仅下发本次有限连杆参考增量，去掉旧角度差追赶，承点/压力/其他姿态任务保留。10.20秒原目标承点不足1秒停止。3.45秒M18减载准入7.927972N/初始8.976743N；9.00秒M18实际离管、M20仍承重，进入feed；共同shift0。feed转场旧姿态滚动再次要求前部约−3.3mm/秒，目标M3随后失触。U54连续参考及同一形变滚动/减载推进接续修正。'
}
for number, note in notes.items():
    case = base/f'U{number}'
    result = json.loads((case/'result.json').read_text(encoding='utf8'))
    plan = json.loads((case/'plan.json').read_text(encoding='utf8'))
    (case/'BASIS.txt').write_text(f'U{number}：从U47/action_t0p05_completed完整实际物理/控制器状态分支。\n'+plan['change']+'\n', encoding='utf8')
    metrics = (f"实际{result['physical_seconds']:.2f}秒，{result['outcome']}，complete=False。固定前/后净"
        f"{result['fixed_front_M0_M6_mean_y_advance_mm']:+.6f}/{result['fixed_rear_M14_M20_mean_x_advance_mm']:+.6f}mm；"
        f"最大关节{result['max_actual_joint_deg']:.6f}°、力矩{result['max_actuator_torque_Nm']:.6f}Nm、管穿深{result['max_pipe_penetration_mm']:.6f}mm。\n")
    (case/'RESULT.txt').write_text(f'U{number}：'+note+'\n'+metrics+
        'M14重新接触为已迁移材料，不是新前沿；M15以后无新增接管，完整整身跨越未完成。只有实际关节ctrl，历史物理计量不重置，模型/驱动不变。跨版本分支仅为中间证据；冻结源码/实际轨迹/原1倍速视频和完整末态保存，大文件本地。\n', encoding='utf8')
headline = '截至2026-10-08累计54轮实际动作已启动，U01–U53已结束，U54减载共同推进正在实际运行；U41一次0秒启动失败不另计实际轮次。U46确认M20源接触1.05秒并恢复有效完整状态；U47修复近重复弧长写入。U49仅守卫恢复循环不能协调；U52关节求解/派发增量一致，安全小步回靠承重段。U53保住M20并使M18实际离管，但feed旧姿态滚动反送、目标支撑失去，尚无实际共同推进或M15以后接管。U54从完整交接状态试验接触保留的减载推进/连续参考。完整整身跨越未完成。'
for name in ['STATUS.txt', 'RESULTS.txt']:
    path = base/name
    lines = path.read_text(encoding='utf8').splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('截至2026-10-08累计'))
    lines[index] = headline
    path.write_text('\n'.join(lines)+'\n', encoding='utf8')
print('Wrote U49-U53 reports and running U54 status')
