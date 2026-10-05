# 第2步：小幅主动试探与可观测性诊断

2026-10-05。继续原四步规划的第2步；这些诊断不是新增完整跨越结果。

## 实际做了什么

- 水平 `horizontal_encoder_live_whole_coil` 的0/16/32/40秒、竖直 `vertical_encoder_live_whole_coil` 的2/10/22/34秒恢复诊断初态。有/无管配对仅在独立负对照副本中关闭目标管碰撞，没有改变有管组摩擦、力矩或重力。
- 每次1.2秒：0.2秒保持，0.8秒同时施加三组互不重叠的0.6°扰动，再0.2秒返回。三组为前部后缘的一对正交关节与前部收拢，频率互异。前部范围来自先前估计的双轴归属，未使用真实接触编号。
- 角度和下发目标生成三组谐波响应，最小幅值为额外机械约束特征。真实接触/自由基座位姿列为 `evaluation_only`，不进入特征。
- 水平8个有/无管样本定阈值0.4865247591，竖直8个冻结复核均匹配。只区分本组管道存在性，不证明力封闭、承载裕量或方向正确，不能直接据此释放尾部。其他障碍也可造成低响应。
- 有管组峰值关节速度7.9–31.6°/s，头部净位移1.7–8.2 mm。无管对照存在快速滑落，不计为平顺运动。

## 文件与复现

- `probe_trials.json`：32次诊断（16次多频扰动＋16次保持对照），含逐帧输入、角度及评分列。
- `summary.json`、`classification.json`、`horizontal_rule.json`：特征、运动影响和冻结规则。
- `feedback_takeover_active_probe.py`：生成数据时源码。`protocol.json` 的竖直多频时刻已按实际记录校正，未改变命令或轨迹。
- `geometry_window*.json`：同窗口单帧/多帧CAD拟合，增大数值差分步长未稳定改善。
- `wrench_direction.json`：假设尾部静止、未知重力方向、只有法向反作用的方向扫描；未改善，不采用。
- 相邻 `active_takeover/`、`active_takeover_distinct/`：较早单方向试探及失败结果。
- `../feedback_takeover/pitch_axis_probe.json`、`fixed_radius_probe.json`：已有轨迹的几何先验诊断，没有物理闭环更新。

在仓库根目录，以新目录复现：

```powershell
python validation/feedback_takeover_active_probe.py --cohort distinct --mode multisine --output experiments/segmented_pole_20261005/active_takeover_multisine_repeat
```

脚本拒绝覆盖旧试探结果。直接运行MuJoCo诊断模型，不经过EXE，不计入19组闭环和128,830条转发命令。全部试探来自已有两条轨迹，尚无独立初态或硬件稳健性验证。
