> 本文保留第二阶段的历史操作说明与协议字段。当前执行链和默认路径已更新，运行请以[README](README.md)及[阶段四协议](COMMUNICATION_V3.md)为准。

# 第二阶段双向通信：v1.3.0

本阶段实现完整几何反馈、整组目标回传、Python 后台交换和 Adaptive Pole 预览接入。没有加入顺应性变形；没有操作真实串口或实体舵机。v1.2.1 发布目录保留。

## 使用

1. 停止旧可视化 EXE，再启动 `蛇形机器人可视化程序_v1.3.0\SnakeRobotVisualizer.exe`。同一端口只能有一个服务端；启动时保持空闲，不自动连接串口。
2. 在 `mujoco` 环境运行原代码目录中的 `Adaptive_Pole_Climbing.py`。程序上传实际 qpos，同时后台读取完整几何并回传 Yamada 目标用于预览。
3. 在 EXE 勾选“目标姿态”可看到回传角度重建的目标骨架。MuJoCo 的实际控制仍只由 Python 的 Yamada 循环写入 d.ctrl。
4. 结束脚本时会取消本连接目标并关闭连接；不会释放或使能真实电机扭矩。

Adaptive 脚本从 `Paper Code\two_joint_visualizer\python_client.py` 加载共用客户端，避免复制两套代码。若移动目录，可设置 `SNAKE_CONTROL_CLIENT_DIR`。端口默认 8765，客户端与 EXE 均可使用 `SNAKE_VISUALIZER_PORT`。`SNAKE_GEOMETRY_EXCHANGE=0` 只关闭新增双向交换，保留原单向桥；`SNAKE_VISUALIZER_ENABLED=0` 关闭全部可视化通信。

可单独执行 `python python_client.py geometry` 读取当前完整反馈。服务空闲、停止或尚无数据时，明确返回 no_frame；不会把旧缓存作为运行中的反馈。

## 职责与单位

- EXE：实际角采集、FK、轴与二维连续曲线、目标预览、唯一串口执行后台。
- Python 客户端：连接/协议/最新帧；不计算曲率，不控制 MuJoCo 物理步进。
- Adaptive Pole：阶段 1 Yamada 运动生成；本阶段读到的截面暂不用于改变目标。
- 几何长度 mm；关节角 deg；圆弧转角 rad。MuJoCo ctrl 的 rad 和奇偶方向由 Adaptive 中的桥接映射转换后再提交。

## 协议

本机 TCP `127.0.0.1:8765`，UTF-8，每个 JSON 对象以换行结束。持久连接支持多次请求，严格一问一答；同一 Python 客户端使用锁串行化请求。服务端处理 TCP 半包/粘包，不执行缺少换行的半包。旧 `state`、`target`、`limits`、`release`、`mujoco_frame` 保持兼容。

### geometry

请求：`{"cmd":"geometry"}`。

返回主要字段：

| 字段 | 含义 |
| --- | --- |
| protocol_version | 2 |
| run_id、config_id | 本次运行配置令牌；每次启动/重建配置都会变化，即使关节数相同也不沿用旧令牌 |
| sequence | 运行内递增帧号；配置变化后从新运行开始 |
| source、mode | 数据源与运行模式 |
| sample_unix_ms | MuJoCo 为发送方采样时刻；未提供时用接收端时间；实机为本轮采集完成后的快照时间 |
| published_unix_ms、age_ms | 几何发布时刻、当前反馈年龄 |
| config | 头段长、机械顺序、轴类型、舵机 ID、零位、方向、关节范围和连杆长 |
| angles_deg、online、joint_sample_unix_ms | 实际角、在线状态、各关节最近有效接收时间；未知为 0 |
| targets_deg、target_points_mm | 这一帧用于显示的目标及其 FK 骨架 |
| points_mm、projected_mm | 同帧实际三维骨架和二维投影 |
| projection_axis | 生成该帧曲线时实际使用的轴点、轴向、水平/竖直基与有效性 |
| raw_axis | 同帧原始轴估计，不与显示轴混用 |
| curve | 完整连续截面 |
| feedback_usable | 年龄≤500 ms、全在线、轴有效且非后备、曲线有效；仅表示基本可用，不保证接触稳定或近似误差足够小 |
| target_state | 查询时刻的命令生命周期状态，独立于采样时刻 |

当前每帧同时返回配置，避免配置请求与几何请求之间切换造成歧义；Python 可以按 config_id 缓存。几何始终取自单个完成帧，不使用 UI 绘制过程中的中间量。实机多舵机仍是顺序采样，整帧不代表所有编码器物理上同一瞬间采样。

`curve` 包含算法版本、valid/status、gamma_min_rad、modified_count、max_midpoint_error_mm 及 segments。各段保留身体顺序，包含：

- segment_index：从 0 开始的段序号；
- kind：line 或 arc；
- joint_index：内部段关联的关节序号，从 1 开始；首尾端段为 null；
- case：论文 A/B/C/D 或端部；
- radius_mm、turn_rad、length_mm：局部圆弧半径、带方向转角和段长；直线半径为 0；
- start_mm、tangent：投影平面内起点与单位切向。

D 类近似偏差原样返回，不能作为管壁精确测量；无效几何仍可读取诊断信息。本阶段目标预览不要求 feedback_usable=true，因为也要支持直线等轴不可观测形状的通信调试。后续顺应性控制需自行决定何时暂停几何更新。

### target_frame

```json
{
  "cmd": "target_frame",
  "execution": "preview",
  "run_id": "从反馈复制",
  "config_id": "从反馈复制",
  "feedback_sequence": 123,
  "command_id": 1,
  "valid_for_ms": 500,
  "targets_deg": [10.0, -20.0]
}
```

上例仅表示两关节结构，实际必须一次提供配置中的全部关节。command_id 是本连接递增正整数；连接由服务器分配身份，客户端不能指定/冒充身份。valid_for_ms 为 20–2000，截止时间是**所依据反馈的采样时间＋有效期**，不是服务器收到命令后再重新计时。只保留最近 64 个反馈帧时间供关联检查。

先检查整个目标的配置、反馈帧、命令序号、数量、数值与范围，然后整体接收。错误返回 `ok:false,error:...`，不局部写入也不静默限幅。

- preview：只影响后续采集帧的目标姿态，不更改实际角、不访问总线。在下一帧到来时应用；没有新反馈时不能声称已经显示。
- live：要求当前 Live 模式、手动扭矩使能、全部关节在线。唯一串口后台批量发送。只完成了模拟总线测试，真实舵机留待第四阶段。

后台只有一个最新待处理目标槽；新目标替换尚未执行的旧目标。已发送给舵机的指令无法撤回。一次只允许一个连接持有整帧目标输出，其他连接可读但不能抢写；占用期间拒绝旧单关节目标写入。过期、取消、断连和停止清除待处理/活动预览。实时过期仅停止下发新目标，不自动释放扭矩，也不能停止舵机执行最后已收到的目标；实机失联策略留给第四阶段确认。

回复 `status:accepted` 只表示接收。`geometry.target_state` 提供：

- accepted_command_id；
- applied_preview_command_id；
- dispatched_command_id（不表示舵机到达）；
- pending_command_id、replaced_pending_count；
- status：none/accepted/applied_preview/dispatched/cancelled/expired/failed。

`cancel_targets` 取消本连接目标；`release` 在原有释放行为前清除整帧目标。关闭连接也会取消其目标。停止运行清除缓存；重连必须重新获取新反馈，不自动重发旧写请求。

## Python 入口

```python
from python_client import SnakeControlClient

with SnakeControlClient() as client:
    geometry = client.get_geometry()
    # 使用当前目标作无运动预览示例；不会使能或访问电机。
    ack = client.submit_targets(
        geometry, geometry["targets_deg"], command_id=1,
        execution="preview", valid_for_ms=500)
    # 保持连接和采集运行，后续 geometry 可观察 applied_preview 状态。
    client.cancel_targets()
```

`GeometryExchange` 是 Adaptive 使用的后台客户端：10 Hz 获取几何，保存最新反馈，接受最新待发送预览向量；主循环调用 publish_targets 不等待网络。latest_geometry 返回副本，避免调用者修改后台缓存。重连/配置变化清空缓存，停止后取消目标。它仅支持 MuJoCo preview，不提供隐式实机控制。

## 验证及边界

- `windows_app/CommunicationTests.cs`：内部几何逐段完全一致、28 关节目标/FK/世界坐标、非法输入原子拒绝、最新替换、过期/取消/配置切换、模拟总线整组下发及 ID/编码器映射。
- `validation/communication/verify_stage2.py`：隔离 Windows EXE 和端口 18779；持久 TCP、JSON 分包/粘包、配置长度改变、断连重连、10 Hz、Adaptive Pole 实际集成。
- `windows_app/run-tests.bat`：回归使用隔离端口 18781，避免正在运行的旧步态发送器改变测试状态。

正式 EXE 测试导出 56 段；30 轮约 10 Hz 交换中，反馈年龄最大约 4 ms，上传＋读取几何＋提交目标往返最大约 16 ms（本机这次测量，不是实时性保证）。Adaptive Pole 5 秒收到 49 次几何、提交 46 次预览，通信失败为 0。数据见 `validation/communication/results/summary.json`、`adaptive_pole.log`。

回归 CoreModel、PaperGeometry、Communication、Feature、ProtocolPacket、CameraReference、UiSmoke、Soak 通过。端口隔离后的 UI 分屏 3D/2D 各 112 次绘制，约 10 Hz；15 秒 soak 151 帧。旧 v1.2.1 EXE 和原 Pole 保留哈希。

未验证：真实串口/舵机、稳定动态爬管、顺应性算法、接触力控制。目标预览经过正式 EXE 的 FK 和显示链，不代表实机执行验收。
