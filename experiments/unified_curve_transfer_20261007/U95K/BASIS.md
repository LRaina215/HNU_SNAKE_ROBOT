# U95K 原始材料说明

本轮从 CLOSE1B 独立连续运行，使用 sources 冻结版本；实际驱动只接收20个关节位置指令。未完成整身转移。

退出时 commands.json 的 ndarray 序列化失败，已保存 trace.npz。末态、控制末态、指令与结果摘要缺失；不重跑本轮补材料，也不从轨迹伪造缺失控制历史。incomplete_record_summary.json 仅提取已有实际轨迹数值。

实际运行 62.90s，共同参考shift 9.623mm。固定前群M0–M6平均+Y -53.753mm，后群M14–M20平均+X 12.631mm。参考shift不是实际送身量。
