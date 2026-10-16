# A13冻结记录

235秒原物理单次试验结束，经独立核查阶段A失败。前六稳定半圈与21杆稳定转移均无，目标直段同时最多7杆，末态源分类力6.789N。B未开始。

结果见OUTCOME.md，独立判定见independent_outcome.json。原始状态为trace.npz；模型、STL、初态、运行代码、积分和控制器检查点均保留。分析代码来自归档时中央版本。actual_side_oblique_1x.mp4由保存的实际状态生成，video_metadata.json确认4699帧及首/中/末读回。

复现运行需要把冻结运行.py复制到仓库experiments下一个新的中央目录，沿用其BASE/ROOT层级后从runner启动，选择全新case名。直接从本冻结子目录启动会改变路径层级。
