# 今晚动作与验证源码清单：2026-10-09

02:50之后新增或修改的动作源码10个，验证/运行源码12个，共22个。每轮真实动作使用该轮sources里的冻结执行版本，不能直接用最新工作区代码复现所有旧轮。新建/修改分类结合前夜间U103F冻结源码是否已有该文件，归档时一并保留下述既有依赖和原模型。路径均可直接点击。

| Category | File | Change |
|---|---|---|
| Action | [snake_control/adaptive_front_winding_transfer.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/snake_control/adaptive_front_winding_transfer.py>) | new during night window |
| Action | [snake_control/articulated_shift_transfer.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/snake_control/articulated_shift_transfer.py>) | modified existing pre-night file |
| Action | [snake_control/axis_wrap_transfer.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/snake_control/axis_wrap_transfer.py>) | new during night window |
| Action | [snake_control/compact_helix_transfer.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/snake_control/compact_helix_transfer.py>) | new during night window |
| Action | [snake_control/executable_chain_transfer.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/snake_control/executable_chain_transfer.py>) | new during night window |
| Action | [snake_control/explicit_winding_transfer.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/snake_control/explicit_winding_transfer.py>) | new during night window |
| Action | [snake_control/fixed_spatial_shift_transfer.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/snake_control/fixed_spatial_shift_transfer.py>) | new during night window |
| Action | [snake_control/global_transfer_route.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/snake_control/global_transfer_route.py>) | modified existing pre-night file |
| Action | [snake_control/moving_compliance_transfer.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/snake_control/moving_compliance_transfer.py>) | new during night window |
| Action | [snake_control/unfolding_transfer.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/snake_control/unfolding_transfer.py>) | new during night window |
| Runner / validation | [validation/check_compact_helix_geometry.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/check_compact_helix_geometry.py>) | new during night window |
| Runner / validation | [validation/check_executable_chain.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/check_executable_chain.py>) | new during night window |
| Runner / validation | [validation/diagnose_explicit_winding_stop.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/diagnose_explicit_winding_stop.py>) | new during night window |
| Runner / validation | [validation/diagnose_receiving_path.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/diagnose_receiving_path.py>) | new during night window |
| Runner / validation | [validation/review_adaptive_front_winding.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/review_adaptive_front_winding.py>) | new during night window |
| Runner / validation | [validation/review_compact_helix.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/review_compact_helix.py>) | new during night window |
| Runner / validation | [validation/review_executable_chain.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/review_executable_chain.py>) | new during night window |
| Runner / validation | [validation/review_explicit_winding.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/review_explicit_winding.py>) | new during night window |
| Runner / validation | [validation/review_fixed_spatial_shift.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/review_fixed_spatial_shift.py>) | new during night window |
| Runner / validation | [validation/review_moving_compliance.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/review_moving_compliance.py>) | new during night window |
| Runner / validation | [validation/review_source_unfolding.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/review_source_unfolding.py>) | new during night window |
| Runner / validation | [validation/unified_curve_transfer_experiment.py](<D:/Desktop/SNAKE/Paper Code/Mujoco/Snake_Robot_Gait/validation/unified_curve_transfer_experiment.py>) | modified existing pre-night file |

Existing dependencies to retain with the archive:

- snake_control/unified_curve_transfer.py
- snake_control/coordinated_curve_transfer.py
- snake_control/direct_curve_transfer.py
- snake_control/spatial_bishop_backbone.py
- snake_control/pipe_clearance.py
- validation/recover_unified_curve_state.py
- validation/recover_unified_checkpoint.py
- validation/hook_pull_experiment.py
- snake_control/hook_pull_review_support.py

Original model and its referenced mesh assets: `mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml and its original referenced assets`.

each N01W through N19C has sources for its execution version; latest working-tree controllers do not reproduce all older cases; post-run review helpers may be copied into sources, frozen action/runner never replaced

Large simulation, raw command and video artifacts remain local.
