# 曲线段拼接与管内分支 rolling：文献检索

日期：2026-10-03。用途：为实机研究方向②“曲线段拼接 + 顺应性控制”提供原始文献和下载入口。本次只跟进进度、核对文献与记录来源，没有修改控制代码或开展实验。

## 0. 用户澄清后的检索目标与直接匹配文献

用户要的是**运用曲线段拼接，实际实现管外轴方向突然转90°的跨管运动**，不是只介绍拼接基础、过法兰或管内相近应用的论文。前一轮将2018和2021论文放在主要位置，匹配不够准确；下面两项为本轮按方法及实验核对后的结果。

### 0.1 两段螺旋加连接段：2023年硕士论文

**卢精文：《蛇形机器人杆件攀爬跨越运动控制研究》**，中国矿业大学，2023年5月，导师胡而已。

- 仓库已有全文：`references/2023_蛇形机器人杆件攀爬跨越运动控制研究.pdf`。本轮未找到可确认的公开网络下载地址，不编造知网下载链接。
- 第4.4节（正文53页起，PDF第68页起）明确设置主枝、侧枝两杆轴线夹角为90°，设计同侧及异侧螺旋分叉跨越。
- 第4.4.1节的主干曲线由五段组成：**螺旋—圆弧—直线—圆弧—螺旋**。第1、5段分别缠绕原杆与目标杆，第2、4段平滑改变切向，第3段连接。表4-5给出各段曲率、挠率、长度和连接扭转角；此同侧构型的两段螺旋旋向相反。
- 运动不是简单让两段各自rolling：先探出、rolling靠近接头，再逐渐恢复前部圆弧和螺旋的曲率以包绕目标杆，随后用切向滑动迁移身体。第5.5.1、5.5.2节（正文77–79页）和图5-25、5-27给出同侧、异侧实机跨越；本轮已渲染并检查图5-25（PDF第93页）。
- 这项最直接匹配用户的“两段螺旋+连接曲线搭上目标管”设想。实验装置是90°分叉杆件，不能把它宣称为特定45°斜切L接头的相同工况，也不能把混合动作描述成全程纯rolling。

### 0.2 拼接曲线并rolling转入分支：2024年期刊论文

**S-shaped rolling gait designed using curve transformations of a snake robot for climbing on a bifurcated pipe**，Jingwen Lu、Chaoquan Tang、Eryi Hu、Zhipeng Li，Bioinspiration & Biomimetics，19，036010，2024。

- [DOI/出版社入口](https://doi.org/10.1088/1748-3190/ad3601)。
- 仓库已有出版全文：`references/2024_S-Shaped_Rolling_Gait_Designed_using_Curve_Transformations_of_a_Snake_Robot_for_Climbing_on_a_Bifurcated_Pipe.pdf`。本轮未确认免费的出版社PDF下载地址。
- 第2.3节处理曲线段拼接处的扭转角，第3节构造九段S形曲线。第4.2节及图21（PDF第15页）展示从主干管转入分支管的实机实验；图18–20主要是继续沿主干、避开支管，须与转入支管区分。
- 它是管外、拼接曲线、rolling转管的直接实验文献，但使用S形构型，不能当作“两段标准螺旋加桥接段”的同一种方法。正文明确存在失败工况，不支持任意方向急弯均可通过的结论。图21标题显示水平主干到向上支管；临近段落的轴向描述有不一致，后续复现需结合图和补充视频核对。

## 1. 当前进度与研究设想

- 已读取最新交接记录、实机调试日志第6点及局部螺距自适应方案。最新代码提交为 `43b8cfa`；直角L、水平/竖直T、U型等14类新场景已搭建，20/26关节共28个模型只完成静态验证，尚未验证攀爬。
- 实验室管道包胶厚度取暂定值1.5 mm，裸管110/160 mm对应仿真外径113/163 mm；摩擦仍为未标定的估计值。
- 用户实机记录：包胶后直管上顺应性电流较低，横向110→160 mm变径通过；原实机顺应性在急弯处不能可靠地重新缠绕目标管。上述是现场观察，不等于完整弯曲轴方法已在实机验证失败。
- 方向②包括：根据前部反馈探索接头方向，以连接曲线引导前部搭上目标管，分别围绕原管和目标管进行rolling，并加入顺应性。局部负载能否识别方向、继续rolling是否不过载、反向缠绕是否能转管，均尚未验证。

## 2. 曲线段拼接的基础方法：2018年T-RO

**Gait Design for a Snake Robot by Connecting Curve Segments and Experimental Demonstration**

- 作者：Tatsuya Takemori、Motoyasu Tanaka、Fumitoshi Matsuno。
- 出处：IEEE Transactions on Robotics，34(5)，1384–1391，2018。
- DOI：[10.1109/TRO.2018.2830346](https://doi.org/10.1109/TRO.2018.2830346)。
- [大学机构库页面](https://uec.repo.nii.ac.jp/records/9078)。
- [机构库PDF下载（作者稿）](https://uec.repo.nii.ac.jp/record/9078/files/TRO_2018_edited_finalVer.pdf)。已通过本机HTTP请求确认返回PDF；网络检索工具访问机构库时出现限流，不代表文件不存在。
- 仓库已有出版版：`references/2018_Gait_Design_for_a_Snake_Robot_by_Connecting_Curve_Segments_and_Experimental_Demonstration.pdf`。

核对原文第I–III节：方法连接曲率、挠率已知的曲线段，并处理连接处Frenet–Serret标架的扭转，使扩展后的Yamada方法可计算关节角。该论文的实验示例是管外过法兰和粗糙地面履带步态，不能将其称为已完成急弯跨管实验的论文。它是方向②的数学基础。

## 3. 更直接的复杂管道应用：2021年Frontiers

**Unified Approach to the Motion Design for a Snake Robot Negotiating Complicated Pipe Structures**

- 作者：Mariko Inazawa、Tatsuya Takemori、Motoyasu Tanaka、Fumitoshi Matsuno。
- 出处：Frontiers in Robotics and AI，8:629368，2021。
- DOI：[10.3389/frobt.2021.629368](https://doi.org/10.3389/frobt.2021.629368)。
- [出版社开放全文](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.629368/full)。
- [出版社PDF下载](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.629368/pdf)。
- [出版社文件服务器PDF（本次成功读取，18页）](https://public-pages-files-2025.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.629368/pdf)。

核对第3.2、4、5.1节：目标形状由两段螺旋和连接部分构成，连接部分包含引导段与按障碍设计的避让段。第5.1节直接给出分支和弯管的形状设计。通过shift control与rolling协调保持连接部分相对障碍的位置和方向；不同螺旋参数还涉及滑移模型。操作者提供运动和方向调整输入，不能描述成依靠关节负载自动识别方向的控制器。

这篇与“两条轴上的螺旋通过连接曲线过渡”构想相关，但主要面向管内复杂结构，不是本轮要求的管外急弯跨越直接验证。第4节的rolling/shift协调和第5.1节可作补充参考，不能替代第0节两项管外实验文献。

它的会议前身为 **Motion Design for a Snake Robot Negotiating Complicated Pipe Structures of a Constant Diameter**，ICRA 2020，8073–8079，DOI：[10.1109/ICRA40945.2020.9197224](https://doi.org/10.1109/ICRA40945.2020.9197224)。本次未核验该会议版PDF下载，只提供识别信息。

## 4. CMU管内网络运动：2016年JFR

**Pipe Network Locomotion with a Snake Robot**

- 作者：David Rollinson、Howie Choset。
- 出处：Journal of Field Robotics，33(3)，322–336，2016。
- DOI：[10.1002/rob.21549](https://doi.org/10.1002/rob.21549)。
- [Wiley论文页面](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21549)。
- [Wiley官方PDF入口](https://onlinelibrary.wiley.com/doi/pdf/10.1002/rob.21549)。

出版社标注在线首发为2014-11-11，卷期为2016年5月，因此2014和2016两种年份可能出现在检索记录中，通常按2016卷期引用。

首次检索的访问限制：PDF入口在未认证网络检索中返回摘要页，当时未取得该期刊论文全文，也未找到可确认的免费期刊全文。OpenAlex在该次查询中标记closed，且未列出机构库全文；这不是“任何地方都没有免费版本”的证明。可通过学校图书馆Wiley访问权限下载。2026-10-03后续已找到并读取本机Downloads中的期刊原文，具体机制见下文。

摘要介绍的是在低维步态参数空间进行形状估计和全身顺应控制。后续正文核对确认其使用螺旋步态加局部平面弯曲模式，不是2018/2021式的显式曲线段拼接，也未给两段螺旋分别设置独立rolling相位。

免费补充来源：作者2014年博士论文 **Control and Design of Snake Robots**。

- [CMU官方页面](https://publications.ri.cmu.edu/control-and-design-of-snake-robots)。
- [CMU官方博士论文PDF](https://publications.ri.cmu.edu/storage/publications/pub_files/2014/6/d_rollinson_robotics_2014.pdf)。已确认返回PDF，约21.6 MB；网络阅读工具因文件过大未解析。本次未精读博士论文。
- 该文件是博士论文，不能替代或冒充2016年JFR期刊全文。

### 4.1 期刊原文核对：如何通过分支管

来源：本机 `C:/Users/Li/Downloads/Journal of Field Robotics - 2014 - Rollinson - Pipe Network Locomotion with a Snake Robot.pdf`，15页，正文322–336页；以下结论来自该期刊原文第3.1、3.2、3.3.3、4.3节，而非仅根据摘要或博士论文推断。文件名2014与在线首发年份一致，卷期引用仍为2016。未将出版社PDF复制到仓库。

**允许身体转弯的是步态形状设计；顺应性让已有的弯曲模式适应管道并沿身体传递。** 第3.1.2节公式(4)–(5)在原螺旋正弦步态上加入局部高斯弯曲，同时削弱该处的螺旋振荡：

\[
A_l=A[1-\varphi(\mu,2\sigma,l)],\qquad \xi=2\pi(\gamma+\nu l),
\]

\[
\bar\theta_l=d_l\begin{cases}
A_l\sin\xi+\cos\phi\,A_\kappa\varphi(\mu,\sigma,l),&\text{lateral},\\
A_l\sin(\xi+\pi/2)-\sin\phi\,A_\kappa\varphi(\mu,\sigma,l),&\text{dorsal}.
\end{cases}
\]

- `l`是身体归一化位置，头部0、尾部1；`d_l`是对应关节长度因子。`γ`驱动滚动，`ν`控制螺旋空间频率。
- `μ`定位弯曲区域，`σ`决定宽度，`Aκ`决定弯曲幅度，`φ`决定弯曲方向。高斯窗口中心值为1，所以弯曲中心的基础螺旋幅度降为0；窗口之外逐渐恢复原螺旋。图5展示头部弯曲和两类关节角变化。
- 从运动学上，局部弯曲改变后续链节的朝向，使弯曲两侧可以形成朝向不同的螺旋部分；无需整身围绕同一根直轴。此处“两侧轴不同”是形状的几何解释，论文没有先拟合两条管轴并分别控制它们。

第3.2节以实测关节角为观测量，用EKF估计实际步态参数及其变化率。第3.3.3节公式(20)将螺旋幅度、弯曲幅度和弯曲位置分别设为当前估计加正偏移：

\[
A^{cmd}=\hat A+\rho_1,\qquad A_\kappa^{cmd}=\hat A_\kappa+\rho_2,\qquad\mu^{cmd}=\hat\mu+\rho_3.
\]

上式的帽号用于明确表示估计值，原文公式直接用参数符号表示当前估计。正偏移分别用于维持管壁接触、推动弯曲及让弯曲向尾部传递；目标基于实际估计持续更新，而非预先按时间强制推进到某个完整形状。这里不以关节扭矩传感作为顺应反馈，不能解释成根据负载自动识别T型分支。

实际转弯流程（第3.1.2节、第4.3节、图12–15）：操作者选择目标方向，在头部初始化弯曲；机器人继续驱动螺旋运动，弯曲幅度和位置顺应管道约束；随着前部进入目标管，弯曲区域逐渐移向尾部，最终整身通过。`φ`按IMU估计的朝向更新，使弯曲保持在所选环境方向，避免随着身体滚动转向其他方向。表II将`γ`设为恒速推进，`ν`和`σ`固定，`A`、`Aκ`、`μ`顺应控制；不要把第3.3.2节另外演示的相位顺应控制混入这组管内过弯实验。

适用范围：管内由周围管壁提供约束与支撑，后部保持螺旋接触不等同于锁定后部关节。原文明确提到过接头时有显著滑移，稳定绕管轴滚动会暂时受扰；不能据此宣称全程无滑移rolling或保证管外悬空转管。迁移到当前管外急弯任务时，局部弯曲模式、方向补偿和反馈传递值得借鉴，但目标管重新包绕与支撑稳定性仍需单独验证。

## 5. 对当前研究的含义

2018年的拼接方法、2021年的管道形状与运动协调、2016年的步态参数顺应控制应分别核对。已有“两段螺旋+连接曲线+rolling”文献，因此这个结构本身不能直接宣称为新贡献。将其与2023顺应性结合后是否有贡献，需要明确新增反馈规则、适用范围和对照结果。

本次没有下载新PDF到仓库，没有更改实机实验原文，也没有证明任何候选方法能通过当前管外非连续急弯。
