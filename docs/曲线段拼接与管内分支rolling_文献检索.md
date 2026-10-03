# 曲线段拼接与管内分支 rolling：文献检索

日期：2026-10-03。用途：为实机研究方向②“曲线段拼接 + 顺应性控制”提供原始文献和下载入口。本次只跟进进度、核对文献与记录来源，没有修改控制代码或开展实验。

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

这篇最接近方向②的“两条轴上的螺旋通过连接曲线过渡”构想，但主要面向管内复杂结构，不能据此保证移植到管外急弯能维持支撑。建议优先读第4节的rolling/shift协调和第5.1节，再对接2023截面顺应性。

它的会议前身为 **Motion Design for a Snake Robot Negotiating Complicated Pipe Structures of a Constant Diameter**，ICRA 2020，8073–8079，DOI：[10.1109/ICRA40945.2020.9197224](https://doi.org/10.1109/ICRA40945.2020.9197224)。本次未核验该会议版PDF下载，只提供识别信息。

## 4. CMU管内网络运动：2016年JFR

**Pipe Network Locomotion with a Snake Robot**

- 作者：David Rollinson、Howie Choset。
- 出处：Journal of Field Robotics，33(3)，322–336，2016。
- DOI：[10.1002/rob.21549](https://doi.org/10.1002/rob.21549)。
- [Wiley论文页面](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21549)。
- [Wiley官方PDF入口](https://onlinelibrary.wiley.com/doi/pdf/10.1002/rob.21549)。

出版社标注在线首发为2014-11-11，卷期为2016年5月，因此2014和2016两种年份可能出现在检索记录中，通常按2016卷期引用。

访问限制：本次PDF入口在未认证网络检索中返回摘要页，尚未取得该期刊论文全文，也未找到可确认的免费期刊全文。OpenAlex在本次查询中标记closed，且未列出机构库全文；这不是“任何地方都没有免费版本”的证明。可通过学校图书馆Wiley访问权限下载。

已确认摘要介绍的是在低维步态参数空间进行形状估计和全身顺应控制。未取得正文前，不将其具体分支机制断言为2018/2021式的曲线段拼接或两段独立rolling。

免费补充来源：作者2014年博士论文 **Control and Design of Snake Robots**。

- [CMU官方页面](https://publications.ri.cmu.edu/control-and-design-of-snake-robots)。
- [CMU官方博士论文PDF](https://publications.ri.cmu.edu/storage/publications/pub_files/2014/6/d_rollinson_robotics_2014.pdf)。已确认返回PDF，约21.6 MB；网络阅读工具因文件过大未解析。本次未精读博士论文。
- 该文件是博士论文，不能替代或冒充2016年JFR期刊全文。

## 5. 对当前研究的含义

2018年的拼接方法、2021年的管道形状与运动协调、2016年的步态参数顺应控制应分别核对。已有“两段螺旋+连接曲线+rolling”文献，因此这个结构本身不能直接宣称为新贡献。将其与2023顺应性结合后是否有贡献，需要明确新增反馈规则、适用范围和对照结果。

本次没有下载新PDF到仓库，没有更改实机实验原文，也没有证明任何候选方法能通过当前管外非连续急弯。
