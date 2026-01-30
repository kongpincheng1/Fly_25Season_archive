# 说明
这是25赛季仿真控制代码的存档，删去了其他无用的代码。

运行的主程序为`sim`文件夹中的`0809_sim_mono.py`

这个ros2包的名称为control

> *对代码进行修改后，请运行`colcon build`重新构建*
## 在工作空间根目录运行下面的代码即可运行程序

```bash
source install/setup.bash
ros2 run control test
```
在`colcon build`之后，在工作空间根目录运行以上代码即可运行`0809_sim_mono.py`

在`setup.py`中，设置的程序入口名称为`test`,你可以自行更改。

程序运行的是视觉模型文件是`models`下的`best_sim.pt`,这是yolov8适用于仿真环境的模型，后续如果有需要可自行训练其他。<br>

## 通过命令行修改参数
你可以通过命令行修改运行参数。

例如：<br>
```bash
ros2 run control test \
  --record_video true \
  --timer_period 0.04 \
  --kp 2.0 --ki 0.2 --kd 0.1 --kf 0.01 \
  --takeoff-height 2.8 
```

更多可用的参数请在`0809_sim_mono.py`内查看

## 任务流程

`0809_sim_mono.py` 内部维护一个多阶段任务状态机，整体流程如下：

>### 1. START
>- 初始化 ROS2 节点
>- 等待 PX4 进入 **Offboard** 模式

>### 2. GLOBAL_SEARCH
>- 无人机上升至固定高度
>- 执行全局目标搜索

>### 3. TARGETING_CYCLE
>- 初步视觉对准目标
>- 二次下降进行精细对准
>- 判断是否满足投放条件

>### 4. TIMEOUT_DROP
>- 若搜索或对准超时
>- 执行保底投放策略

>### 5. TRANSIT_TO_RECON_OFFBOARD
>- 切换至侦察 Offboard 模式
>- 调整飞行状态

>### 6. RECON_SEARCH
>- 执行后续巡航或侦察任务

>***祝你好运***
