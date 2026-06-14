# 3D 可视化模块 (renderer.py)

> 代码位置：`src/utils/renderer.py`

---

## 一、模块概述

`renderer.py` 提供 UAV-FAS 通信系统的实时 3D 可视化功能，基于 matplotlib 的 `mplot3d` 工具包实现。在仿真过程中实时渲染无人机（搭载12端口FAS流体天线）、RIS（64单元有源反射面，位于(20,-20,12.5)）、用户、窃听者的位置以及信道链路。

---

## 二、类结构

### 2.1 Arrow3D

3D 箭头类，继承自 `matplotlib.patches.FancyArrowPatch`，用于在 3D 坐标系中绘制带箭头的信道链路。

```python
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs)
    def draw(self, renderer)
```

### 2.2 Render

主渲染类，管理整个系统的 3D 可视化。

```python
class Render(object):
    def __init__(self, system, canv_x=(-25, 25), canv_y=(0, 50), canv_z=(0, 60))
```

**参数**：
- `system`: MiniSystem 环境对象
- `canv_x`, `canv_y`, `canv_z`: 画布坐标范围

---

## 三、核心方法

### 3.1 render_pause()

暂停渲染，显示当前系统状态的 3D 视图。用于交互式调试。

```python
def render_pause(self):
    plt.ion()
    ax = self.plot_config()
    self.plot_entities(ax)
    self.plot_channels(ax)
    self.plot_text(ax)
    plt.show(self.fig)
    plt.cla()
    self.pause = False
    plt.ioff()
```

### 3.2 render(interval)

实时渲染，显示当前系统状态后暂停指定时间。

```python
def render(self, interval):
    plt.ion()
    ax = self.plot_config()
    self.plot_entities(ax)
    self.plot_channels(ax)
    self.plot_text(ax)
    plt.pause(interval)
    plt.cla()
    plt.ioff()
```

### 3.3 plot_config()

配置 3D 坐标轴：设置标签、范围、视角。

```python
def plot_config(self):
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-25, 25)
    ax.set_ylim3d(0, 50)
    ax.set_zlim3d(0, 60)
    ax.view_init(90, 0)  # 俯视图
    self.fig.canvas.mpl_connect('key_press_event', self.plot_click)
    return ax
```

**交互功能**：按任意键切换暂停/继续（`self.pause ^= True`）。

### 3.4 plot_entities(ax)

绘制所有实体的位置和属性信息：

| 实体 | 颜色 | 显示信息 |
|------|------|---------|
| UAV | 红色 (r) | 坐标 + "UAV" 标签 |
| RIS | 绿色 (g) | 坐标 + "RIS" 标签 |
| 用户 | 蓝色 (b) | 坐标 + 噪声功率 + 容量 + 安全容量 |
| 窃听者 | 黄色 (y) | 坐标 + 窃听容量 |

### 3.5 plot_channels(ax)

绘制所有信道链路（3D 箭头 + 标注）：

| 信道 | 颜色 | 说明 |
|------|------|------|
| h_R_k (RIS→用户) | 蓝色 (b) | RIS 到用户的反射信道 |
| h_R_p (RIS→窃听者) | 黄色 (y) | RIS 到窃听者的反射信道 |
| h_U_k (UAV→用户) | 蓝色 (b) | UAV 到用户的直射信道 |
| h_U_p (UAV→窃听者) | 黄色 (y) | UAV 到窃听者的直射信道 |
| H_UR (UAV→RIS) | 红色 (r) | UAV 到 RIS 的信道 |

### 3.6 plot_one_channel(ax, channel, color)

绘制单条信道链路，标注信道参数：

```
信道名称
n=路径损耗指数     sigma=阴影衰落
PL=路径损耗(归一化)
PL(dB)=路径损耗(分贝)
```

### 3.7 plot_text(ax)

在画布右上角显示系统状态：

```
pause = True/False
t_index = 当前时隙索引
```

---

## 四、使用方式

### 4.1 在环境中自动调用

环境（`MiniSystem`）在 `__init__` 中自动创建 `Render` 对象：

```python
# src/envs/uav_comm_env.py
self.render_obj = Render(self)
```

每次 `step()` 时 `t_index` 自动递增：

```python
def step(self, ...):
    self.render_obj.t_index += 1
    ...
```

### 4.2 手动渲染

```python
from src.utils.renderer import Render
from src.envs import MiniSystem

system = MiniSystem(...)
render = Render(system)

# 渲染一帧
render.render(interval=0.5)

# 暂停渲染（交互模式）
render.render_pause()
```

### 4.3 在仿真脚本中使用

```python
# scripts/run_simulation.py
system = MiniSystem(...)
render = Render(system)

for step in range(100):
    system.step(...)
    render.render(interval=0.1)
```

---

## 五、依赖

| 包 | 用途 |
|----|------|
| `matplotlib` | 绑定和绑图 |
| `mpl_toolkits.mplot3d` | 3D 坐标轴 |
| `matplotlib.patches.FancyArrowPatch` | 3D 箭头 |
| `mpl_toolkits.mplot3d.proj3d` | 3D 投影计算 |

---

## 六、注意事项

1. **交互模式**：`plt.ion()` / `plt.ioff()` 控制交互模式，确保在非 GUI 环境（如服务器）中不会阻塞。

2. **暂停功能**：按任意键切换暂停状态（`self.pause ^= True`），用于交互式调试。

3. **视角设置**：默认 `view_init(90, 0)` 为俯视图，可通过修改 `plot_config()` 中的参数调整。

4. **坐标范围**：默认画布范围 `x:[-25,25], y:[0,50], z:[0,60]`，可通过构造函数参数调整。

5. **性能**：每帧渲染会清除并重绘所有元素（`plt.cla()`），适合实时可视化但不适合高频渲染。
