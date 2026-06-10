"""生成学术论文级HTML训练报告 - IEEE期刊风格"""
import re, json, scipy.io as sio, numpy as np

# ========== 数据准备 ==========
log_path = r'C:\Users\红\AppData\Local\Temp\claude\c--Users---Desktop-0606-------------------------\a4b5dd69-4484-41dd-8084-0389d1ef1403\tasks\bzqzj2zbe.output'
episodes, scores = [], []
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        # 新格式: ep:    0 | reward:  408.444 | avg10:  408.444 | cap: 0.3514 | avg_cap10: 0.3514
        m = re.search(r'ep:\s+(\d+)\s+\|\s+reward:\s+([\d\.\-]+)', line)
        if m:
            episodes.append(int(m.group(1)))
            scores.append(float(m.group(2)))
            continue
        # 旧格式兼容
        m = re.search(r'ep_num:\s+(\d+)\s+ep_score:\s+([\d\.\-]+)', line)
        if m:
            episodes.append(int(m.group(1)))
            scores.append(float(m.group(2)))

def moving_avg(data, w=20):
    return [float(np.mean(data[max(0,i-w+1):i+1])) for i in range(len(data))]
ma_scores = moving_avg(scores)

mat_dir = 'data/storage/uav_bs_fas/scratch/td3_see'
selected_eps = [0, 200, 500, 700, 999]
episode_data = {}
for ep in selected_eps:
    try:
        d = sio.loadmat(f'{mat_dir}/simulation_result_ep_{ep}.mat')
        r = d[f'result_{ep}']
        data = {}
        for name in r.dtype.names:
            v = r[name][0, 0]
            data[name] = v.tolist() if hasattr(v, 'tolist') else v
        episode_data[ep] = data
    except Exception as e:
        print(f'Error loading ep {ep}: {e}')

# 实体位置 (从init_location.xlsx)
entities = {
    'uav_start': [0, 25],
    'user0': [4, 47],
    'user1': [25, 25],
    'attacker0': [47, -4],
    'ris': [0, 50],
}

final_100_avg = float(np.mean(scores[-100:]))
best_idx = int(np.argmax(scores))
best_ep = episodes[best_idx]
best_score = float(max(scores))
init_avg = float(np.mean(scores[:10]))
speedup = final_100_avg / init_avg

# JSON数据
j_episodes = json.dumps(episodes)
j_scores = json.dumps([round(s, 4) for s in scores])
j_ma = json.dumps([round(s, 4) for s in ma_scores])

traj_data = {}
cap_data = {}
sec_data = {}
ris_data = {}
for ep in selected_eps:
    if ep in episode_data:
        uav = np.array(episode_data[ep]['UAV_state'])
        traj_data[ep] = {'x': uav[:,0].tolist(), 'y': uav[:,1].tolist()}
        uc = np.array(episode_data[ep]['user_capacity'])
        ac = np.array(episode_data[ep]['attaker_capacity']).squeeze()
        # ac shape: (100, 2) -> [窃听User0容量, 窃听User1容量]
        cap_data[ep] = {
            'u0': uc[:,0].tolist(), 'u1': uc[:,1].tolist(),
            'a0': ac[:,0].tolist(), 'a1': ac[:,1].tolist()
        }
for ep in [200, 700, 999]:
    if ep in episode_data:
        sc = np.array(episode_data[ep]['secure_capacity'])
        sec_data[ep] = {'u0': sc[:,0].tolist(), 'u1': sc[:,1].tolist()}
for ep in selected_eps:
    if ep in episode_data:
        data = {}
        if 'RIS_signal_phase' in episode_data[ep]:
            data['ris_signal'] = np.array(episode_data[ep]['RIS_signal_phase']).flatten().tolist()
        if 'RIS_jam_phase' in episode_data[ep]:
            data['ris_jam'] = np.array(episode_data[ep]['RIS_jam_phase']).flatten().tolist()
        if 'FAS_active_port' in episode_data[ep]:
            data['fas_port'] = np.array(episode_data[ep]['FAS_active_port']).flatten().tolist()
        if data:
            ris_data[ep] = data

# 摘要表
table_rows = []
for key, label, col in [
    ('secure_capacity', '平均安全容量 (用户0)', 0),
    ('user_capacity', '平均用户容量 (用户0)', 0),
]:
    vals = []
    for ep in [0, 200, 500, 999]:
        if ep in episode_data:
            arr = np.array(episode_data[ep][key])
            v = arr[:, col].mean() if col is not None else arr.mean()
            vals.append(f'{v:.4f}')
        else:
            vals.append('-')
    table_rows.append((label, vals))

# ========== 生成HTML ==========
html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>FAS-UAV保密通信TD3训练报告</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
@page {size: A4; margin: 20mm;}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Noto Sans SC','Times New Roman',serif;background:#fff;color:#222;line-height:1.6}

/* 论文标题区 */
.title-block{text-align:center;padding:40px 60px 30px;border-bottom:2px solid #000}
.title-block h1{font-size:22px;font-weight:700;margin-bottom:8px;letter-spacing:1px}
.title-block .subtitle{font-size:13px;color:#555}
.title-block .meta{font-size:11px;color:#888;margin-top:6px}

/* 统计摘要 */
.summary{display:flex;justify-content:center;gap:32px;padding:24px 40px;border-bottom:1px solid #ddd;flex-wrap:wrap}
.summary .item{text-align:center;min-width:120px}
.summary .val{font-size:24px;font-weight:700;color:#000}
.summary .lbl{font-size:11px;color:#666;margin-top:2px}

/* 图表区 */
.figure{padding:20px 40px;page-break-inside:avoid}
.figure .fig-label{font-size:12px;font-weight:700;margin-bottom:6px}
.figure .fig-caption{font-size:11px;color:#555;margin-top:6px;text-align:center}
.chart{background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:8px;margin:8px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}

/* 表格 */
.tbl-section{padding:20px 40px}
.tbl-section table{width:100%;border-collapse:collapse;font-size:12px}
.tbl-section th,.tbl-section td{padding:8px 12px;border:1px solid #ccc;text-align:center}
.tbl-section th{background:#f5f5f5;font-weight:700}
.tbl-section caption{font-size:12px;font-weight:700;margin-bottom:8px;text-align:left}

/* 页脚 */
.footer{text-align:center;padding:20px;font-size:10px;color:#999;border-top:1px solid #ddd;margin-top:20px}
</style>
</head>
<body>

<div class="title-block">
  <h1>流体天线辅助无人机安全通信 — TD3强化学习训练报告</h1>
  <div class="subtitle">FAS-Assisted UAV Secure Communication System &mdash; Twin Delayed DDPG Training Report</div>
  <div class="meta">奖励函数: 改进v2 (安全速率+窃听者压制+RIS比例干扰) &nbsp;|&nbsp; 训练轮次: 1000 &nbsp;|&nbsp; 每轮步数: 100</div>
</div>

<div class="summary">
  <div class="item"><div class="val">__TOTAL_EP__</div><div class="lbl">训练轮次</div></div>
  <div class="item"><div class="val">__FINAL_AVG__</div><div class="lbl">最终100轮均分</div></div>
  <div class="item"><div class="val">__BEST_SCORE__</div><div class="lbl">最高分 (Ep.__BEST_EP__)</div></div>
  <div class="item"><div class="val">__INIT_AVG__</div><div class="lbl">初始均分(前10轮)</div></div>
  <div class="item"><div class="val">__SPEEDUP__x</div><div class="lbl">性能提升</div></div>
</div>

<!-- 图1: 训练曲线 -->
<div class="figure">
  <div class="fig-label">图1 &nbsp; 训练奖励收敛曲线</div>
  <div class="chart" id="c_reward"></div>
  <div class="fig-caption">Fig.1 &nbsp; Training reward convergence curve. Blue line: moving average (window=20); gray line: per-episode reward.</div>
</div>

<!-- 图2: 无人机轨迹 -->
<div class="figure">
  <div class="fig-label">图2 &nbsp; 不同训练阶段无人机飞行轨迹</div>
  <div class="grid2" id="traj_container"></div>
  <div class="fig-caption">Fig.2 &nbsp; UAV trajectories at different training stages. Green triangle: start; Red cross: end; Blue line: trajectory; Markers: entities.</div>
</div>

<!-- 图3: 信道容量 -->
<div class="figure">
  <div class="fig-label">图3 &nbsp; 用户与窃听者信道容量对比</div>
  <div class="grid2" id="cap_container"></div>
  <div class="fig-caption">Fig.3 &nbsp; 信道容量对比。窃听者对两个用户的窃听容量(虚线) vs 用户自身容量(实线)。</div>
</div>

<!-- 图4: 安全容量 -->
<div class="figure">
  <div class="fig-label">图4 &nbsp; 安全容量随时间变化</div>
  <div class="grid2" id="sec_container"></div>
  <div class="fig-caption">Fig.4 &nbsp; Secrecy capacity over time slots.</div>
</div>

<!-- 图5: RIS相位 + FAS端口 -->
<div class="figure">
  <div class="fig-label">图5 &nbsp; RIS相位与FAS端口选择</div>
  <div class="grid2" id="ris_container"></div>
  <div class="fig-caption">Fig.5 &nbsp; RIS phase (signal reflection + jamming) and FAS active port index.</div>
</div>

<!-- 表1: 摘要 -->
<div class="tbl-section">
  <table>
    <caption>表1 &nbsp; 训练关键指标对比</caption>
    <tr><th>指标</th><th>Episode 0</th><th>Episode 200</th><th>Episode 500</th><th>Episode 999</th></tr>
    __TABLE_ROWS__
  </table>
</div>

<div class="footer">FAS-UAV Secure Communication &mdash; TD3 Reinforcement Learning Training Report</div>

<script>
// IEEE风格配色
var C = {
  blue: '#000000',
  red: '#C00000',
  green: '#007500',
  cyan: '#0075C0',
  magenta: '#C000C0',
  orange: '#ED7D31',
  gray: '#808080',
  lightgray: '#BFBFBF'
};

var layout_base = {
  font: {family: 'Times New Roman, serif', size: 12, color: '#000'},
  paper_bgcolor: '#fff',
  plot_bgcolor: '#fff',
  margin: {t: 35, b: 45, l: 55, r: 20},
  height: 320,
  legend: {bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#ccc', borderwidth: 1, font: {size: 11}}
};

var episodes = __EPISODES__;
var scores = __SCORES__;
var ma_scores = __MA_SCORES__;
var traj = __TRAJ__;
var cap = __CAP__;
var sec = __SEC__;
var risData = __RIS__;

// ====== 图1: 训练曲线 ======
Plotly.newPlot('c_reward', [
  {x: episodes, y: scores, mode: 'lines', name: '逐轮奖励',
   line: {color: C.lightgray, width: 0.8}},
  {x: episodes, y: ma_scores, mode: 'lines', name: '滑动均值 (w=20)',
   line: {color: C.cyan, width: 2.5}}
], Object.assign({}, layout_base, {
  height: 350,
  xaxis: {title: {text: '训练轮次 (Episode)', font: {size: 13}},
           tickfont: {size: 11}, gridcolor: '#eee', dtick: 100},
  yaxis: {title: {text: '奖励 (Reward / SEE)', font: {size: 13}},
           tickfont: {size: 11}, gridcolor: '#eee'}
}));

// ====== 图2: 无人机轨迹 ======
var trajEps = [0, 50, 200, 500, 999];
var trajContainer = document.getElementById('traj_container');

// 实体标记样式
var entityMarkers = [
  {x: [__USER0_X__], y: [__USER0_Y__], name: '用户0', mode: 'markers',
   marker: {size: 11, color: '#fff', symbol: 'circle', line: {color: C.green, width: 2}}},
  {x: [__USER1_X__], y: [__USER1_Y__], name: '用户1', mode: 'markers',
   marker: {size: 11, color: '#fff', symbol: 'diamond', line: {color: C.green, width: 2}}},
  {x: [__ATT0_X__], y: [__ATT0_Y__], name: '窃听者', mode: 'markers',
   marker: {size: 11, color: C.red, symbol: 'cross', line: {color: C.red, width: 2}}},
  {x: [__RIS_X__], y: [__RIS_Y__], name: 'RIS', mode: 'markers',
   marker: {size: 11, color: '#fff', symbol: 'square', line: {color: C.magenta, width: 2}}}
];

trajEps.forEach(function(ep) {
  if (!traj[ep]) return;
  var div = document.createElement('div');
  div.className = 'chart';
  div.id = 'traj_' + ep;
  trajContainer.appendChild(div);

  var traces = [
    {x: traj[ep].x, y: traj[ep].y, mode: 'lines', name: 'UAV轨迹',
     line: {color: C.cyan, width: 2}},
    {x: [traj[ep].x[0]], y: [traj[ep].y[0]], mode: 'markers', name: '起点',
     marker: {size: 10, color: C.green, symbol: 'triangle-up'}},
    {x: [traj[ep].x[traj[ep].x.length-1]], y: [traj[ep].y[traj[ep].y.length-1]],
     mode: 'markers', name: '终点',
     marker: {size: 10, color: C.red, symbol: 'x', line: {width: 2}}}
  ].concat(entityMarkers);

  Plotly.newPlot(div.id, traces, Object.assign({}, layout_base, {
    height: 340,
    title: {text: '(a) Episode ' + ep, font: {size: 13, color: '#000'}},
    xaxis: {title: {text: 'X (m)', font: {size: 12}}, range: [-80, 80],
             tickfont: {size: 10}, gridcolor: '#eee', dtick: 20},
    yaxis: {title: {text: 'Y (m)', font: {size: 12}}, range: [-60, 60],
             tickfont: {size: 10}, gridcolor: '#eee', dtick: 20},
    shapes: [{type: 'rect', x0: -25, x1: 25, y0: 0, y1: 50,
              line: {color: '#999', width: 1, dash: 'dot'}, fillcolor: 'rgba(0,0,0,0)'}],
    legend: {x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', font: {size: 9}}
  }));
});

// ====== 图3: 信道容量 ======
var capEps = [0, 200, 500, 999];
var capContainer = document.getElementById('cap_container');
capEps.forEach(function(ep) {
  if (!cap[ep]) return;
  var div = document.createElement('div');
  div.className = 'chart';
  div.id = 'cap_' + ep;
  capContainer.appendChild(div);
  var x = [];
  for (var i = 0; i < 100; i++) x.push(i);
  Plotly.newPlot(div.id, [
    {x: x, y: cap[ep].u0, name: '用户0容量', line: {color: C.cyan, width: 1.5}},
    {x: x, y: cap[ep].u1, name: '用户1容量', line: {color: C.green, width: 1.5}},
    {x: x, y: cap[ep].a0, name: '窃听User0', line: {color: C.red, width: 1.5, dash: 'dash'}},
    {x: x, y: cap[ep].a1, name: '窃听User1', line: {color: C.orange, width: 1.5, dash: 'dash'}}
  ], Object.assign({}, layout_base, {
    title: {text: '(b) Episode ' + ep, font: {size: 13, color: '#000'}},
    xaxis: {title: {text: '时隙 (Time Slot)', font: {size: 12}},
             tickfont: {size: 10}, gridcolor: '#eee'},
    yaxis: {title: {text: '容量 (bits/s/Hz)', font: {size: 12}},
             tickfont: {size: 10}, gridcolor: '#eee'},
    legend: {x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', font: {size: 10}}
  }));
});

// ====== 图4: 安全容量 ======
var secEps = [200, 500, 999];
var secContainer = document.getElementById('sec_container');
secEps.forEach(function(ep) {
  if (!sec[ep]) return;
  var div = document.createElement('div');
  div.className = 'chart';
  div.id = 'sec_' + ep;
  secContainer.appendChild(div);
  var x = [];
  for (var i = 0; i < 100; i++) x.push(i);
  Plotly.newPlot(div.id, [
    {x: x, y: sec[ep].u0, name: '用户0', fill: 'tozeroy',
     line: {color: C.cyan, width: 1.5}, fillcolor: 'rgba(0,117,192,0.15)'},
    {x: x, y: sec[ep].u1, name: '用户1', fill: 'tozeroy',
     line: {color: C.green, width: 1.5}, fillcolor: 'rgba(0,117,0,0.1)'}
  ], Object.assign({}, layout_base, {
    title: {text: '(c) Episode ' + ep, font: {size: 13, color: '#000'}},
    xaxis: {title: {text: '时隙 (Time Slot)', font: {size: 12}},
             tickfont: {size: 10}, gridcolor: '#eee'},
    yaxis: {title: {text: '安全容量 (bits/s/Hz)', font: {size: 12}},
             tickfont: {size: 10}, gridcolor: '#eee'},
    legend: {x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', font: {size: 10}}
  }));
});

// ====== 图5: RIS相位 + FAS端口 ======
var risEps = Object.keys(risData).map(Number);
var risContainer = document.getElementById('ris_container');
risEps.forEach(function(ep) {
  if (!risData[ep]) return;
  var div = document.createElement('div');
  div.className = 'chart';
  div.id = 'ris_' + ep;
  risContainer.appendChild(div);
  var traces = [];
  var x = [];
  for (var i = 0; i < (risData[ep].ris_signal || []).length; i++) x.push(i);
  if (risData[ep].ris_signal) {
    traces.push({x: x, y: risData[ep].ris_signal, name: 'RIS信号相位', line: {color: C.cyan, width: 1.5}});
  }
  if (risData[ep].ris_jam) {
    traces.push({x: x, y: risData[ep].ris_jam, name: 'RIS干扰相位', line: {color: C.red, width: 1.5, dash: 'dash'}});
  }
  if (risData[ep].fas_port) {
    traces.push({x: x, y: risData[ep].fas_port, name: 'FAS端口号', yaxis: 'y2', line: {color: C.orange, width: 1.5}});
  }
  Plotly.newPlot(div.id, traces, Object.assign({}, layout_base, {
    title: {text: '(e) Episode ' + ep, font: {size: 13, color: '#000'}},
    xaxis: {title: {text: '时隙 (Time Slot)', font: {size: 12}}, tickfont: {size: 10}, gridcolor: '#eee'},
    yaxis: {title: {text: 'RIS相位 (归一化)', font: {size: 12}}, tickfont: {size: 10}, gridcolor: '#eee', range: [-1.2, 1.2]},
    yaxis2: {title: {text: 'FAS端口号', font: {size: 12}, font: {color: C.orange}}, tickfont: {size: 10, color: C.orange}, overlaying: 'y', side: 'right'},
    legend: {x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', font: {size: 10}},
    height: 380
  }));
});
</script>
</body>
</html>"""

# 填充占位符
html = html.replace('__TOTAL_EP__', str(len(episodes)))
html = html.replace('__FINAL_AVG__', f'{final_100_avg:.2f}')
html = html.replace('__BEST_SCORE__', f'{best_score:.2f}')
html = html.replace('__BEST_EP__', str(best_ep))
html = html.replace('__INIT_AVG__', f'{init_avg:.2f}')
html = html.replace('__SPEEDUP__', f'{speedup:.1f}')

html = html.replace('__EPISODES__', j_episodes)
html = html.replace('__SCORES__', j_scores)
html = html.replace('__MA_SCORES__', j_ma)
html = html.replace('__TRAJ__', json.dumps(traj_data))
html = html.replace('__CAP__', json.dumps(cap_data))
html = html.replace('__SEC__', json.dumps(sec_data))
html = html.replace('__RIS__', json.dumps(ris_data))

# 实体坐标
html = html.replace('__USER0_X__', str(entities['user0'][0]))
html = html.replace('__USER0_Y__', str(entities['user0'][1]))
html = html.replace('__USER1_X__', str(entities['user1'][0]))
html = html.replace('__USER1_Y__', str(entities['user1'][1]))
html = html.replace('__ATT0_X__', str(entities['attacker0'][0]))
html = html.replace('__ATT0_Y__', str(entities['attacker0'][1]))
html = html.replace('__RIS_X__', str(entities['ris'][0]))
html = html.replace('__RIS_Y__', str(entities['ris'][1]))

rows_html = ''
for label, vals in table_rows:
    cells = ''.join(f'<td>{v}</td>' for v in vals)
    rows_html += f'<tr><td style="text-align:left;font-weight:700">{label}</td>{cells}</tr>\n'
html = html.replace('__TABLE_ROWS__', rows_html)

out_path = 'data/storage/uav_bs_fas/scratch/td3_see/training_report.html'
with open(out_path, 'w', encoding='utf-8-sig') as f:
    f.write(html)
print(f'报告已生成: {out_path}')
