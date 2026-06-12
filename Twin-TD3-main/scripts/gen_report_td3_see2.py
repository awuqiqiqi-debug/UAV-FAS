#!/usr/bin/env python3
"""
Generate interactive HTML training report for TTD3 UAV secure communication.
Reads .mat and .csv data from td3_see_3 and produces a Plotly.js-based report.
"""

import os
import csv
import json
import numpy as np
import scipy.io as sio

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "data", "storage",
    "uav_bs_fas", "scratch", "td3_see_3"
)
DATA_DIR = os.path.abspath(DATA_DIR)
OUTPUT_HTML = os.path.join(DATA_DIR, "training_report.html")

TOTAL_EPISODES = 2000
STEPS_PER_EPISODE = 100
SELECTED_EPS = [0, 500, 1000, 1999]
TRAJ_EPS = [0, 500, 1000, 1999]
MA_WINDOW = 20

# Fixed entity positions
ENTITY_MARKERS = [
    {"x": [4], "y": [47], "name": "用户0"},
    {"x": [25], "y": [25], "name": "用户1"},
    {"x": [47], "y": [-4], "name": "窃听者"},
    {"x": [0], "y": [50], "name": "RIS"},
]


# ============================================================
# Helper: load structured .mat result
# ============================================================
def load_episode_result(mat_path, ep):
    data = sio.loadmat(mat_path)
    key = "result_%d" % ep
    if key not in data:
        raise KeyError("Key '%s' not found in %s" % (key, mat_path))
    return data[key][0, 0]


def safe_array(result, field):
    try:
        return np.asarray(result[field])
    except Exception:
        return None


# ============================================================
# Load data
# ============================================================
print("Loading training rewards ...")
rewards = []
with open(os.path.join(DATA_DIR, "training_rewards.csv")) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        rewards.append(float(row[0]))
rewards = np.array(rewards)

print("Loading training capacities ...")
capacities = []
with open(os.path.join(DATA_DIR, "training_capacities.csv")) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        capacities.append(float(row[0]))
capacities = np.array(capacities)


def moving_average(arr, w):
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - w + 1)
        out[i] = np.mean(arr[start : i + 1])
    return out


ma_rewards = moving_average(rewards, MA_WINDOW)

n_total = len(rewards)
final_100_mean = float(np.mean(rewards[-100:]))
max_reward = float(np.max(rewards))
max_ep = int(np.argmax(rewards))
initial_10_mean = float(np.mean(rewards[:10]))
perf_improvement = (
    final_100_mean / abs(initial_10_mean)
    if initial_10_mean < 0
    else final_100_mean / initial_10_mean
)

# Load per-episode data
print("Loading per-episode data ...")
episode_data = {}
for ep in SELECTED_EPS:
    mat_path = os.path.join(DATA_DIR, "simulation_result_ep_%d.mat" % ep)
    if os.path.exists(mat_path):
        try:
            result = load_episode_result(mat_path, ep)
            uav_state = safe_array(result, "UAV_state")
            user_cap = safe_array(result, "user_capacity")
            sec_cap = safe_array(result, "secure_capacity")
            attacker_cap = safe_array(result, "attaker_capacity")
            ris_signal = safe_array(result, "RIS_signal_phase")
            ris_jam = safe_array(result, "RIS_jam_phase")
            fas_port = safe_array(result, "FAS_active_port")

            # Flatten attacker_cap from (100,1,2) to list of [a0, a1]
            ac_list = []
            if attacker_cap is not None:
                ac = attacker_cap.reshape(attacker_cap.shape[0], -1)
                ac_list = ac.tolist()

            episode_data[ep] = {
                "uav_x": uav_state[:, 0].tolist() if uav_state is not None else [],
                "uav_y": uav_state[:, 1].tolist() if uav_state is not None else [],
                "user_cap": user_cap.tolist() if user_cap is not None else [],
                "sec_cap": sec_cap.tolist() if sec_cap is not None else [],
                "attacker_cap": ac_list,
                "ris_signal": ris_signal.flatten().tolist() if ris_signal is not None else [],
                "ris_jam": ris_jam.flatten().tolist() if ris_jam is not None else [],
                "fas_port": fas_port.flatten().tolist() if fas_port is not None else [],
            }
            print("  Loaded ep %d" % ep)
        except Exception as e:
            print("  WARNING: Failed to load ep %d: %s" % (ep, e))

# Table data
table_rows = []
for ep in SELECTED_EPS:
    if ep in episode_data:
        ed = episode_data[ep]
        sc = np.array(ed["sec_cap"])
        uc = np.array(ed["user_cap"])
        avg_sec_u0 = float(sc[:, 0].mean()) if sc.size > 0 else 0.0
        avg_uc_u0 = float(uc[:, 0].mean()) if uc.size > 0 else 0.0
        table_rows.append((ep, avg_sec_u0, avg_uc_u0))
    else:
        table_rows.append((ep, 0.0, 0.0))

# ============================================================
# Build JSON data for embedding
# ============================================================
print("Generating HTML report ...")

data_dict = {
    "episodes": list(range(TOTAL_EPISODES)),
    "scores": [None if np.isnan(v) else round(v, 3) for v in rewards],
    "ma_scores": [None if np.isnan(v) else round(v, 3) for v in ma_rewards],
    "traj": {str(ep): episode_data[ep] for ep in TRAJ_EPS if ep in episode_data},
    "cap": {str(ep): episode_data[ep] for ep in SELECTED_EPS if ep in episode_data},
    "sec": {str(ep): episode_data[ep] for ep in SELECTED_EPS if ep in episode_data},
    "ris": {str(ep): episode_data[ep] for ep in SELECTED_EPS if ep in episode_data},
}
data_json = json.dumps(data_dict, ensure_ascii=False)

# Table rows HTML
table_html_rows = ""
for ep, avg_sec, avg_uc in table_rows:
    table_html_rows += '    <tr><td style="text-align:left;font-weight:700">平均安全容量 (用户0)</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%.4f</td></tr>\n' % (
        table_rows[0][1], table_rows[1][1], table_rows[2][1], table_rows[3][1]
    )
    break  # Only need to build this once

table_html_rows = ""
cells_sec = "".join("<td>%.4f</td>" % r[1] for r in table_rows)
table_html_rows += '    <tr><td style="text-align:left;font-weight:700">平均安全容量 (用户0)</td>%s</tr>\n' % cells_sec
cells_uc = "".join("<td>%.4f</td>" % r[2] for r in table_rows)
table_html_rows += '    <tr><td style="text-align:left;font-weight:700">平均用户容量 (用户0)</td>%s</tr>\n' % cells_uc

ep_headers = "".join("<th>Episode %d</th>" % r[0] for r in table_rows)

# ============================================================
# Build HTML using string replacement (avoid f-string brace issues)
# ============================================================
html_template = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>FAS-UAV保密通信TD3训练报告 (td3_see_3)</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
@page {size: A4; margin: 20mm;}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Noto Sans SC','Times New Roman',serif;background:#fff;color:#222;line-height:1.6}

.title-block{text-align:center;padding:40px 60px 30px;border-bottom:2px solid #000}
.title-block h1{font-size:22px;font-weight:700;margin-bottom:8px;letter-spacing:1px}
.title-block .subtitle{font-size:13px;color:#555}
.title-block .meta{font-size:11px;color:#888;margin-top:6px}

.summary{display:flex;justify-content:center;gap:32px;padding:24px 40px;border-bottom:1px solid #ddd;flex-wrap:wrap}
.summary .item{text-align:center;min-width:120px}
.summary .val{font-size:24px;font-weight:700;color:#000}
.summary .lbl{font-size:11px;color:#666;margin-top:2px}

.figure{padding:20px 40px;page-break-inside:avoid}
.figure .fig-label{font-size:12px;font-weight:700;margin-bottom:6px}
.figure .fig-caption{font-size:11px;color:#555;margin-top:6px;text-align:center}
.chart{background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:8px;margin:8px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}

.tbl-section{padding:20px 40px}
.tbl-section table{width:100%;border-collapse:collapse;font-size:12px}
.tbl-section th,.tbl-section td{padding:8px 12px;border:1px solid #ccc;text-align:center}
.tbl-section th{background:#f5f5f5;font-weight:700}
.tbl-section caption{font-size:12px;font-weight:700;margin-bottom:8px;text-align:left}

.footer{text-align:center;padding:20px;font-size:10px;color:#999;border-top:1px solid #ddd;margin-top:20px}
</style>
</head>
<body>

<div class="title-block">
  <h1>流体天线辅助无人机安全通信 — TD3强化学习训练报告</h1>
  <div class="subtitle">FAS-Assisted UAV Secure Communication System &mdash; Twin Delayed TD3 Training Report (td3_see_3)</div>
  <div class="meta">奖励函数: SEE (安全能效) &nbsp;|&nbsp; 训练轮次: __TOTAL_EPISODES__ &nbsp;|&nbsp; 每轮步数: __STEPS_PER_EPISODE__</div>
</div>

<div class="summary">
  <div class="item"><div class="val">__TOTAL_EPISODES__</div><div class="lbl">训练轮次</div></div>
  <div class="item"><div class="val">__FINAL100_MEAN__</div><div class="lbl">最终100轮均分</div></div>
  <div class="item"><div class="val">__MAX_REWARD__</div><div class="lbl">最高分 (Ep.__MAX_EP__)</div></div>
  <div class="item"><div class="val">__INIT10_MEAN__</div><div class="lbl">初始均分(前10轮)</div></div>
  <div class="item"><div class="val">__PERF_IMP__x</div><div class="lbl">性能提升</div></div>
</div>

<div class="figure">
  <div class="fig-label">图1 &nbsp; 训练奖励收敛曲线</div>
  <div class="chart" id="c_reward"></div>
  <div class="fig-caption">Fig.1 &nbsp; Training reward convergence curve. Blue line: moving average (window=__MA_WINDOW__); gray line: per-episode reward.</div>
</div>

<div class="figure">
  <div class="fig-label">图2 &nbsp; 不同训练阶段无人机飞行轨迹</div>
  <div class="grid2" id="traj_container"></div>
  <div class="fig-caption">Fig.2 &nbsp; UAV trajectories at different training stages. Green triangle: start; Red cross: end; Blue line: trajectory; Markers: entities.</div>
</div>

<div class="figure">
  <div class="fig-label">图3 &nbsp; 用户与窃听者信道容量对比</div>
  <div class="grid2" id="cap_container"></div>
  <div class="fig-caption">Fig.3 &nbsp; Channel capacity comparison. Eavesdropper capacity (dashed) vs. user capacity (solid).</div>
</div>

<div class="figure">
  <div class="fig-label">图4 &nbsp; 安全容量随时间变化</div>
  <div class="grid2" id="sec_container"></div>
  <div class="fig-caption">Fig.4 &nbsp; Secrecy capacity over time slots.</div>
</div>

<div class="figure">
  <div class="fig-label">图5 &nbsp; RIS相位与FAS端口选择</div>
  <div class="grid2" id="ris_container"></div>
  <div class="fig-caption">Fig.5 &nbsp; RIS phase (signal reflection + jamming) and FAS active port index.</div>
</div>

<div class="tbl-section">
  <table>
    <caption>表1 &nbsp; 训练关键指标对比</caption>
    <tr><th>指标</th>__EP_HEADERS__</tr>
__TABLE_ROWS__
  </table>
</div>

<div class="footer">FAS-UAV Secure Communication &mdash; TD3 Reinforcement Learning Training Report (td3_see_3)</div>

<script>
var C = {
  blue: '#000000', red: '#C00000', green: '#007500', cyan: '#0075C0',
  magenta: '#C000C0', orange: '#ED7D31', gray: '#808080', lightgray: '#BFBFBF'
};
var layout_base = {
  font: {family: 'Times New Roman, serif', size: 12, color: '#000'},
  paper_bgcolor: '#fff', plot_bgcolor: '#fff',
  margin: {t: 35, b: 45, l: 55, r: 20}, height: 320,
  legend: {bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#ccc', borderwidth: 1, font: {size: 11}}
};

var D = __DATA_JSON__;

// ====== Fig.1: Training reward convergence ======
Plotly.newPlot('c_reward', [
  {x: D.episodes, y: D.scores, mode: 'lines', name: '逐轮奖励',
   line: {color: C.lightgray, width: 0.8}},
  {x: D.episodes, y: D.ma_scores, mode: 'lines', name: '滑动均值 (w=__MA_WINDOW__)',
   line: {color: C.cyan, width: 2.5}}
], Object.assign({}, layout_base, {
  height: 350,
  xaxis: {title: {text: '训练轮次 (Episode)', font: {size: 13}},
           tickfont: {size: 11}, gridcolor: '#eee', dtick: 100},
  yaxis: {title: {text: '奖励 (Reward / SEE)', font: {size: 13}},
           tickfont: {size: 11}, gridcolor: '#eee'}
}));

// ====== Fig.2: UAV trajectories ======
var trajEps = __TRAJ_EPS_JSON__;
var trajContainer = document.getElementById('traj_container');
var entityMarkers = [
  {x: [4], y: [47], name: '用户0', mode: 'markers',
   marker: {size: 11, color: '#fff', symbol: 'circle', line: {color: C.green, width: 2}}},
  {x: [25], y: [25], name: '用户1', mode: 'markers',
   marker: {size: 11, color: '#fff', symbol: 'diamond', line: {color: C.green, width: 2}}},
  {x: [47], y: [-4], name: '窃听者', mode: 'markers',
   marker: {size: 11, color: C.red, symbol: 'cross', line: {color: C.red, width: 2}}},
  {x: [0], y: [50], name: 'RIS', mode: 'markers',
   marker: {size: 11, color: '#fff', symbol: 'square', line: {color: C.magenta, width: 2}}}
];

trajEps.forEach(function(ep) {
  if (!D.traj[ep]) return;
  var div = document.createElement('div');
  div.className = 'chart'; div.id = 'traj_' + ep;
  trajContainer.appendChild(div);
  var traces = [
    {x: D.traj[ep].uav_x, y: D.traj[ep].uav_y, mode: 'lines', name: 'UAV轨迹',
     line: {color: C.cyan, width: 2}},
    {x: [D.traj[ep].uav_x[0]], y: [D.traj[ep].uav_y[0]], mode: 'markers', name: '起点',
     marker: {size: 10, color: C.green, symbol: 'triangle-up'}},
    {x: [D.traj[ep].uav_x[D.traj[ep].uav_x.length-1]], y: [D.traj[ep].uav_y[D.traj[ep].uav_y.length-1]],
     mode: 'markers', name: '终点',
     marker: {size: 10, color: C.red, symbol: 'x', line: {width: 2}}}
  ].concat(entityMarkers);
  Plotly.newPlot(div.id, traces, Object.assign({}, layout_base, {
    height: 340,
    title: {text: '(a) Episode ' + ep, font: {size: 13, color: '#000'}},
    xaxis: {title: {text: 'X (m)', font: {size: 12}}, range: [-80, 80],
             tickfont: {size: 10}, gridcolor: '#eee', dtick: 20},
    yaxis: {title: {text: 'Y (m)', font: {size: 12}}, range: [-10, 60],
             tickfont: {size: 10}, gridcolor: '#eee', dtick: 20},
    shapes: [{type: 'rect', x0: -50, x1: 50, y0: -50, y1: 50,
              line: {color: '#999', width: 1, dash: 'dot'}, fillcolor: 'rgba(0,0,0,0)'}],
    legend: {x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', font: {size: 9}}
  }));
});

// ====== Fig.3: Channel capacity ======
var capEps = __SELECTED_EPS_JSON__;
var capContainer = document.getElementById('cap_container');
capEps.forEach(function(ep) {
  if (!D.cap[ep]) return;
  var div = document.createElement('div');
  div.className = 'chart'; div.id = 'cap_' + ep;
  capContainer.appendChild(div);
  var x = []; for (var i = 0; i < __STEPS__; i++) x.push(i);
  var uc = D.cap[ep].user_cap;
  var ac = D.cap[ep].attacker_cap;
  var a0 = [], a1 = [];
  if (ac && ac.length > 0) {
    for (var i = 0; i < ac.length; i++) {
      if (Array.isArray(ac[i]) && ac[i].length >= 2) {
        a0.push(ac[i][0]); a1.push(ac[i][1]);
      } else {
        a0.push(Array.isArray(ac[i]) ? ac[i][0] : ac[i]); a1.push(0);
      }
    }
  }
  Plotly.newPlot(div.id, [
    {x: x, y: uc.map(function(r){return r[0]}), name: '用户0容量', line: {color: C.cyan, width: 1.5}},
    {x: x, y: uc.map(function(r){return r[1]}), name: '用户1容量', line: {color: C.green, width: 1.5}},
    {x: x, y: a0, name: '窃听User0', line: {color: C.red, width: 1.5, dash: 'dash'}},
    {x: x, y: a1, name: '窃听User1', line: {color: C.orange, width: 1.5, dash: 'dash'}}
  ], Object.assign({}, layout_base, {
    title: {text: '(b) Episode ' + ep, font: {size: 13, color: '#000'}},
    xaxis: {title: {text: '时隙 (Time Slot)', font: {size: 12}}, tickfont: {size: 10}, gridcolor: '#eee'},
    yaxis: {title: {text: '容量 (bits/s/Hz)', font: {size: 12}}, tickfont: {size: 10}, gridcolor: '#eee'},
    legend: {x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', font: {size: 10}}
  }));
});

// ====== Fig.4: Secrecy capacity ======
var secEps = __SELECTED_EPS_JSON__;
var secContainer = document.getElementById('sec_container');
secEps.forEach(function(ep) {
  if (!D.sec[ep]) return;
  var div = document.createElement('div');
  div.className = 'chart'; div.id = 'sec_' + ep;
  secContainer.appendChild(div);
  var x = []; for (var i = 0; i < __STEPS__; i++) x.push(i);
  var sc = D.sec[ep].sec_cap;
  Plotly.newPlot(div.id, [
    {x: x, y: sc.map(function(r){return r[0]}), name: '用户0', fill: 'tozeroy',
     line: {color: C.cyan, width: 1.5}, fillcolor: 'rgba(0,117,192,0.15)'},
    {x: x, y: sc.map(function(r){return r[1]}), name: '用户1', fill: 'tozeroy',
     line: {color: C.green, width: 1.5}, fillcolor: 'rgba(0,117,0,0.1)'}
  ], Object.assign({}, layout_base, {
    title: {text: '(c) Episode ' + ep, font: {size: 13, color: '#000'}},
    xaxis: {title: {text: '时隙 (Time Slot)', font: {size: 12}}, tickfont: {size: 10}, gridcolor: '#eee'},
    yaxis: {title: {text: '安全容量 (bits/s/Hz)', font: {size: 12}}, tickfont: {size: 10}, gridcolor: '#eee'},
    legend: {x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', font: {size: 10}}
  }));
});

// ====== Fig.5: RIS phase + FAS port ======
var risEps = Object.keys(D.ris).map(Number);
var risContainer = document.getElementById('ris_container');
risEps.forEach(function(ep) {
  if (!D.ris[ep]) return;
  var div = document.createElement('div');
  div.className = 'chart'; div.id = 'ris_' + ep;
  risContainer.appendChild(div);
  var traces = [];
  var x = [];
  var rd = D.ris[ep];
  var sigLen = (rd.ris_signal || []).length;
  for (var i = 0; i < sigLen; i++) x.push(i);
  if (rd.ris_signal && rd.ris_signal.length)
    traces.push({x: x, y: rd.ris_signal, name: 'RIS信号相位', line: {color: C.cyan, width: 1.5}});
  if (rd.ris_jam && rd.ris_jam.length)
    traces.push({x: x, y: rd.ris_jam, name: 'RIS干扰相位', line: {color: C.red, width: 1.5, dash: 'dash'}});
  if (rd.fas_port && rd.fas_port.length)
    traces.push({x: x, y: rd.fas_port, name: 'FAS端口号', yaxis: 'y2', line: {color: C.orange, width: 1.5}});
  Plotly.newPlot(div.id, traces, Object.assign({}, layout_base, {
    title: {text: '(e) Episode ' + ep, font: {size: 13, color: '#000'}},
    xaxis: {title: {text: '时隙 (Time Slot)', font: {size: 12}}, tickfont: {size: 10}, gridcolor: '#eee'},
    yaxis: {title: {text: 'RIS相位 (归一化)', font: {size: 12}}, tickfont: {size: 10}, gridcolor: '#eee', range: [-1.2, 1.2]},
    yaxis2: {title: {text: 'FAS端口号', font: {size: 12, color: C.orange}}, tickfont: {size: 10, color: C.orange}, overlaying: 'y', side: 'right'},
    legend: {x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top', font: {size: 10}},
    height: 380
  }));
});
</script>
</body>
</html>"""

# ============================================================
# Perform replacements
# ============================================================
html = html_template
html = html.replace("__TOTAL_EPISODES__", str(TOTAL_EPISODES))
html = html.replace("__STEPS_PER_EPISODE__", str(STEPS_PER_EPISODE))
html = html.replace("__FINAL100_MEAN__", "%.2f" % final_100_mean)
html = html.replace("__MAX_REWARD__", "%.2f" % max_reward)
html = html.replace("__MAX_EP__", str(max_ep))
html = html.replace("__INIT10_MEAN__", "%.2f" % initial_10_mean)
html = html.replace("__PERF_IMP__", "%.1f" % perf_improvement)
html = html.replace("__MA_WINDOW__", str(MA_WINDOW))
html = html.replace("__STEPS__", str(STEPS_PER_EPISODE))
html = html.replace("__TRAJ_EPS_JSON__", json.dumps(TRAJ_EPS))
html = html.replace("__SELECTED_EPS_JSON__", json.dumps(SELECTED_EPS))
html = html.replace("__EP_HEADERS__", ep_headers)
html = html.replace("__TABLE_ROWS__", table_html_rows)
html = html.replace("__DATA_JSON__", data_json)

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("\nReport generated: %s" % OUTPUT_HTML)
print("  Total episodes: %d" % TOTAL_EPISODES)
print("  Final 100-ep mean reward: %.2f" % final_100_mean)
print("  Max reward: %.2f (ep %d)" % (max_reward, max_ep))
print("  Initial 10-ep mean: %.2f" % initial_10_mean)
print("  Performance improvement: %.1fx" % perf_improvement)
