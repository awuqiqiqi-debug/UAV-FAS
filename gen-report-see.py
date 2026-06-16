"""生成td3_see通用的学术论文级HTML训练报告"""
import json, scipy.io as sio, numpy as np, csv

mat_dir = 'Twin-TD3-main/data/storage/uav_bs_fas/scratch/td3_see_7'

rewards_csv = f'{mat_dir}/training_rewards.csv'
episodes, scores = [], []
with open(rewards_csv, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for i, row in enumerate(reader):
        episodes.append(i)
        scores.append(float(row[0]))

def moving_avg(data, w=20):
    return [float(np.mean(data[max(0,i-w+1):i+1])) for i in range(len(data))]
ma_scores = moving_avg(scores)

selected_eps = [0, 50, 100, 200, 300, 350, 399]
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

entities = {
    'uav_start': [0, 25],
    'user0': [4, 47],
    'user1': [25, 25],
    'attacker0': [47, -4],
    'ris': [20, -20],
}

final_100_avg = float(np.mean(scores[-100:]))
best_idx = int(np.argmax(scores))
best_ep = episodes[best_idx]
best_score = float(max(scores))
init_avg = float(np.mean(scores[:10]))
speedup = final_100_avg / init_avg if init_avg != 0 else 1

j_episodes = json.dumps(episodes)
j_scores = json.dumps([round(s, 4) for s in scores])
j_ma = json.dumps([round(s, 4) for s in ma_scores])

traj_data, cap_data, sec_data, fas_data = {}, {}, {}, {}
for ep in selected_eps:
    if ep in episode_data:
        uav = np.array(episode_data[ep]['UAV_state'])
        traj_data[ep] = {'x': uav[:,0].tolist(), 'y': uav[:,1].tolist()}
        uc = np.array(episode_data[ep]['user_capacity'])
        ac = np.array(episode_data[ep]['attaker_capacity']).squeeze()
        cap_data[ep] = {
            'u0': uc[:,0].tolist(), 'u1': uc[:,1].tolist(),
            'a0': ac[:,0].tolist(), 'a1': ac[:,1].tolist()
        }
for ep in [50, 100, 200, 300, 350, 399]:
    if ep in episode_data:
        sc = np.array(episode_data[ep]['secure_capacity'])
        sec_data[ep] = {'u0': sc[:,0].tolist(), 'u1': sc[:,1].tolist()}

for ep in selected_eps:
    if ep in episode_data:
        fd = {}
        if 'RIS_signal_phase' in episode_data[ep]:
            fd['ris_signal'] = np.array(episode_data[ep]['RIS_signal_phase']).flatten().tolist()
        if 'RIS_jam_phase' in episode_data[ep]:
            fd['ris_jam'] = np.array(episode_data[ep]['RIS_jam_phase']).flatten().tolist()
        bf = episode_data[ep]['beamforming_matrix']
        port0, port1 = [], []
        for i in range(len(bf)):
            p = bf[i][0].flatten()
            port0.append(int(p[0]))
            port1.append(int(p[1]))
        fd['fas_port0'] = port0
        fd['fas_port1'] = port1
        fas_data[ep] = fd

table_rows = []
for key, label, col in [
    ('secure_capacity', '安全容量 (用户0)', 0),
    ('secure_capacity', '安全容量 (用户1)', 1),
    ('user_capacity', '用户容量 (用户0)', 0),
    ('user_capacity', '用户容量 (用户1)', 1),
    ('attaker_capacity', '窃听容量 (→用户0)', 0),
    ('attaker_capacity', '窃听容量 (→用户1)', 1),
]:
    vals = []
    for ep in [0, 50, 100, 200, 300, 350, 399]:
        if ep in episode_data:
            arr = np.array(episode_data[ep][key])
            if arr.ndim == 3:
                v = arr[:, 0, col].mean()
            else:
                v = arr[:, col].mean()
            vals.append(f'{v:.4f}')
        else:
            vals.append('-')
    table_rows.append((label, vals))

rows_html = ''
for label, vals in table_rows:
    cells = ''.join(f'<td>{v}</td>' for v in vals)
    rows_html += f'<tr><td style="text-align:left;font-weight:700">{label}</td>{cells}</tr>\n'

html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>FAS-UAV保密通信TD3训练报告 (td3_see)</title>
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
.note{background:#fff8e1;border-left:4px solid #ffa000;padding:12px 16px;margin:8px 40px;font-size:12px;color:#555;border-radius:0 4px 4px 0}
.note b{color:#e65100}
.footer{text-align:center;padding:20px;font-size:10px;color:#999;border-top:1px solid #ddd;margin-top:20px}
</style>
</head>
<body>
<div class="title-block">
  <h1>流体天线辅助无人机安全通信 — TD3强化学习训练报告</h1>
  <div class="subtitle">FAS-Assisted UAV Secure Communication System &mdash; Twin Delayed TD3 Training Report (td3_see)</div>
  <div class="meta">奖励函数: SEE (安全能量效率) &nbsp;|&nbsp; 训练轮次: ''' + str(len(episodes)) + ''' &nbsp;|&nbsp; 每轮步数: 100 &nbsp;|&nbsp; FAS同时激活端口: 2</div>
</div>
<div class="summary">
  <div class="item"><div class="val">''' + str(len(episodes)) + '''</div><div class="lbl">训练轮次</div></div>
  <div class="item"><div class="val">''' + f'{final_100_avg:.2f}' + '''</div><div class="lbl">最终100轮均分</div></div>
  <div class="item"><div class="val">''' + f'{best_score:.2f}' + '''</div><div class="lbl">最高分 (Ep.''' + str(best_ep) + ''')</div></div>
  <div class="item"><div class="val">''' + f'{init_avg:.2f}' + '''</div><div class="lbl">初始均分(前10轮)</div></div>
  <div class="item"><div class="val">''' + f'{speedup:.1f}' + '''x</div><div class="lbl">性能提升</div></div>
</div>

<div class="figure">
  <div class="fig-label">图1 &nbsp; 训练奖励收敛曲线</div>
  <div class="chart" id="c_reward"></div>
  <div class="fig-caption">Fig.1 &nbsp; Training reward convergence curve. Blue line: moving average (window=20); gray line: per-episode reward.</div>
</div>

<div class="figure">
  <div class="fig-label">图2 &nbsp; 不同训练阶段无人机飞行轨迹</div>
  <div class="grid2" id="traj_container"></div>
  <div class="fig-caption">Fig.2 &nbsp; UAV trajectories at different training stages. Green triangle: start; Red cross: end; Blue line: trajectory; Markers: entities.</div>
</div>

<div class="figure">
  <div class="fig-label">图3 &nbsp; 用户与窃听者信道容量对比</div>
  <div class="note"><b>说明：</b>窃听者对用户0和用户1分别窃听，红色虚线=窃听→用户0，橙色点线=窃听→用户1。两者略有差异：RIS反射路径使用用户k的RIS→用户信道(h_R_k)，不同用户位置导致反射路径不同。直射路径远强于RIS反射路径(~1万倍)，因此差异较小。</div>
  <div class="grid2" id="cap_container"></div>
  <div class="fig-caption">Fig.3 &nbsp; 用户容量(实线) vs 窃听者对各用户的窃听容量(虚线)。红色虚线=窃听→用户0，橙色点线=窃听→用户1。</div>
</div>

<div class="figure">
  <div class="fig-label">图4 &nbsp; 安全容量随时间变化</div>
  <div class="grid2" id="sec_container"></div>
  <div class="fig-caption">Fig.4 &nbsp; Secrecy capacity over time slots.</div>
</div>

<div class="figure">
  <div class="fig-label">图5 &nbsp; RIS相位与FAS双端口选择</div>
  <div class="note"><b>说明：</b>FAS每次同时激活2个端口（Gumbel-Softmax Top-2），图中显示两个端口的选择轨迹（端口A和端口B），取值范围0~11。</div>
  <div class="grid2" id="fas_container"></div>
  <div class="fig-caption">Fig.5 &nbsp; RIS phase (left) and FAS dual-port selection (right). Two active ports per time step.</div>
</div>

<div class="tbl-section">
  <table>
    <caption>表1 &nbsp; 训练关键指标对比</caption>
    <tr><th>指标</th><th>Episode 0</th><th>Episode 50</th><th>Episode 100</th><th>Episode 200</th><th>Episode 300</th><th>Episode 350</th><th>Episode 399</th></tr>
    ''' + rows_html + '''
  </table>
</div>
<div class="footer">FAS-UAV Secure Communication &mdash; TD3 Reinforcement Learning Training Report (td3_see)</div>
<script>
var C = {blue:'#000000',red:'#C00000',green:'#007500',cyan:'#0075C0',magenta:'#C000C0',orange:'#ED7D31',gray:'#808080',lightgray:'#BFBFBF'};
var layout_base = {font:{family:'Times New Roman,serif',size:12,color:'#000'},paper_bgcolor:'#fff',plot_bgcolor:'#fff',margin:{t:35,b:45,l:55,r:20},height:320,legend:{bgcolor:'rgba(255,255,255,0.9)',bordercolor:'#ccc',borderwidth:1,font:{size:11}}};
var episodes = ''' + j_episodes + ''';
var scores = ''' + j_scores + ''';
var ma_scores = ''' + j_ma + ''';
var traj = ''' + json.dumps(traj_data) + ''';
var cap = ''' + json.dumps(cap_data) + ''';
var sec = ''' + json.dumps(sec_data) + ''';
var fasData = ''' + json.dumps(fas_data) + ''';

Plotly.newPlot('c_reward', [
  {x:episodes, y:scores, mode:'lines', name:'逐轮奖励', line:{color:C.lightgray,width:0.8}},
  {x:episodes, y:ma_scores, mode:'lines', name:'滑动均值 (w=20)', line:{color:C.cyan,width:2.5}}
], Object.assign({}, layout_base, {
  height:350,
  xaxis:{title:{text:'训练轮次 (Episode)',font:{size:13}},tickfont:{size:11},gridcolor:'#eee',dtick:10},
  yaxis:{title:{text:'奖励 (Reward/SEE)',font:{size:13}},tickfont:{size:11},gridcolor:'#eee'}
}));

var trajEps = [0, 50, 100, 200, 300, 350, 399];
var trajContainer = document.getElementById('traj_container');
var entityMarkers = [
  {x:[4],y:[47],name:'用户0',mode:'markers',marker:{size:11,color:'#fff',symbol:'circle',line:{color:C.green,width:2}}},
  {x:[25],y:[25],name:'用户1',mode:'markers',marker:{size:11,color:'#fff',symbol:'diamond',line:{color:C.green,width:2}}},
  {x:[47],y:[-4],name:'窃听者',mode:'markers',marker:{size:11,color:C.red,symbol:'cross',line:{color:C.red,width:2}}},
  {x:[20],y:[-20],name:'RIS',mode:'markers',marker:{size:11,color:'#fff',symbol:'square',line:{color:C.magenta,width:2}}}
];
trajEps.forEach(function(ep){
  if(!traj[ep])return;
  var div=document.createElement('div');div.className='chart';div.id='traj_'+ep;trajContainer.appendChild(div);
  var traces=[
    {x:traj[ep].x,y:traj[ep].y,mode:'lines',name:'UAV轨迹',line:{color:C.cyan,width:2}},
    {x:[traj[ep].x[0]],y:[traj[ep].y[0]],mode:'markers',name:'起点',marker:{size:10,color:C.green,symbol:'triangle-up'}},
    {x:[traj[ep].x[traj[ep].x.length-1]],y:[traj[ep].y[traj[ep].y.length-1]],mode:'markers',name:'终点',marker:{size:10,color:C.red,symbol:'x',line:{width:2}}}
  ].concat(entityMarkers);
  Plotly.newPlot(div.id,traces,Object.assign({},layout_base,{
    height:340,title:{text:'Episode '+ep,font:{size:13,color:'#000'}},
    xaxis:{title:{text:'X (m)',font:{size:12}},range:[-80,80],dtick:20,tickfont:{size:10},gridcolor:'#eee'},
    yaxis:{title:{text:'Y (m)',font:{size:12}},range:[-60,60],dtick:20,tickfont:{size:10},gridcolor:'#eee'},
    shapes:[{type:'rect',x0:-50,x1:50,y0:-50,y1:50,line:{color:'#999',width:1,dash:'dot'},fillcolor:'rgba(0,0,0,0)'}],
    legend:{x:0.01,y:0.99,xanchor:'left',yanchor:'top',font:{size:9}}
  }));
});

var capEps=[0,50,100,200,300,350,399];var capContainer=document.getElementById('cap_container');
capEps.forEach(function(ep){
  if(!cap[ep])return;
  var div=document.createElement('div');div.className='chart';div.id='cap_'+ep;capContainer.appendChild(div);
  var x=[];for(var i=0;i<100;i++)x.push(i);
  Plotly.newPlot(div.id,[
    {x:x,y:cap[ep].u0,name:'用户0容量',line:{color:C.cyan,width:2}},
    {x:x,y:cap[ep].u1,name:'用户1容量',line:{color:C.green,width:2}},
    {x:x,y:cap[ep].a0,name:'窃听→用户0',line:{color:C.red,width:2,dash:'dash'}},
    {x:x,y:cap[ep].a1,name:'窃听→用户1',line:{color:C.orange,width:2,dash:'dot'}}
  ],Object.assign({},layout_base,{
    title:{text:'Episode '+ep,font:{size:13,color:'#000'}},
    xaxis:{title:{text:'时隙 (Time Slot)',font:{size:12}},tickfont:{size:10},gridcolor:'#eee'},
    yaxis:{title:{text:'容量 (bits/s/Hz)',font:{size:12}},tickfont:{size:10},gridcolor:'#eee'},
    legend:{x:0.01,y:0.99,xanchor:'left',yanchor:'top',font:{size:10}}
  }));
});

var secEps=[50,100,200,300,350,399];var secContainer=document.getElementById('sec_container');
secEps.forEach(function(ep){
  if(!sec[ep])return;
  var div=document.createElement('div');div.className='chart';div.id='sec_'+ep;secContainer.appendChild(div);
  var x=[];for(var i=0;i<100;i++)x.push(i);
  Plotly.newPlot(div.id,[
    {x:x,y:sec[ep].u0,name:'用户0',fill:'tozeroy',line:{color:C.cyan,width:1.5},fillcolor:'rgba(0,117,192,0.15)'},
    {x:x,y:sec[ep].u1,name:'用户1',fill:'tozeroy',line:{color:C.green,width:1.5},fillcolor:'rgba(0,117,0,0.1)'}
  ],Object.assign({},layout_base,{
    title:{text:'Episode '+ep,font:{size:13,color:'#000'}},
    xaxis:{title:{text:'时隙 (Time Slot)',font:{size:12}},tickfont:{size:10},gridcolor:'#eee'},
    yaxis:{title:{text:'安全容量 (bits/s/Hz)',font:{size:12}},tickfont:{size:10},gridcolor:'#eee'},
    legend:{x:0.01,y:0.99,xanchor:'left',yanchor:'top',font:{size:10}}
  }));
});

var fasEps=Object.keys(fasData).map(Number);
var fasContainer=document.getElementById('fas_container');
fasEps.forEach(function(ep){
  if(!fasData[ep])return;
  var risDiv=document.createElement('div');risDiv.className='chart';risDiv.id='ris_'+ep;fasContainer.appendChild(risDiv);
  var x=[];for(var i=0;i<(fasData[ep].ris_signal||[]).length;i++)x.push(i);
  var risTraces=[];
  if(fasData[ep].ris_signal){risTraces.push({x:x,y:fasData[ep].ris_signal,name:'RIS信号相位',line:{color:C.cyan,width:1.5}});}
  if(fasData[ep].ris_jam){risTraces.push({x:x,y:fasData[ep].ris_jam,name:'RIS干扰相位',line:{color:C.red,width:1.5,dash:'dash'}});}
  Plotly.newPlot(risDiv.id,risTraces,Object.assign({},layout_base,{
    title:{text:'RIS相位 (Ep '+ep+')',font:{size:13,color:'#000'}},
    xaxis:{title:{text:'时隙 (Time Slot)',font:{size:12}},tickfont:{size:10},gridcolor:'#eee'},
    yaxis:{title:{text:'RIS相位 (归一化)',font:{size:12}},tickfont:{size:10},gridcolor:'#eee',range:[-1.2,1.2]},
    legend:{x:0.01,y:0.99,xanchor:'left',yanchor:'top',font:{size:10}},
    height:360
  }));

  var fasDiv=document.createElement('div');fasDiv.className='chart';fasDiv.id='fas_'+ep;fasContainer.appendChild(fasDiv);
  var fasTraces=[
    {x:x,y:fasData[ep].fas_port0,name:'端口A (第1活跃)',mode:'lines+markers',marker:{size:4},line:{color:C.cyan,width:1.5}},
    {x:x,y:fasData[ep].fas_port1,name:'端口B (第2活跃)',mode:'lines+markers',marker:{size:4},line:{color:C.orange,width:1.5,dash:'dot'}}
  ];
  Plotly.newPlot(fasDiv.id,fasTraces,Object.assign({},layout_base,{
    title:{text:'FAS双端口选择 (Ep '+ep+')',font:{size:13,color:'#000'}},
    xaxis:{title:{text:'时隙 (Time Slot)',font:{size:12}},tickfont:{size:10},gridcolor:'#eee'},
    yaxis:{title:{text:'FAS端口号 (0~11)',font:{size:12}},tickfont:{size:10},gridcolor:'#eee',dtick:1,range:[-0.5,11.5]},
    legend:{x:0.01,y:0.99,xanchor:'left',yanchor:'top',font:{size:10}},
    height:360
  }));
});
</script>
</body>
</html>'''

out_path = f'{mat_dir}/training_report_see.html'
with open(out_path, 'w', encoding='utf-8-sig') as f:
    f.write(html)
print(f'Report saved: {out_path}')
print(f'Stats: episodes={len(episodes)}, final_avg={final_100_avg:.2f}, best={best_score:.2f} (ep {best_ep})')
