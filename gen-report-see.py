"""生成td3_see训练报告 - 1000轮版本 (新位置配置)"""
import json, scipy.io as sio, numpy as np, csv, os

mat_dir = 'Twin-TD3-main/data/storage/uav_bs_fas/scratch/td3_see_14'

# ========== 读取训练数据 ==========
rewards_csv = f'{mat_dir}/training_rewards.csv'
episodes, scores = [], []
with open(rewards_csv, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for i, row in enumerate(reader):
        episodes.append(i)
        scores.append(float(row[0]))

def moving_avg(data, w=50):
    return [float(np.mean(data[max(0,i-w+1):i+1])) for i in range(len(data))]

ma50 = moving_avg(scores, w=50)
ma100 = moving_avg(scores, w=100)

# ========== 选择关键episode (每100轮) ==========
selected_eps = list(range(0, 1000, 100)) + [999]  # 0, 100, 200, ..., 900, 999
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

# ========== 计算统计指标 ==========
final_100_avg = float(np.mean(scores[-100:]))
best_idx = int(np.argmax(scores))
best_ep = episodes[best_idx]
best_score = float(max(scores))
init_avg = float(np.mean(scores[:10]))
speedup = final_100_avg / init_avg if init_avg != 0 else 1

# ========== 准备图表数据 ==========
j_episodes = json.dumps(episodes)
j_scores = json.dumps([round(s, 2) for s in scores])
j_ma50 = json.dumps([round(s, 2) for s in ma50])
j_ma100 = json.dumps([round(s, 2) for s in ma100])

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
        sc = np.array(episode_data[ep]['secure_capacity'])
        sec_data[ep] = {'u0': sc[:,0].tolist(), 'u1': sc[:,1].tolist()}
        fd = {}
        if 'RIS_signal_phase' in episode_data[ep]:
            fd['ris_signal'] = np.array(episode_data[ep]['RIS_signal_phase']).flatten().tolist()
        if 'RIS_jam_phase' in episode_data[ep]:
            fd['ris_jam'] = np.array(episode_data[ep]['RIS_jam_phase']).flatten().tolist()
        if 'jam_ratio' in episode_data[ep]:
            fd['jam_ratio'] = np.array(episode_data[ep]['jam_ratio']).flatten().tolist()
        if 'user1_ratio' in episode_data[ep]:
            fd['user1_ratio'] = np.array(episode_data[ep]['user1_ratio']).flatten().tolist()
        if 'user2_ratio' in episode_data[ep]:
            fd['user2_ratio'] = np.array(episode_data[ep]['user2_ratio']).flatten().tolist()
        if 'ris_allocation' in episode_data[ep]:
            alloc = np.array(episode_data[ep]['ris_allocation'])
            # 压缩中间维度: (steps, 1, 64) -> (steps, 64)
            if alloc.ndim == 3:
                alloc = alloc[:, 0, :]
            fd['ris_allocation'] = alloc.tolist()
        if 'see' in episode_data[ep]:
            fd['see'] = np.array(episode_data[ep]['see']).flatten().tolist()
        bf = episode_data[ep]['beamforming_matrix']
        port0, port1 = [], []
        for i in range(len(bf)):
            p = bf[i][0].flatten()
            port0.append(int(p[0]))
            port1.append(int(p[1]))
        fd['fas_port0'] = port0
        fd['fas_port1'] = port1
        fas_data[ep] = fd

# ========== 生成汇总表 ==========
table_rows_html = ''
for key, label, col in [
    ('secure_capacity', '安全容量 (用户0)', 0),
    ('secure_capacity', '安全容量 (用户1)', 1),
    ('user_capacity', '用户容量 (用户0)', 0),
    ('user_capacity', '用户容量 (用户1)', 1),
    ('attaker_capacity', '窃听容量 (→用户0)', 0),
    ('attaker_capacity', '窃听容量 (→用户1)', 1),
]:
    cells = ''
    for ep in selected_eps:
        if ep in episode_data:
            arr = np.array(episode_data[ep][key])
            v = arr[:, 0, col].mean() if arr.ndim == 3 else arr[:, col].mean()
            cells += '<td>' + f'{v:.2f}' + '</td>'
        else:
            cells += '<td>-</td>'
    table_rows_html += '<tr><td style="text-align:left;font-weight:700">' + label + '</td>' + cells + '</tr>\n'

ep_headers = ''.join('<th>Ep ' + str(ep) + '</th>' for ep in selected_eps)

# ========== HTML模板 (不用f-string避免花括号冲突) ==========
html = '<!DOCTYPE html>\n'
html += '<html lang="zh-CN">\n<head>\n'
html += '<meta charset="UTF-8">\n'
html += '<title>FAS-UAV TD3训练报告 (1100轮, 新位置配置)</title>\n'
html += '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n'
html += '<style>\n'
html += '@import url("https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap");\n'
html += '*{margin:0;padding:0;box-sizing:border-box}\n'
html += 'body{font-family:"Noto Sans SC","Helvetica Neue",sans-serif;background:#f8f9fa;color:#1a1a2e;line-height:1.6}\n'
html += '.header{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);color:#fff;padding:48px 60px 36px;text-align:center}\n'
html += '.header h1{font-size:26px;font-weight:700;letter-spacing:2px;margin-bottom:8px}\n'
html += '.header .subtitle{font-size:14px;color:#b8b8d0;font-weight:300}\n'
html += '.header .meta{font-size:12px;color:#8888aa;margin-top:12px}\n'
html += '.header .meta span{margin:0 12px}\n'
html += '.stats{display:flex;justify-content:center;gap:24px;padding:28px 40px;background:#fff;border-bottom:1px solid #e8e8e8;flex-wrap:wrap}\n'
html += '.stats .card{text-align:center;min-width:130px;padding:16px 20px;border-radius:8px;background:#f0f4ff}\n'
html += '.stats .val{font-size:28px;font-weight:700;color:#302b63}\n'
html += '.stats .lbl{font-size:11px;color:#666;margin-top:4px;font-weight:500}\n'
html += '.section{padding:24px 40px;background:#fff;margin:16px 40px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,0.06)}\n'
html += '.section-title{font-size:15px;font-weight:700;color:#302b63;margin-bottom:12px;padding-bottom:8px;border-bottom:2px solid #302b63;display:inline-block}\n'
html += '.section-caption{font-size:11px;color:#888;text-align:center;margin-top:8px;font-style:italic}\n'
html += '.chart{background:#fff;border:1px solid #e8e8e8;border-radius:6px;padding:6px;margin:8px 0}\n'
html += '.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}\n'
html += '@media(max-width:900px){.grid2{grid-template-columns:1fr}}\n'
html += '.tbl{width:100%;border-collapse:collapse;font-size:12px;margin-top:12px}\n'
html += '.tbl th,.tbl td{padding:10px 14px;border:1px solid #e0e0e0;text-align:center}\n'
html += '.tbl th{background:#302b63;color:#fff;font-weight:600;font-size:11px}\n'
html += '.tbl td{background:#fafbff}\n'
html += '.tbl caption{font-size:13px;font-weight:700;color:#302b63;margin-bottom:8px;text-align:left}\n'
html += '.note{background:#fff8e1;border-left:4px solid #ffa000;padding:12px 16px;margin:12px 0;font-size:12px;color:#666;border-radius:0 6px 6px 0}\n'
html += '.note b{color:#e65100}\n'
html += '.footer{text-align:center;padding:24px;font-size:10px;color:#999;background:#fff;margin-top:16px}\n'
html += '</style>\n</head>\n<body>\n'

# 标题区
html += '<div class="header">\n'
html += '  <h1>流体天线辅助无人机安全通信</h1>\n'
html += '  <div class="subtitle">FAS-Assisted UAV Secure Communication &mdash; Twin Delayed TD3 Training Report (New Layout)</div>\n'
html += '  <div class="meta">\n'
html += '    <span>算法: TD3</span>\n'
html += '    <span>奖励: SEE</span>\n'
html += '    <span>轮次: ' + str(len(episodes)) + '</span>\n'
html += '    <span>每轮步数: 100</span>\n'
html += '    <span>FAS端口: 2</span>\n'
html += '    <span>位置: UAV(0,-30,50) User0(-15,30,0) User1(15,30,0) Attacker(10,35,0) RIS(0,20,12.5)</span>\n'
html += '  </div>\n</div>\n'

# 统计卡片
html += '<div class="stats">\n'
html += '  <div class="card"><div class="val">' + str(len(episodes)) + '</div><div class="lbl">训练轮次</div></div>\n'
html += '  <div class="card"><div class="val">' + f'{final_100_avg:.1f}' + '</div><div class="lbl">最终100轮均分</div></div>\n'
html += '  <div class="card"><div class="val">' + f'{best_score:.1f}' + '</div><div class="lbl">最高分 (Ep.' + str(best_ep) + ')</div></div>\n'
html += '  <div class="card"><div class="val">' + f'{init_avg:.1f}' + '</div><div class="lbl">初始均分</div></div>\n'
html += '  <div class="card"><div class="val">' + f'{speedup:.1f}' + 'x</div><div class="lbl">性能提升</div></div>\n'
html += '</div>\n'

# 图1
html += '<div class="section">\n'
html += '  <div class="section-title">图1 &nbsp; 训练奖励收敛曲线</div>\n'
html += '  <div class="chart" id="c_reward" style="height:380px"></div>\n'
html += '  <div class="section-caption">Fig.1 Training reward convergence. Orange: per-episode; Blue: MA-50; Green: MA-100.</div>\n'
html += '</div>\n'

# 图2
html += '<div class="section">\n'
html += '  <div class="section-title">图2 &nbsp; 不同训练阶段无人机飞行轨迹</div>\n'
html += '  <div class="grid2" id="traj_container"></div>\n'
html += '  <div class="section-caption">Fig.2 UAV trajectories at different training stages.</div>\n'
html += '</div>\n'

# 图3
html += '<div class="section">\n'
html += '  <div class="section-title">图3 &nbsp; 用户容量 vs 窃听容量</div>\n'
html += '  <div class="note"><b>说明：</b>实线=用户容量，虚线=窃听者对各用户的窃听容量。</div>\n'
html += '  <div class="grid2" id="cap_container"></div>\n'
html += '  <div class="section-caption">Fig.3 User capacity (solid) vs eavesdropper capacity (dashed).</div>\n'
html += '</div>\n'

# 图4
html += '<div class="section">\n'
html += '  <div class="section-title">图4 &nbsp; 安全容量随时间变化</div>\n'
html += '  <div class="grid2" id="sec_container"></div>\n'
html += '  <div class="section-caption">Fig.4 Secrecy capacity = User capacity - Eavesdropper capacity.</div>\n'
html += '</div>\n'

# 图4.5 SEE曲线
html += '<div class="section">\n'
html += '  <div class="section-title">图4.5 &nbsp; 安全能效 (SEE) 变化</div>\n'
html += '  <div class="note"><b>说明：</b>SEE = 安全速率 / 能耗 (bits/J)，衡量单位能耗的安全性能。</div>\n'
html += '  <div class="chart" id="see_chart" style="height:350px"></div>\n'
html += '  <div class="section-caption">Fig.4.5 Secure Energy Efficiency (SEE) = Secrecy Rate / Energy Consumption.</div>\n'
html += '</div>\n'

# 图5
html += '<div class="section">\n'
html += '  <div class="section-title">图5 &nbsp; RIS相位与FAS端口选择</div>\n'
html += '  <div class="note"><b>说明：</b>FAS每次同时激活2个端口（Gumbel-Softmax Top-2），取值范围0~11。</div>\n'
html += '  <div class="grid2" id="fas_container"></div>\n'
html += '  <div class="section-caption">Fig.5 RIS phase (left) and FAS dual-port selection (right).</div>\n'
html += '</div>\n'

# 图6
html += '<div class="section">\n'
html += '  <div class="section-title">图6 &nbsp; RIS资源分配比例变化</div>\n'
html += '  <div class="note"><b>说明：</b>三条曲线分别表示：η(干扰窃听者)、β₁(反射给User1)、β₂(反射给User2)。</div>\n'
html += '  <div class="chart" id="ratio_chart" style="height:450px"></div>\n'
html += '  <div class="section-caption">Fig.6 RIS resource allocation ratios (η, β₁, β₂).</div>\n'
html += '</div>\n'

# 图7
html += '<div class="section">\n'
html += '  <div class="section-title">图7 &nbsp; RIS单元分配可视化</div>\n'
html += '  <div class="note"><b>说明：</b>热力图显示64个RIS单元的分配：绿色=反射给User1，蓝色=反射给User2，红色=干扰窃听者。每行一个episode。</div>\n'
html += '  <div class="chart" id="allocation_chart" style="height:400px"></div>\n'
html += '  <div class="section-caption">Fig.7 RIS unit allocation heatmap (green=User1, blue=User2, red=jam).</div>\n'
html += '</div>\n'

# 表1
html += '<div class="section">\n'
html += '  <table class="tbl">\n'
html += '    <caption>表1 &nbsp; 关键指标随训练轮次变化</caption>\n'
html += '    <tr><th>指标</th>' + ep_headers + '</tr>\n'
html += '    ' + table_rows_html + '\n'
html += '  </table>\n</div>\n'

html += '<div class="footer">FAS-UAV Secure Communication &mdash; Twin-TD3 Training Report (New Layout) &mdash; ' + str(len(episodes)) + ' Episodes</div>\n'

# ========== JavaScript部分 (不用f-string) ==========
html += '<script>\n'
html += 'var C={cyan:"#0075C0",green:"#007500",red:"#C00000",orange:"#ED7D31",magenta:"#C000C0",gray:"#808080",lightgray:"#BFBFBF",blue:"#000000"};\n'
html += 'var layout_base={font:{family:"Helvetica Neue,Arial,sans-serif",size:11,color:"#333"},paper_bgcolor:"#fff",plot_bgcolor:"#fafbff",margin:{t:40,b:45,l:55,r:20},height:300,legend:{bgcolor:"rgba(255,255,255,0.95)",bordercolor:"#ddd",borderwidth:1,font:{size:10}}};\n'
html += 'var episodes=' + j_episodes + ';\n'
html += 'var scores=' + j_scores + ';\n'
html += 'var ma50=' + j_ma50 + ';\n'
html += 'var ma100=' + j_ma100 + ';\n'
html += 'var traj=' + json.dumps(traj_data) + ';\n'
html += 'var cap=' + json.dumps(cap_data) + ';\n'
html += 'var sec=' + json.dumps(sec_data) + ';\n'
html += 'var fasData=' + json.dumps(fas_data) + ';\n'

# 图1: 收敛曲线
html += 'Plotly.newPlot("c_reward",[\n'
html += '  {x:episodes,y:scores,mode:"lines",name:"逐轮奖励",line:{color:C.lightgray,width:0.6}},\n'
html += '  {x:episodes,y:ma50,mode:"lines",name:"MA-50",line:{color:C.cyan,width:2.5}},\n'
html += '  {x:episodes,y:ma100,mode:"lines",name:"MA-100",line:{color:C.green,width:2,dash:"dash"}}\n'
html += '],Object.assign({},layout_base,{\n'
html += '  height:380,\n'
html += '  xaxis:{title:{text:"训练轮次 (Episode)",font:{size:12}},tickfont:{size:10},gridcolor:"#eee",dtick:100},\n'
html += '  yaxis:{title:{text:"奖励 (Reward)",font:{size:12}},tickfont:{size:10},gridcolor:"#eee"}\n'
html += '}));\n'

# 图2: 轨迹
html += 'var trajEps=' + json.dumps(selected_eps) + ';\n'
html += 'var trajContainer=document.getElementById("traj_container");\n'
html += 'var entityMarkers=[\n'
html += '  {x:[-15],y:[30],name:"用户0",mode:"markers",marker:{size:10,color:"#fff",symbol:"circle",line:{color:C.green,width:2}}},\n'
html += '  {x:[15],y:[30],name:"用户1",mode:"markers",marker:{size:10,color:"#fff",symbol:"diamond",line:{color:C.green,width:2}}},\n'
html += '  {x:[10],y:[35],name:"窃听者",mode:"markers",marker:{size:10,color:C.red,symbol:"cross",line:{color:C.red,width:2}}},\n'
html += '  {x:[0],y:[20],name:"RIS",mode:"markers",marker:{size:10,color:"#fff",symbol:"square",line:{color:C.magenta,width:2}}}\n'
html += '];\n'
html += 'trajEps.forEach(function(ep){\n'
html += '  if(!traj[ep])return;\n'
html += '  var div=document.createElement("div");div.className="chart";div.id="traj_"+ep;trajContainer.appendChild(div);\n'
html += '  Plotly.newPlot(div.id,[\n'
html += '    {x:traj[ep].x,y:traj[ep].y,mode:"lines",name:"UAV轨迹",line:{color:C.cyan,width:2}},\n'
html += '    {x:[traj[ep].x[0]],y:[traj[ep].y[0]],mode:"markers",name:"起点",marker:{size:9,color:C.green,symbol:"triangle-up"}},\n'
html += '    {x:[traj[ep].x[traj[ep].x.length-1]],y:[traj[ep].y[traj[ep].y.length-1]],mode:"markers",name:"终点",marker:{size:9,color:C.red,symbol:"x",line:{width:2}}}\n'
html += '  ].concat(entityMarkers),Object.assign({},layout_base,{\n'
html += '    height:320,title:{text:"Episode "+ep,font:{size:12,color:"#333"}},\n'
html += '    xaxis:{title:{text:"X (m)",font:{size:11}},range:[-50,50],dtick:10,tickfont:{size:9},gridcolor:"#eee"},\n'
html += '    yaxis:{title:{text:"Y (m)",font:{size:11}},range:[-40,60],dtick:10,tickfont:{size:9},gridcolor:"#eee"},\n'
html += '    shapes:[{type:"rect",x0:-50,x1:50,y0:-50,y1:50,line:{color:"#bbb",width:1,dash:"dot"},fillcolor:"rgba(0,0,0,0)"}],\n'
html += '    legend:{x:0.01,y:0.99,xanchor:"left",yanchor:"top",font:{size:9}}\n'
html += '  }));\n'
html += '});\n'

# 图3: 容量对比
html += 'var capEps=' + json.dumps(selected_eps) + ';\n'
html += 'var capContainer=document.getElementById("cap_container");\n'
html += 'capEps.forEach(function(ep){\n'
html += '  if(!cap[ep])return;\n'
html += '  var div=document.createElement("div");div.className="chart";div.id="cap_"+ep;capContainer.appendChild(div);\n'
html += '  var x=[];for(var i=0;i<100;i++)x.push(i);\n'
html += '  Plotly.newPlot(div.id,[\n'
html += '    {x:x,y:cap[ep].u0,name:"用户0",line:{color:C.cyan,width:2}},\n'
html += '    {x:x,y:cap[ep].u1,name:"用户1",line:{color:C.green,width:2}},\n'
html += '    {x:x,y:cap[ep].a0,name:"窃听→用户0",line:{color:C.red,width:2,dash:"dash"}},\n'
html += '    {x:x,y:cap[ep].a1,name:"窃听→用户1",line:{color:C.orange,width:2,dash:"dot"}}\n'
html += '  ],Object.assign({},layout_base,{\n'
html += '    title:{text:"Episode "+ep,font:{size:12,color:"#333"}},\n'
html += '    xaxis:{title:{text:"时隙",font:{size:11}},tickfont:{size:9},gridcolor:"#eee"},\n'
html += '    yaxis:{title:{text:"容量 (bits/s/Hz)",font:{size:11}},tickfont:{size:9},gridcolor:"#eee"},\n'
html += '    legend:{x:0.01,y:0.99,xanchor:"left",yanchor:"top",font:{size:9}}\n'
html += '  }));\n'
html += '});\n'

# 图4: 安全容量
html += 'var secEps=' + json.dumps(selected_eps) + ';\n'
html += 'var secContainer=document.getElementById("sec_container");\n'
html += 'secEps.forEach(function(ep){\n'
html += '  if(!sec[ep])return;\n'
html += '  var div=document.createElement("div");div.className="chart";div.id="sec_"+ep;secContainer.appendChild(div);\n'
html += '  var x=[];for(var i=0;i<100;i++)x.push(i);\n'
html += '  Plotly.newPlot(div.id,[\n'
html += '    {x:x,y:sec[ep].u0,name:"用户0",fill:"tozeroy",line:{color:C.cyan,width:1.5},fillcolor:"rgba(0,117,192,0.12)"},\n'
html += '    {x:x,y:sec[ep].u1,name:"用户1",fill:"tozeroy",line:{color:C.green,width:1.5},fillcolor:"rgba(0,117,0,0.08)"}\n'
html += '  ],Object.assign({},layout_base,{\n'
html += '    title:{text:"Episode "+ep,font:{size:12,color:"#333"}},\n'
html += '    xaxis:{title:{text:"时隙",font:{size:11}},tickfont:{size:9},gridcolor:"#eee"},\n'
html += '    yaxis:{title:{text:"安全容量 (bits/s/Hz)",font:{size:11}},tickfont:{size:9},gridcolor:"#eee"},\n'
html += '    legend:{x:0.01,y:0.99,xanchor:"left",yanchor:"top",font:{size:9}}\n'
html += '  }));\n'
html += '});\n'

# 图5: RIS与FAS
html += 'var fasEps=Object.keys(fasData).map(Number);\n'

# SEE曲线 (需要在fasEps定义之后)
html += 'var seeEps=[];\n'
html += 'var seeValues=[];\n'
html += 'fasEps.forEach(function(ep){\n'
html += '  if(fasData[ep] && fasData[ep].see){\n'
html += '    var avg_see=fasData[ep].see.reduce(function(a,b){return a+b},0)/fasData[ep].see.length;\n'
html += '    seeEps.push(ep);\n'
html += '    seeValues.push(avg_see);\n'
html += '  }\n'
html += '});\n'
html += 'Plotly.newPlot("see_chart",[\n'
html += '  {x:seeEps,y:seeValues,mode:"lines+markers",name:"平均SEE",line:{color:"#FF6B00",width:2.5},marker:{size:8}}\n'
html += '],Object.assign({},layout_base,{\n'
html += '  height:350,\n'
html += '  xaxis:{title:{text:"训练轮次 (Episode)",font:{size:12}},tickfont:{size:10},gridcolor:"#eee",dtick:2},\n'
html += '  yaxis:{title:{text:"SEE (bits/J)",font:{size:12}},tickfont:{size:10},gridcolor:"#eee"}\n'
html += '}));\n'

html += 'var fasContainer=document.getElementById("fas_container");\n'
html += 'fasEps.forEach(function(ep){\n'
html += '  if(!fasData[ep])return;\n'
html += '  var risDiv=document.createElement("div");risDiv.className="chart";risDiv.id="ris_"+ep;fasContainer.appendChild(risDiv);\n'
html += '  var x=[];for(var i=0;i<(fasData[ep].ris_signal||[]).length;i++)x.push(i);\n'
html += '  var risTraces=[];\n'
html += '  if(fasData[ep].ris_signal)risTraces.push({x:x,y:fasData[ep].ris_signal,name:"RIS信号相位",line:{color:C.cyan,width:1.5}});\n'
html += '  if(fasData[ep].ris_jam)risTraces.push({x:x,y:fasData[ep].ris_jam,name:"RIS干扰相位",line:{color:C.red,width:1.5,dash:"dash"}});\n'
html += '  Plotly.newPlot(risDiv.id,risTraces,Object.assign({},layout_base,{\n'
html += '    title:{text:"RIS相位 (Ep "+ep+")",font:{size:12,color:"#333"}},\n'
html += '    xaxis:{title:{text:"时隙",font:{size:11}},tickfont:{size:9},gridcolor:"#eee"},\n'
html += '    yaxis:{title:{text:"相位 (归一化)",font:{size:11}},tickfont:{size:9},gridcolor:"#eee",range:[-1.2,1.2]},\n'
html += '    legend:{x:0.01,y:0.99,xanchor:"left",yanchor:"top",font:{size:9}},height:340\n'
html += '  }));\n'
html += '  var fasDiv=document.createElement("div");fasDiv.className="chart";fasDiv.id="fas_"+ep;fasContainer.appendChild(fasDiv);\n'
html += '  Plotly.newPlot(fasDiv.id,[\n'
html += '    {x:x,y:fasData[ep].fas_port0,name:"端口A",mode:"lines+markers",marker:{size:3},line:{color:C.cyan,width:1.5}},\n'
html += '    {x:x,y:fasData[ep].fas_port1,name:"端口B",mode:"lines+markers",marker:{size:3},line:{color:C.orange,width:1.5,dash:"dot"}}\n'
html += '  ],Object.assign({},layout_base,{\n'
html += '    title:{text:"FAS端口选择 (Ep "+ep+")",font:{size:12,color:"#333"}},\n'
html += '    xaxis:{title:{text:"时隙",font:{size:11}},tickfont:{size:9},gridcolor:"#eee"},\n'
html += '    yaxis:{title:{text:"端口号 (0-11)",font:{size:11}},tickfont:{size:9},gridcolor:"#eee",dtick:1,range:[-0.5,11.5]},\n'
html += '    legend:{x:0.01,y:0.99,xanchor:"left",yanchor:"top",font:{size:9}},height:340\n'
html += '  }));\n'
html += '});\n'

# 图6: 三类比例变化
html += 'var ratioEps=[];\n'
html += 'var etaValues=[], u1Values=[], u2Values=[];\n'
html += 'fasEps.forEach(function(ep){\n'
html += '  if(fasData[ep] && fasData[ep].jam_ratio){\n'
html += '    var avg_eta=fasData[ep].jam_ratio.reduce(function(a,b){return a+b},0)/fasData[ep].jam_ratio.length;\n'
html += '    var avg_u1=(fasData[ep].user1_ratio||[]).reduce(function(a,b){return a+b},0)/Math.max(1,(fasData[ep].user1_ratio||[]).length);\n'
html += '    var avg_u2=(fasData[ep].user2_ratio||[]).reduce(function(a,b){return a+b},0)/Math.max(1,(fasData[ep].user2_ratio||[]).length);\n'
html += '    ratioEps.push(ep);\n'
html += '    etaValues.push(avg_eta);\n'
html += '    u1Values.push(avg_u1);\n'
html += '    u2Values.push(avg_u2);\n'
html += '  }\n'
html += '});\n'
html += 'Plotly.newPlot("ratio_chart",[\n'
html += '  {x:ratioEps,y:etaValues,mode:"lines+markers",name:"η (干扰)",line:{color:C.red,width:2.5},marker:{size:8}},\n'
html += '  {x:ratioEps,y:u1Values,mode:"lines+markers",name:"β₁ (User1)",line:{color:"#00AA00",width:2.5},marker:{size:8}},\n'
html += '  {x:ratioEps,y:u2Values,mode:"lines+markers",name:"β₂ (User2)",line:{color:C.cyan,width:2.5},marker:{size:8}},\n'
html += '  {x:[0,1000],y:[0.33,0.33],mode:"lines",name:"均衡线",line:{color:C.gray,width:1,dash:"dash"}}\n'
html += '],Object.assign({},layout_base,{\n'
html += '  height:450,\n'
html += '  xaxis:{title:{text:"训练轮次 (Episode)",font:{size:12}},tickfont:{size:10},gridcolor:"#eee",dtick:100},\n'
html += '  yaxis:{title:{text:"分配比例",font:{size:12}},tickfont:{size:10},gridcolor:"#eee",range:[0,0.6]},\n'
html += '  legend:{x:0.01,y:0.99,xanchor:"left",yanchor:"top",font:{size:10}}\n'
html += '}));\n'

# 图7: RIS单元分配热力图 (三类)
html += 'var allocEps=[];\n'
html += 'var allocData=[];\n'
html += 'fasEps.forEach(function(ep){\n'
html += '  if(fasData[ep] && fasData[ep].ris_allocation){\n'
html += '    allocEps.push("Ep "+ep);\n'
html += '    allocData.push(fasData[ep].ris_allocation[0] || fasData[ep].ris_allocation);\n'
html += '  }\n'
html += '});\n'
html += 'Plotly.newPlot("allocation_chart",[\n'
html += '  {z:allocData,x:Array.from({length:64},function(_,i){return i}),y:allocEps,\n'
html += '   type:"heatmap",colorscale:[[0,"#00AA00"],[0.5,"#0075C0"],[1,"#C00000"]],\n'
html += '   zmin:0,zmax:2,\n'
html += '   showscale:true,colorbar:{title:"类型",tickvals:[0,1,2],ticktext:["User1反射","User2反射","干扰"],len:0.8},\n'
html += '   hovertemplate:"单元: %{x}<br>类型: %{z:.0f}<extra></extra>"}\n'
html += '],Object.assign({},layout_base,{\n'
html += '  height:Math.max(400,allocEps.length*50+120),\n'
html += '  xaxis:{title:{text:"RIS单元编号",font:{size:12}},tickfont:{size:10},gridcolor:"#eee",dtick:8},\n'
html += '  yaxis:{title:{text:"Episode",font:{size:12}},tickfont:{size:10},gridcolor:"#eee",autorange:"reversed"},\n'
html += '  margin:{t:40,b:50,l:80,r:100}\n'
html += '}));\n'

html += '</script>\n</body>\n</html>'

out_path = f'{mat_dir}/training_report_see.html'
with open(out_path, 'w', encoding='utf-8-sig') as f:
    f.write(html)
print(f'Report saved: {out_path}')
print(f'Stats: episodes={len(episodes)}, final_avg={final_100_avg:.2f}, best={best_score:.2f} (ep {best_ep})')
