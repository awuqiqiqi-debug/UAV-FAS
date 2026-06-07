"""Generate interactive HTML viewer for training results"""
import scipy.io, os, numpy as np, json

path = 'data/storage/scratch/td3_mfris_fas_v1_2'
ris_pos = [12.0, 35.0]
uav_start = [-10.0, 10.0]

all_rewards = []
all_sec = []
all_att = []
all_dists = []
all_traj = {}

sample_eps = [0, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

for ep in range(501):
    f = os.path.join(path, f'simulation_result_ep_{ep}.mat')
    if not os.path.exists(f): continue
    data = scipy.io.loadmat(f)
    result = data[f'result_{ep}']

    uav_state = result['UAV_state'][0,0]
    reward = result['reward'][0,0]
    sec_cap = result['secure_capacity'][0,0]
    att_cap = result['attaker_capacity'][0,0]

    end_pos = uav_state[-1, :2].tolist()
    end_dist = float(np.sqrt((end_pos[0]-ris_pos[0])**2 + (end_pos[1]-ris_pos[1])**2))

    all_rewards.append(float(np.mean(reward)))
    all_sec.append(float(np.mean(sec_cap)))
    all_att.append(float(np.mean(att_cap)))
    all_dists.append(end_dist)

    if ep in sample_eps:
        traj = [[float(uav_state[i,0]), float(uav_state[i,1])] for i in range(len(uav_state))]
        all_traj[str(ep)] = traj

def smooth(data, w=15):
    return [float(x) for x in np.convolve(data, np.ones(w)/w, mode='valid')]

data_json = json.dumps({
    'rewards': [round(x,4) for x in all_rewards],
    'rewards_smooth': [round(x,4) for x in smooth(all_rewards)],
    'sec_smooth': [round(x,4) for x in smooth(all_sec)],
    'att_smooth': [round(x,4) for x in smooth(all_att)],
    'dist_smooth': [round(x,4) for x in smooth(all_dists)],
    'traj': all_traj,
    'ris_pos': ris_pos,
    'uav_start': uav_start,
})

best_ep = int(np.argmin(all_dists))

html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Twin-TD3 Multi-functional RIS + FAS Training Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Segoe UI', sans-serif; background: #0f1923; color: #e0e0e0; }}
.header {{ background: linear-gradient(135deg, #1a2a3a, #0d2137); padding: 30px 40px; border-bottom: 2px solid #00d4ff; }}
.header h1 {{ font-size: 24px; color: #00d4ff; }}
.header p {{ color: #8899aa; margin-top: 5px; font-size: 14px; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
.stats-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
.stat-card {{ background: #1a2a3a; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #2a3a4a; }}
.stat-card .value {{ font-size: 32px; font-weight: bold; color: #00d4ff; }}
.stat-card .label {{ font-size: 13px; color: #8899aa; margin-top: 5px; }}
.stat-card .sub {{ font-size: 12px; color: #5a7a5a; margin-top: 3px; }}
.charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
.chart-box {{ background: #1a2a3a; border-radius: 10px; padding: 20px; border: 1px solid #2a3a4a; }}
.chart-box h3 {{ color: #00d4ff; margin-bottom: 10px; font-size: 16px; }}
.chart-box canvas {{ max-height: 300px; }}
.traj-section {{ margin: 30px 0; }}
.traj-section h2 {{ color: #00d4ff; margin-bottom: 15px; font-size: 18px; }}
.traj-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
.traj-card {{ background: #1a2a3a; border-radius: 10px; padding: 15px; border: 1px solid #2a3a4a; text-align: center; }}
.traj-card h4 {{ color: #ccc; font-size: 13px; margin-bottom: 8px; }}
.traj-card canvas {{ width: 100%; }}
.info-box {{ background: #1a2a3a; border-radius: 10px; padding: 20px; border: 1px solid #2a3a4a; margin: 20px 0; }}
.info-box h3 {{ color: #00d4ff; margin-bottom: 10px; }}
.info-box ul {{ padding-left: 20px; line-height: 1.8; }}
.info-box li {{ color: #aabbcc; }}
.footer {{ text-align: center; padding: 20px; color: #556; font-size: 12px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Twin-TD3 Multi-functional RIS + FAS Training Results</h1>
  <p>500 Episodes | UAV-BS-FAS + Multi-functional RIS (Reflect + Jam) | SSR Reward</p>
</div>
<div class="container">
  <div class="stats-row">
    <div class="stat-card">
      <div class="value">{all_rewards[-1]:.3f}</div>
      <div class="label">Final Reward</div>
      <div class="sub">Initial: {all_rewards[0]:.3f} | Peak: {max(all_rewards):.3f}</div>
    </div>
    <div class="stat-card">
      <div class="value" style="color:#00ff88">{all_sec[-1]:.3f}</div>
      <div class="label">Secure Capacity</div>
      <div class="sub">Initial: {all_sec[0]:.3f} | Peak: {max(all_sec):.3f}</div>
    </div>
    <div class="stat-card">
      <div class="value" style="color:#ff6644">{all_att[-1]:.3f}</div>
      <div class="label">Attacker Capacity</div>
      <div class="sub">Avg: {np.mean(all_att):.3f} | Jamming target</div>
    </div>
    <div class="stat-card">
      <div class="value" style="color:#ffaa00">{all_dists[-1]:.1f}m</div>
      <div class="label">Distance to RIS</div>
      <div class="sub">Best: {min(all_dists):.1f}m (ep {best_ep})</div>
    </div>
  </div>

  <div class="charts-row">
    <div class="chart-box">
      <h3>Reward Curve (Smoothed)</h3>
      <canvas id="rewardChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Capacity: Secure vs Attacker</h3>
      <canvas id="capacityChart"></canvas>
    </div>
  </div>
  <div class="charts-row">
    <div class="chart-box">
      <h3>RIS Distance (End of Episode)</h3>
      <canvas id="distChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Reward Distribution (Last 100 Episodes)</h3>
      <canvas id="histChart"></canvas>
    </div>
  </div>

  <div class="traj-section">
    <h2>UAV Trajectory Evolution</h2>
    <div class="traj-grid">
      <div class="traj-card"><h4>ep 0 (Start)</h4><canvas id="traj0"></canvas></div>
      <div class="traj-card"><h4>ep 50</h4><canvas id="traj50"></canvas></div>
      <div class="traj-card"><h4>ep 200 (Mid)</h4><canvas id="traj200"></canvas></div>
      <div class="traj-card"><h4>ep 500 (Final)</h4><canvas id="traj500"></canvas></div>
    </div>
  </div>

  <div class="info-box">
    <h3>System Model</h3>
    <ul>
      <li><b>UAV-BS-FAS:</b> UAV carries Base Station (4 ULA antennas) + Fluid Antenna System (8 discrete ports UPA)</li>
      <li><b>Multi-functional RIS:</b> 4 elements, dual-mode - signal reflection amplification (theta_R) + artificial noise jamming (theta_J)</li>
      <li><b>Users:</b> 2 legitimate users | <b>Eavesdropper:</b> 1 attacker</li>
      <li><b>Channel:</b> mmWave 28GHz UMi model with direct + RIS-reflected paths</li>
      <li><b>RL:</b> Twin-TD3 dual-agent (Agent1: G+F+RIS=56 actions | Agent2: UAV trajectory=2 actions)</li>
      <li><b>Reward:</b> SSR + RIS proximity bonus (0.5) + Jamming bonus (0.2) - Eavesdropper penalty</li>
    </ul>
  </div>
</div>
<div class="footer">Twin-TD3 Multi-functional RIS + Discrete-port FAS | Secure UAV Communication</div>

<script>
const D = {data_json};
const labels = Array.from({{length: D.rewards_smooth.length}}, (_,i) => i+7);

Chart.defaults.color = '#8899aa';
Chart.defaults.borderColor = '#2a3a4a';

new Chart(document.getElementById('rewardChart'), {{
  type: 'line',
  data: {{ labels, datasets: [{{ label:'Reward', data: D.rewards_smooth, borderColor:'#00d4ff', borderWidth:2, pointRadius:0, fill:{{ target:'origin', above:'rgba(0,212,255,0.08)' }} }}] }},
  options: {{ responsive:true, plugins:{{ legend:{{display:false}} }}, scales:{{ x:{{display:false}}, y:{{grid:{{color:'#1a2a3a'}}}} }} }}
}});

new Chart(document.getElementById('capacityChart'), {{
  type: 'line',
  data: {{ labels, datasets: [
    {{ label:'Secure Capacity', data: D.sec_smooth, borderColor:'#00ff88', borderWidth:2, pointRadius:0 }},
    {{ label:'Attacker Capacity', data: D.att_smooth, borderColor:'#ff6644', borderWidth:2, pointRadius:0 }}
  ] }},
  options: {{ responsive:true, scales:{{ x:{{display:false}}, y:{{grid:{{color:'#1a2a3a'}}}} }} }}
}});

new Chart(document.getElementById('distChart'), {{
  type: 'line',
  data: {{ labels, datasets: [{{ label:'Distance to RIS (m)', data: D.dist_smooth, borderColor:'#ffaa00', borderWidth:2, pointRadius:0, fill:{{ target:'origin', above:'rgba(255,170,0,0.08)' }} }}] }},
  options: {{ responsive:true, plugins:{{ legend:{{display:false}} }}, scales:{{ x:{{title:{{display:true,text:'Episode'}}}}, y:{{title:{{display:true,text:'Distance (m)'}}, grid:{{color:'#1a2a3a'}}}} }} }}
}});

const last100 = D.rewards.slice(-100);
const bins = Array.from({{length:15}}, (_,i) => (i*0.05).toFixed(2));
const histData = bins.map(b => last100.filter(v => v >= parseFloat(b) && v < parseFloat(b)+0.05).length);
new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{ labels: bins, datasets: [{{ label:'Count', data: histData, backgroundColor:'rgba(0,212,255,0.5)', borderColor:'#00d4ff', borderWidth:1 }}] }},
  options: {{ responsive:true, plugins:{{ legend:{{display:false}} }}, scales:{{ x:{{title:{{display:true,text:'Reward Range'}}}}, y:{{grid:{{color:'#1a2a3a'}}}} }} }}
}});

function drawTraj(canvasId, epKey) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.parentElement.clientWidth - 30;
  const H = canvas.height = 200;
  const pad = 25;
  const xMin=-30, xMax=30, yMin=-5, yMax=55;
  const sx = (W-2*pad)/(xMax-xMin);
  const sy = (H-2*pad)/(yMax-yMin);
  function tc(x,y) {{ return [pad+(x-xMin)*sx, H-pad-(y-yMin)*sy]; }}

  ctx.strokeStyle = '#1a2a3a';
  ctx.lineWidth = 0.5;
  for(let x=-20;x<=20;x+=10) {{ const [cx]=tc(x,0); ctx.beginPath(); ctx.moveTo(cx,pad); ctx.lineTo(cx,H-pad); ctx.stroke(); }}
  for(let y=0;y<=50;y+=10) {{ const [,cy]=tc(0,y); ctx.beginPath(); ctx.moveTo(pad,cy); ctx.lineTo(W-pad,cy); ctx.stroke(); }}

  const [rx,ry] = tc(D.ris_pos[0], D.ris_pos[1]);
  ctx.fillStyle = '#ffaa00';
  ctx.beginPath(); ctx.moveTo(rx,ry-8); ctx.lineTo(rx-7,ry+5); ctx.lineTo(rx+7,ry+5); ctx.closePath(); ctx.fill();
  ctx.font = '10px sans-serif';
  ctx.fillText('RIS', rx+10, ry+4);

  const [sx0,sy0] = tc(D.uav_start[0], D.uav_start[1]);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(sx0-3, sy0-3, 6, 6);

  const traj = D.traj[epKey];
  if(!traj) return;
  ctx.strokeStyle = 'rgba(0,255,136,0.7)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  traj.forEach((p,i) => {{ const [cx,cy] = tc(p[0],p[1]); i===0 ? ctx.moveTo(cx,cy) : ctx.lineTo(cx,cy); }});
  ctx.stroke();

  const [ex,ey] = tc(traj[traj.length-1][0], traj[traj.length-1][1]);
  ctx.fillStyle = '#00ff88';
  ctx.beginPath(); ctx.arc(ex,ey,4,0,Math.PI*2); ctx.fill();
}}

['0','50','200','500'].forEach(ep => {{ drawTraj('traj'+ep, ep); }});
</script>
</body>
</html>'''

with open('results_viewer.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Saved: results_viewer.html')
