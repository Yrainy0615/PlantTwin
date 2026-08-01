"""Build a self-contained HTML visualization of the structure-in-the-loop process.

Reads the per-step `history` + `delta_snapshots` captured by optimize_per_scene.py
(`--learn-struct`) for one or more phases (v12 delta-only, v13 densify) and emits an
interactive HTML (Plotly via CDN) with:
  - loss curves (total / rgb / stem_proj / pin / struct_l2 / struct_smooth), log-y;
  - |delta| growth (max / p95 / mean) per step;
  - a per-node |delta| evolution heatmap [snapshots x nodes];
  - a phase summary table (N nodes, densified edges, final metrics).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_phase(path: str, name: str) -> dict:
    p = torch.load(path, map_location='cpu', weights_only=False)
    history = p.get('history', [])
    snaps = p.get('delta_snapshots', None)
    phase = {
        'name': name,
        'n_nodes': int(p['delta_rest_pos'].shape[0]) if 'delta_rest_pos' in p else None,
        'history': history,
        'has_snaps': snaps is not None,
    }
    if snaps is not None:
        dn = np.asarray(snaps['delta_norm'])            # [S, N]
        phase['snap_steps'] = list(map(int, snaps['steps']))
        phase['snap_delta'] = (dn * 100.0).tolist()     # cm
    if 'densified_tree' in p:
        phase['densified_n'] = int(p['densified_tree']['nodes'].shape[0])
    if 'densify_split_endpoints' in p:
        phase['n_split'] = int(p['densify_split_endpoints'].shape[0])
    if history:
        phase['final'] = history[-1]
    return phase


PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Structure-in-the-Loop Optimization</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0; background:#0f1115; color:#e6e6e6; }
  header { padding: 20px 28px; background:#171a21; border-bottom:1px solid #262b36; }
  h1 { margin:0; font-size:20px; }
  .sub { color:#8a93a6; font-size:13px; margin-top:6px; line-height:1.5; }
  .wrap { padding: 18px 28px 40px; }
  .phase { margin-bottom: 34px; }
  .phase h2 { font-size:16px; border-left:4px solid #4f8cff; padding-left:10px; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:18px; }
  .card { background:#171a21; border:1px solid #262b36; border-radius:10px; padding:8px; }
  .full { grid-column: 1 / -1; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  td, th { border:1px solid #2a2f3a; padding:5px 9px; text-align:right; }
  th:first-child, td:first-child { text-align:left; color:#9fb3d0; }
  .kpi { display:flex; gap:14px; flex-wrap:wrap; margin:6px 0 4px; }
  .kpi div { background:#1e2430; border:1px solid #2a3340; border-radius:8px; padding:8px 12px; }
  .kpi b { color:#6fd3ff; font-size:18px; }
  .kpi span { color:#8a93a6; font-size:11px; display:block; }
  code { color:#ffd27f; }
</style>
</head>
<body>
<header>
  <h1>Motion-Cue Structure Optimization &mdash; Structure-in-the-Loop</h1>
  <div class="sub">
    用 video motion fit 的残差驱动 branch tree 的结构优化：可学习的 per-node rest 修正
    <code>&delta;</code>（<code>rest_pos_eff = nodes + &delta;</code>），canonical pose 像素级锁死，
    <code>&delta;</code> 只在运动帧显现。<code>&delta;</code> 大的 edge = motion 表达力不足 &rarr; densification 插中点加关节。
    损失：<code>L = L_RGB + 0.005&middot;L_stem_proj + 0.001&middot;L_pin + 0.5&middot;||&delta;||&sup2; + 2.0&middot;||&delta;_c&minus;&delta;_p||&sup2;</code>
  </div>
</header>
<div class="wrap" id="root"></div>
<script>
const PHASES = {phases_json};

function lineFig(elId, hist, keys, title, logy) {
  const steps = hist.map(r => r.step);
  const traces = [];
  for (const [k, color] of keys) {
    if (hist.some(r => k in r)) {
      traces.push({ x: steps, y: hist.map(r => (k in r ? r[k] : null)),
        mode:'lines', name:k, line:{color:color, width:2}, connectgaps:true });
    }
  }
  Plotly.newPlot(elId, traces, {
    title:{text:title, font:{color:'#e6e6e6', size:14}},
    paper_bgcolor:'#171a21', plot_bgcolor:'#171a21',
    font:{color:'#b9c2d0', size:11},
    margin:{l:50,r:12,t:34,b:36},
    xaxis:{title:'step', gridcolor:'#222834'},
    yaxis:{type: logy?'log':'linear', gridcolor:'#222834'},
    legend:{orientation:'h', y:-0.22},
  }, {displayModeBar:false, responsive:true});
}

function heatFig(elId, snapSteps, snapDelta, title) {
  // snapDelta: [S][N] in cm. Transpose so nodes on Y, steps on X.
  const S = snapDelta.length, N = snapDelta[0].length;
  const z = [];
  for (let n=0; n<N; n++) { const row=[]; for (let s=0; s<S; s++) row.push(snapDelta[s][n]); z.push(row); }
  Plotly.newPlot(elId, [{ z:z, x:snapSteps, type:'heatmap', colorscale:'Inferno',
      colorbar:{title:'|δ| cm', titlefont:{color:'#b9c2d0'}, tickfont:{color:'#b9c2d0'}} }], {
    title:{text:title, font:{color:'#e6e6e6', size:14}},
    paper_bgcolor:'#171a21', plot_bgcolor:'#171a21', font:{color:'#b9c2d0', size:11},
    margin:{l:50,r:12,t:34,b:36},
    xaxis:{title:'step', gridcolor:'#222834'},
    yaxis:{title:'node index', gridcolor:'#222834'},
  }, {displayModeBar:false, responsive:true});
}

const root = document.getElementById('root');
PHASES.forEach((ph, pi) => {
  const div = document.createElement('div'); div.className='phase';
  let split = ('n_split' in ph) ? `densify ${ph.n_split} edges &rarr; N=${ph.densified_n}` : 'δ only (no densify)';
  const f = ph.final || {};
  div.innerHTML = `<h2>Phase ${pi+1}: ${ph.name} &nbsp;<span style="color:#8a93a6;font-size:13px">(${split})</span></h2>
    <div class="kpi">
      <div><b>${(f.loss!=null?f.loss.toFixed(4):'-')}</b><span>final loss</span></div>
      <div><b>${(f.rgb!=null?f.rgb.toFixed(4):'-')}</b><span>RGB</span></div>
      <div><b>${(f.stem_proj!=null?f.stem_proj.toFixed(1):'-')}</b><span>stem_proj</span></div>
      <div><b>${(f.pin!=null?f.pin.toFixed(1):'-')}</b><span>pin</span></div>
      <div><b>${(f.delta_max!=null?(f.delta_max*100).toFixed(1)+' cm':'-')}</b><span>max |δ|</span></div>
      <div><b>${ph.n_nodes!=null?ph.n_nodes:'-'}</b><span>N nodes</span></div>
    </div>
    <div class="grid">
      <div class="card" id="loss_${pi}"></div>
      <div class="card" id="delta_${pi}"></div>
      <div class="card full" id="heat_${pi}"></div>
    </div>`;
  root.appendChild(div);

  lineFig(`loss_${pi}`, ph.history,
    [['loss','#4f8cff'],['rgb','#7fe07f'],['stem_proj','#ffb04f'],['pin','#ff6f9c'],
     ['struct_l2','#9b8cff'],['struct_smooth','#5fd3d3'],['geom','#ff3b3b'],['motion','#36d0c0']],
    'Loss terms (log-y) — incl. geom / motion constraints', true);
  lineFig(`delta_${pi}`, ph.history,
    [['delta_max','#ff5050'],['delta_p95','#ff9b50'],['delta_mean','#ffd27f'],['geom_n_out','#9b8cff']],
    '|δ| growth + #nodes out-of-branch (geom_n_out)', false);
  if (ph.has_snaps) {
    heatFig(`heat_${pi}`, ph.snap_steps, ph.snap_delta,
      'Per-node |δ| evolution (bright = structure wants more articulation here)');
  } else {
    document.getElementById(`heat_${pi}`).innerHTML =
      '<div style="padding:20px;color:#8a93a6">no delta snapshots</div>';
  }
});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', action='append', nargs=2, metavar=('NAME', 'PATH'),
                        help='Repeatable: a phase label and its final_params.pt path.')
    parser.add_argument('--out', default='outputs/per_scene_optim/struct_opt_process.html')
    args = parser.parse_args()

    if not args.phase:
        args.phase = [
            ['v12 (δ only)', 'outputs/per_scene_optim/newplant9_v12_struct/final_params.pt'],
            ['v13 (densify)', 'outputs/per_scene_optim/newplant9_v13_dens/final_params.pt'],
        ]

    phases = []
    for name, path in args.phase:
        if not Path(path).exists():
            print(f'[skip] {path} not found')
            continue
        ph = load_phase(path, name)
        if not ph['history']:
            print(f'[warn] {path} has no history (re-run optimize_per_scene.py to capture it)')
        phases.append(ph)

    html = PAGE.replace('{phases_json}', json.dumps(phases))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')
    print(f'wrote {out}  ({len(phases)} phases)')


if __name__ == '__main__':
    main()
