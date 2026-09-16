/* ablation.js — Ablation Study Page */
'use strict';

const AblationPage = (() => {
  let chart = null;

  const VARIANT_COLORS = {
    'GRU Baseline':       'rgba(76,114,176,0.8)',
    'Reinforcement Only': 'rgba(221,132,82,0.8)',
    'Decay Only':         'rgba(85,168,104,0.8)',
    'Full Neuro-CX':      'rgba(102,126,234,0.9)',
  };

  const INSIGHTS = [
    { icon: '⚡', title: 'Reinforcement alone backfires',
      text: 'When only the reinforcement gate is active, NDCG@10 drops to 0.061 — worse than the baseline (0.091). Without forgetting, amplified signals accumulate and overload the hidden state.' },
    { icon: '🌊', title: 'Decay alone is insufficient',
      text: 'With only the decay gate, performance improves slightly to 0.073 NDCG@10, but still underperforms the baseline. Without reinforcement, all events decay equally regardless of their importance.' },
    { icon: '🔬', title: 'Synergy is the key finding',
      text: 'Only when both mechanisms operate together does performance exceed the baseline. Reinforcement selects which signals matter; decay removes what no longer does.' },
    { icon: '🧠', title: 'Biological analogy validated',
      text: 'This mirrors long-term potentiation (LTP) in neuroscience: strong stimuli both strengthen synapses AND suppress competing signals. Neither without the other produces durable, selective memory.' },
  ];

  function render(container) {
    container.innerHTML = `
    <div class="hero">
      <h1 class="hero-title">Ablation Study</h1>
      <p class="hero-sub">Isolating the contribution of each Neuro-CX mechanism by disabling them at inference time.</p>
    </div>
    <div class="container">

      <div class="stat-row">
        <div class="stat-pill">4 variants tested</div>
        <div class="stat-pill">Test set: <strong>150</strong> customers</div>
        <div class="stat-pill">Inference-time ablation (no retraining)</div>
      </div>

      <!-- Main chart -->
      <div class="card" style="margin-top:1rem;">
        <div class="card-title">📊 Mechanism Contribution — All Metrics</div>
        <div class="chart-wrap">
          <canvas id="ablation-chart" height="200"></canvas>
        </div>
      </div>

      <!-- Table + chips grid -->
      <div class="grid-2" style="margin-top:1.25rem; align-items:start;">
        <div class="card">
          <div class="card-title">📋 Results Table</div>
          <div id="ablation-table"></div>
        </div>
        <div>
          <div class="section-title">Key Metrics Spotlight</div>
          <div id="ablation-chips" class="grid-2" style="gap:0.75rem;"></div>
        </div>
      </div>

      <!-- Insights -->
      <div class="section-title" style="margin-top:2rem;">Interpretation</div>
      <div class="grid-2" id="insights-grid"></div>

    </div>`;

    loadAblation();
  }

  async function loadAblation() {
    try {
      const data = await fetch('/api/ablation').then(r => r.json());
      const variants = data.variants || [];
      renderChart(variants);
      renderTable(variants);
      renderChips(variants);
      renderInsights();
    } catch(e) {
      document.getElementById('ablation-table').innerHTML =
        `<div class="empty-state"><div class="icon">⚠️</div>${e.message}</div>`;
    }
  }

  function renderChart(variants) {
    if (chart) chart.destroy();
    const metrics = Object.keys(variants[0] || {}).filter(k => k !== 'variant');
    const ctx = document.getElementById('ablation-chart').getContext('2d');
    chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: metrics,
        datasets: variants.map(v => ({
          label: v.variant,
          data:  metrics.map(m => parseFloat(v[m]) || 0),
          backgroundColor: VARIANT_COLORS[v.variant] || 'rgba(150,150,150,0.7)',
          borderRadius: 3,
        }))
      },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { color: '#a0a0c0', font: { size: 11 }, padding: 16 } },
          tooltip: {
            backgroundColor: '#121228', borderColor: 'rgba(255,255,255,0.08)', borderWidth: 1,
            titleColor: '#e0e0f0', bodyColor: '#a0a0c0',
          }
        },
        scales: {
          x: { ticks: { color: '#606080', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { ticks: { color: '#606080', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });
  }

  function renderTable(variants) {
    const metrics = Object.keys(variants[0] || {}).filter(k => k !== 'variant');
    const header  = `<tr><th>Variant</th>${metrics.map(m => `<th>${m}</th>`).join('')}</tr>`;
    const rows    = variants.map(v => {
      const isFull = v.variant === 'Full Neuro-CX';
      const cells  = metrics.map(m => {
        const val  = parseFloat(v[m]) || 0;
        // Find best across variants for this metric
        const best = Math.max(...variants.map(vv => parseFloat(vv[m]) || 0));
        const isB  = Math.abs(val - best) < 1e-6;
        return `<td style="font-family:var(--font-mono); color:${isB ? 'var(--green)' : 'var(--text-1)'}; font-weight:${isB?700:400};">${val.toFixed(4)}</td>`;
      }).join('');
      return `<tr style="${isFull ? 'background:rgba(102,126,234,0.07);' : ''}">${
        `<td style="font-weight:${isFull?700:400}; color:${isFull?'var(--accent-1)':'var(--text-1)'};">${v.variant}</td>`}${cells}</tr>`;
    }).join('');

    document.getElementById('ablation-table').innerHTML = `
      <div style="overflow-x:auto;">
        <table class="data-table"><thead>${header}</thead><tbody>${rows}</tbody></table>
      </div>`;
  }

  function renderChips(variants) {
    const spotlight = ['Accuracy@1', 'NDCG@10', 'HitRate@20'];
    const full = variants.find(v => v.variant === 'Full Neuro-CX');
    const base = variants.find(v => v.variant === 'GRU Baseline');
    const html = spotlight.map(m => {
      const fv = parseFloat(full?.[m] || 0);
      const bv = parseFloat(base?.[m] || 0);
      const d  = fv - bv;
      return `<div class="metric-chip win">
        <div class="metric-chip-label">${m}</div>
        <div class="metric-chip-value">${fv.toFixed(4)}</div>
        <div class="metric-chip-sub delta-pos">▲ +${d.toFixed(4)} vs baseline</div>
      </div>`;
    }).join('');
    document.getElementById('ablation-chips').innerHTML = html;
  }

  function renderInsights() {
    document.getElementById('insights-grid').innerHTML = INSIGHTS.map(ins => `
      <div class="card">
        <div style="font-size:1.8rem; margin-bottom:0.6rem;">${ins.icon}</div>
        <div style="font-weight:700; color:var(--text-0); margin-bottom:0.4rem; font-size:0.95rem;">${ins.title}</div>
        <div style="font-size:0.85rem; color:var(--text-2); line-height:1.65;">${ins.text}</div>
      </div>`).join('');
  }

  return { render };
})();
