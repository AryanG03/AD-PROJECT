/* metrics.js — Metrics Comparison Page */
'use strict';

const MetricsPage = (() => {
  let barChart = null;

  function render(container) {
    container.innerHTML = `
    <div class="hero">
      <h1 class="hero-title">Evaluation Metrics</h1>
      <p class="hero-sub">Full test-set comparison of Neuro-CX vs the GRU Baseline across all ranking metrics.</p>
    </div>
    <div class="container">
      <div id="metrics-summary" class="stat-row"></div>
      <div class="grid-2" style="align-items:start; margin-top:1rem;">
        <div class="card col-span-2">
          <div class="card-title">📊 Score Comparison (Test Set · n=150)</div>
          <div class="chart-wrap" style="max-height:360px;">
            <canvas id="metrics-chart" height="180"></canvas>
          </div>
        </div>
      </div>
      <div class="card" style="margin-top:1.25rem;">
        <div class="card-title">📋 Full Results Table</div>
        <div id="metrics-table"></div>
      </div>
      <div class="grid-4" style="margin-top:1.25rem;" id="metric-chips"></div>
    </div>`;

    loadMetrics();
  }

  async function loadMetrics() {
    try {
      const data = await fetch('/api/metrics').then(r => r.json());
      renderSummary(data);
      renderChart(data.rows);
      renderTable(data.rows);
      renderChips(data.rows);
    } catch(e) {
      document.getElementById('metrics-table').innerHTML =
        `<div class="empty-state"><div class="icon">⚠️</div>Failed to load metrics: ${e.message}</div>`;
    }
  }

  function renderSummary(data) {
    document.getElementById('metrics-summary').innerHTML = `
      <div class="stat-pill">Neuro-CX wins <strong>${data.neurocx_wins} / ${data.total_metrics}</strong> metrics</div>
      <div class="stat-pill">Test set: <strong>150</strong> customers</div>
      <div class="stat-pill">Vocabulary: <strong>200</strong> items</div>`;
  }

  function renderChart(rows) {
    if (barChart) barChart.destroy();
    const ctx = document.getElementById('metrics-chart').getContext('2d');
    barChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: rows.map(r => r.metric),
        datasets: [
          {
            label: 'GRU Baseline',
            data: rows.map(r => r.baseline),
            backgroundColor: 'rgba(76,114,176,0.75)',
            borderColor:     'rgba(76,114,176,1)',
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: 'Neuro-CX',
            data: rows.map(r => r.neurocx),
            backgroundColor: 'rgba(102,126,234,0.85)',
            borderColor:     'rgba(102,126,234,1)',
            borderWidth: 1,
            borderRadius: 4,
          },
        ]
      },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#a0a0c0', font: { size: 12 } } },
          tooltip: {
            backgroundColor: '#121228',
            borderColor: 'rgba(255,255,255,0.08)',
            borderWidth: 1,
            titleColor: '#e0e0f0',
            bodyColor: '#a0a0c0',
            callbacks: {
              afterBody: (items) => {
                const b = items[0]?.raw;
                const n = items[1]?.raw;
                if (b == null || n == null) return '';
                const d = ((n - b) / b * 100).toFixed(1);
                return [`Δ = ${n >= b ? '+' : ''}${(n - b).toFixed(4)} (${d}%)`];
              }
            }
          }
        },
        scales: {
          x: { ticks: { color: '#606080', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { ticks: { color: '#606080', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });
  }

  function renderTable(rows) {
    const tableRows = rows.map(r => {
      const win = r.winner === 'neurocx';
      const d   = r.delta;
      const dStr = (d >= 0 ? '+' : '') + d.toFixed(4);
      const dClass = d > 0 ? 'delta-pos' : d < 0 ? 'delta-neg' : 'delta-neu';
      return `<tr>
        <td style="font-weight:600;color:var(--text-0);">${r.metric}</td>
        <td style="font-family:var(--font-mono);">${r.baseline.toFixed(4)}</td>
        <td style="font-family:var(--font-mono); color:${win ? 'var(--green)':'var(--text-1)'}; font-weight:${win?700:400};">${r.neurocx.toFixed(4)}</td>
        <td class="${dClass}" style="font-family:var(--font-mono);">${dStr}</td>
        <td>${win ? '🏆 Neuro-CX' : '📊 Baseline'}</td>
      </tr>`;
    }).join('');

    document.getElementById('metrics-table').innerHTML = `
      <table class="data-table">
        <thead><tr><th>Metric</th><th>Baseline</th><th>Neuro-CX</th><th>Delta</th><th>Winner</th></tr></thead>
        <tbody>${tableRows}</tbody>
      </table>`;
  }

  function renderChips(rows) {
    const html = rows.map(r => {
      const win = r.winner === 'neurocx';
      const d   = r.delta;
      return `<div class="metric-chip ${win ? 'win' : ''}">
        <div class="metric-chip-label">${r.metric}</div>
        <div class="metric-chip-value">${r.neurocx.toFixed(4)}</div>
        <div class="metric-chip-sub ${d >= 0 ? 'delta-pos' : 'delta-neg'}">${win ? '▲' : '▼'} ${Math.abs(d).toFixed(4)} vs baseline</div>
      </div>`;
    }).join('');
    document.getElementById('metric-chips').innerHTML = html;
  }

  return { render };
})();
