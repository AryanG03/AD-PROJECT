/* explorer.js — Customer Explorer Page */
'use strict';

const ExplorerPage = (() => {
  const API = '';  // same origin
  let customers = [];
  let currentCustomer = null;
  let currentEvents   = [];
  let currentStep     = 0;
  let currentModel    = 'neuro_cx';
  let hiddenHistory   = [];
  let actionHistory   = [];
  let heatmapChart    = null;

  const ACTION_COLORS = {
    purchase:    '#74c69d', view: '#74b9ff',
    add_to_cart: '#c77dff', review: '#fca46d', bounce: '#f08080',
  };

  // ── Render page skeleton ──────────────────────────────────────
  function render(container) {
    container.innerHTML = `
    <div class="hero">
      <h1 class="hero-title">Customer Explorer</h1>
      <p class="hero-sub">Step through a customer's interaction sequence and watch recommendations adapt in real time.</p>
    </div>
    <div class="container">

      <!-- Top controls row -->
      <div style="display:flex; gap:1rem; align-items:flex-end; flex-wrap:wrap; margin-bottom:1.25rem;">
        <div style="flex:1; min-width:220px;">
          <label for="cust-search">Customer ID</label>
          <div class="select-wrapper">
            <select id="cust-select"></select>
          </div>
        </div>
        <div>
          <label>Model</label>
          <div class="model-toggle">
            <button class="model-toggle-btn active" data-model="neuro_cx" id="toggle-ncx">🧠 Neuro-CX</button>
            <button class="model-toggle-btn" data-model="baseline" id="toggle-base">📊 Baseline</button>
          </div>
        </div>
        <div>
          <label>Top-K</label>
          <div class="select-wrapper" style="width:90px;">
            <select id="topk-select">
              <option value="5">5</option>
              <option value="10" selected>10</option>
              <option value="15">15</option>
              <option value="20">20</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Stat pills -->
      <div class="stat-row" id="stat-row">
        <div class="stat-pill"><strong id="stat-events">—</strong> events</div>
        <div class="stat-pill">Step <strong id="stat-step">0</strong></div>
        <div class="stat-pill">Model: <strong id="stat-model">Neuro-CX</strong></div>
      </div>

      <!-- Step bar -->
      <div class="step-bar" style="margin-bottom:1.25rem;">
        <button class="btn btn-ghost btn-sm" id="btn-prev" disabled>◀ Prev</button>
        <div class="step-info">
          <div class="step-label" id="step-label">Select a customer to begin</div>
          <div class="step-event" id="step-event">—</div>
        </div>
        <button class="btn btn-ghost btn-sm" id="btn-reset">↺</button>
        <button class="btn btn-primary btn-sm" id="btn-next" disabled>Next ▶</button>
      </div>
      <div class="progress-track" style="margin-bottom:1.5rem;">
        <div class="progress-fill" id="progress-fill" style="width:0%"></div>
      </div>

      <!-- Main grid -->
      <div class="grid-2" style="align-items:start;">

        <!-- Left: Recommendations -->
        <div>
          <div class="card" style="min-height:400px;">
            <div class="card-title">🎯 Recommendations</div>
            <div id="rec-list"><div class="empty-state"><div class="icon">🎯</div>Run inference to see recommendations</div></div>
          </div>

          <!-- Signal widget -->
          <div class="signal-widget" style="margin-top:1rem;" id="signal-widget">
            <div class="signal-box">
              <div class="signal-box-label">Signal Weight</div>
              <div class="signal-box-value" id="sig-weight">—</div>
              <div class="signal-box-sub" id="sig-weight-label">—</div>
            </div>
            <div class="signal-box" id="decay-box" style="display:none;">
              <div class="signal-box-label">Decay Factor</div>
              <div class="signal-box-value" id="sig-decay">—</div>
              <div class="signal-box-sub" id="sig-decay-label">—</div>
            </div>
            <div class="signal-box">
              <div class="signal-box-label">Hidden Norm</div>
              <div class="signal-box-value" id="sig-norm">—</div>
              <div class="signal-box-sub">||h||</div>
            </div>
          </div>
        </div>

        <!-- Right: Hidden State Heatmap -->
        <div class="card" style="min-height:400px;">
          <div class="card-title">🧬 Hidden State Evolution</div>
          <div id="heatmap-placeholder" class="empty-state"><div class="icon">🧬</div>Step through events to see state evolve</div>
          <div class="chart-wrap" id="heatmap-wrap" style="display:none;">
            <canvas id="heatmap-chart" height="280"></canvas>
          </div>
          <div style="margin-top:1rem;" id="norm-wrap" style="display:none;">
            <div class="card-title" style="margin-top:0.5rem;">Mean Activation</div>
            <div class="chart-wrap">
              <canvas id="norm-chart" height="100"></canvas>
            </div>
          </div>
        </div>

      </div>

      <!-- Event history table -->
      <div class="card" style="margin-top:1.25rem;">
        <div class="card-title">📋 Interaction History</div>
        <div id="history-table-wrap"><div class="empty-state" style="padding:1.5rem;">Select a customer to see their history</div></div>
      </div>

    </div>`;

    bindEvents(container);
    loadCustomers();
  }

  // ── Bind events ───────────────────────────────────────────────
  function bindEvents(container) {
    container.querySelector('#cust-select').addEventListener('change', e => {
      loadCustomer(e.target.value);
    });

    container.querySelectorAll('.model-toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        container.querySelectorAll('.model-toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentModel = btn.dataset.model;
        document.getElementById('stat-model').textContent = currentModel === 'neuro_cx' ? 'Neuro-CX' : 'Baseline';
        resetStepper();
        if (currentCustomer) runInference();
      });
    });

    document.getElementById('btn-prev').addEventListener('click', () => {
      if (currentStep > 0) { currentStep--; runInference(); }
    });
    document.getElementById('btn-next').addEventListener('click', () => {
      if (currentStep < currentEvents.length - 1) { currentStep++; runInference(); }
    });
    document.getElementById('btn-reset').addEventListener('click', resetStepper);
  }

  // ── Load customer list ────────────────────────────────────────
  async function loadCustomers() {
    try {
      const data = await apiFetch('/api/customers?page_size=200');
      customers = data.customers || [];
      const sel = document.getElementById('cust-select');
      sel.innerHTML = customers.map(c => `<option value="${c}">${c}</option>`).join('');
      if (customers.length) loadCustomer(customers[0]);
    } catch(e) { App.toast('Failed to load customers: ' + e.message, 'err'); }
  }

  // ── Load a customer's events ──────────────────────────────────
  async function loadCustomer(cid) {
    currentCustomer = cid;
    resetStepper();
    try {
      const data = await apiFetch(`/api/customers/${encodeURIComponent(cid)}`);
      currentEvents = data.events || [];
      document.getElementById('stat-events').textContent = currentEvents.length;
      renderHistoryTable();
      runInference();
    } catch(e) { App.toast('Failed to load customer: ' + e.message, 'err'); }
  }

  // ── Reset stepper state ───────────────────────────────────────
  function resetStepper() {
    currentStep   = 0;
    hiddenHistory = [];
    actionHistory = [];
    if (heatmapChart) { heatmapChart.destroy(); heatmapChart = null; }
    document.getElementById('heatmap-wrap').style.display = 'none';
    document.getElementById('heatmap-placeholder').style.display = 'flex';
    document.getElementById('norm-wrap') && (document.getElementById('norm-wrap').style.display = 'none');
    document.getElementById('rec-list').innerHTML = '<div class="empty-state"><div class="icon">🎯</div>Run inference to see recommendations</div>';
    updateStepUI();
  }

  // ── Run inference via API ─────────────────────────────────────
  async function runInference() {
    if (!currentCustomer || !currentEvents.length) return;
    const topK = parseInt(document.getElementById('topk-select').value);

    updateStepUI();
    setNavBtns(true);

    try {
      const data = await apiFetch('/api/inference', 'POST', {
        customer_id: currentCustomer,
        model_name:  currentModel,
        step:        currentStep,
        top_k:       topK,
      });

      renderRecommendations(data.recommendations);
      renderSignalWidget(data);
      updateHiddenHistory(data.hidden_sample, data.current_action);
      renderHeatmap();
    } catch(e) {
      App.toast('Inference error: ' + e.message, 'err');
    } finally {
      setNavBtns(false);
    }
  }

  // ── Update step UI elements ───────────────────────────────────
  function updateStepUI() {
    const n    = currentEvents.length;
    const step = currentStep;
    const ev   = currentEvents[step];

    document.getElementById('stat-step').textContent = `${step + 1} / ${n || '—'}`;
    document.getElementById('progress-fill').style.width = n ? `${((step + 1) / n) * 100}%` : '0%';

    if (ev) {
      const badge = `<span class="badge badge-${ev.action}">${ev.action_icon} ${ev.action}</span>`;
      document.getElementById('step-label').innerHTML = `Step ${step + 1} of ${n} &nbsp;·&nbsp; ${badge}`;
      document.getElementById('step-event').textContent = `📦 ${ev.item_name} · ⏱ ${ev.dwell_time}s`;
    }

    document.getElementById('btn-prev').disabled = step === 0;
    document.getElementById('btn-next').disabled = step >= n - 1;
  }

  function setNavBtns(loading) {
    const next = document.getElementById('btn-next');
    const prev = document.getElementById('btn-prev');
    if (loading) { next.innerHTML = '<span class="spinner"></span>'; next.disabled = true; prev.disabled = true; }
    else updateStepUI();
  }

  // ── Render recommendations ────────────────────────────────────
  function renderRecommendations(recs) {
    if (!recs || !recs.length) return;
    const maxScore = recs[0].score;
    const html = recs.map(r => `
      <div class="rec-item" style="animation-delay:${(r.rank-1)*0.03}s">
        <span class="rec-rank ${r.rank <= 3 ? 'top' : ''}">#${r.rank}</span>
        <span class="rec-name">${r.name}</span>
        <div class="rec-bar-wrap"><div class="rec-bar" style="width:${(r.score/maxScore*100).toFixed(1)}%"></div></div>
        <span class="rec-score">${r.pct.toFixed(2)}%</span>
      </div>`).join('');
    document.getElementById('rec-list').innerHTML = html;
  }

  // ── Render signal widget ──────────────────────────────────────
  function renderSignalWidget(data) {
    const w = data.signal_weight;
    document.getElementById('sig-weight').textContent = w.toFixed(1);
    document.getElementById('sig-weight-label').textContent =
      w >= 3 ? 'Strong reinforce' : w >= 1.5 ? 'Moderate' : w >= 1 ? 'Neutral' : 'Weak signal';
    document.getElementById('sig-norm').textContent = data.hidden_norm.toFixed(3);

    const decayBox = document.getElementById('decay-box');
    if (data.decay_factor !== null && data.decay_factor !== undefined) {
      decayBox.style.display = '';
      document.getElementById('sig-decay').textContent = data.decay_factor.toFixed(3);
      document.getElementById('sig-decay-label').textContent =
        data.decay_factor > 0.8 ? 'Low forgetting' : data.decay_factor > 0.5 ? 'Moderate' : 'High forgetting';
    } else {
      decayBox.style.display = 'none';
    }
  }

  // ── Hidden state history ──────────────────────────────────────
  function updateHiddenHistory(h, action) {
    if (!h) return;
    hiddenHistory.push(h.slice(0, 32));   // keep first 32 dims for display
    actionHistory.push(action);
    if (hiddenHistory.length > 20) { hiddenHistory.shift(); actionHistory.shift(); }
  }

  // ── Heatmap (Chart.js) ────────────────────────────────────────
  function renderHeatmap() {
    if (hiddenHistory.length < 1) return;

    document.getElementById('heatmap-placeholder').style.display = 'none';
    document.getElementById('heatmap-wrap').style.display = '';

    const dims    = hiddenHistory[0].length;
    const labels  = actionHistory.map((a, i) => `t${i + 1} ${a[0].toUpperCase()}`);

    // Build flat RGBA image data
    const H = hiddenHistory.length;
    const W = dims;
    const allVals = hiddenHistory.flat();
    const minV    = Math.min(...allVals);
    const maxV    = Math.max(...allVals);
    const range   = maxV - minV || 1;

    function valToColor(v) {
      const n = (v - minV) / range;  // 0..1
      if (n < 0.5) {
        const t = n * 2;
        return `rgba(${Math.round(116*(1-t))},${Math.round(185+70*t)},${Math.round(255*(1-t)+116*t)},0.9)`;
      } else {
        const t = (n - 0.5) * 2;
        return `rgba(${Math.round(102+165*t)},${Math.round(126*(1-t)+198*t)},${Math.round(234*(1-t)+157*t)},0.9)`;
      }
    }

    const datasets = hiddenHistory.map((row, i) => ({
      label: labels[i],
      data: row.map((v, d) => ({ x: d, y: i, v })),
      backgroundColor: row.map(v => valToColor(v)),
    }));

    if (heatmapChart) heatmapChart.destroy();
    const ctx = document.getElementById('heatmap-chart').getContext('2d');
    heatmapChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: Array.from({length: dims}, (_, i) => `h${i}`),
        datasets: [{
          label: 'Hidden state (last step)',
          data: hiddenHistory[hiddenHistory.length - 1],
          backgroundColor: hiddenHistory[hiddenHistory.length - 1].map(v => valToColor(v)),
          borderRadius: 2,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        animation: { duration: 300 },
        plugins: { legend: { display: false }, tooltip: {
          callbacks: { label: ctx => `h[${ctx.dataIndex}] = ${ctx.raw.toFixed(4)}` }
        }},
        scales: {
          x: { display: false },
          y: {
            ticks: { color: '#606080', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.05)' }
          }
        }
      }
    });
  }

  // ── History table ─────────────────────────────────────────────
  function renderHistoryTable() {
    if (!currentEvents.length) return;
    const rows = currentEvents.map((ev, i) => `
      <tr class="${i === currentStep ? 'current-row' : ''}">
        <td style="color:var(--text-3); font-family:var(--font-mono);">${i+1}</td>
        <td><span class="badge badge-${ev.action}">${ev.action_icon} ${ev.action}</span></td>
        <td>${ev.item_name}</td>
        <td style="font-family:var(--font-mono); color:var(--text-2);">${ev.dwell_time}s</td>
        <td style="font-family:var(--font-mono); color:var(--accent-1);">${ev.signal_weight.toFixed(1)}</td>
      </tr>`).join('');

    document.getElementById('history-table-wrap').innerHTML = `
      <div style="overflow-x:auto; max-height:220px; overflow-y:auto;">
        <table class="data-table">
          <thead><tr><th>#</th><th>Action</th><th>Item</th><th>Dwell</th><th>Weight</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }

  // ── API helper ────────────────────────────────────────────────
  async function apiFetch(url, method = 'GET', body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(url, opts);
    if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || res.statusText); }
    return res.json();
  }

  return { render };
})();
