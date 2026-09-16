/* about.js — About Page */
'use strict';

const AboutPage = (() => {
  const TEAM = [
    { name: 'Neha',      role: 'Data Engineering Lead',   color: '#74b9ff',
      tasks: 'Dataset justification · Schema contract · clean_data.py · sequence_builder.py · PyTorch DataLoader · 31 unit tests · data pipeline report section' },
    { name: 'Aryan',     role: 'Model Development Lead',  color: '#c77dff',
      tasks: 'GRU Baseline · Neuro-CX architecture · reinforcement gate · decay gate · trainer loop · AMP fixes · hyperparameter tuning · model report section' },
    { name: 'Diya',      role: 'Evaluation Lead',         color: '#74c69d',
      tasks: 'NDCG/HitRate/Accuracy metrics · evaluation harness · comparison charts · ablation study (4 variants) · evaluation report section' },
    { name: 'Shravani',  role: 'App & Integration Lead',  color: '#fca46d',
      tasks: 'FastAPI backend · REST API design · model registry · HTML/CSS/JS frontend · SPA routing · final report compilation' },
  ];

  function render(container) {
    container.innerHTML = `
    <div class="hero">
      <h1 class="hero-title">About Neuro-CX</h1>
      <p class="hero-sub">A neuroplasticity-inspired GRU recommender with reinforcement and forgetting mechanisms.</p>
    </div>
    <div class="container">

      <!-- Architecture diagram -->
      <div class="section-title">System Architecture</div>
      <div class="arch-diagram">
        <span><span class="hl">Browser</span> <span class="dim">──────────────────▶</span> <span class="hl">FastAPI</span> <span class="dim">(port 8000)</span></span>
        <span><span class="dim">  GET  /</span>                           <span class="dim">├─</span> <span class="hl2">frontend/index.html</span></span>
        <span><span class="dim">  GET  /static/*</span>                   <span class="dim">├─</span> <span class="hl2">CSS · JS · Chart.js</span></span>
        <span><span class="dim">  GET  /api/customers</span>              <span class="dim">├─</span> <span class="hl2">list 1,000 customers</span></span>
        <span><span class="dim">  GET  /api/customers/{id}</span>         <span class="dim">├─</span> <span class="hl2">interaction sequence</span></span>
        <span><span class="dim">  POST /api/inference</span>              <span class="dim">├─</span> <span class="hl2">torch model forward pass</span></span>
        <span><span class="dim">  GET  /api/metrics</span>                <span class="dim">├─</span> <span class="hl2">comparison.json</span></span>
        <span><span class="dim">  GET  /api/ablation</span>               <span class="dim">└─</span> <span class="hl2">ablation_results.csv</span></span>
      </div>

      <!-- Neuro-CX mechanism diagram -->
      <div class="section-title" style="margin-top:2rem;">Neuro-CX Mechanism</div>
      <div class="arch-diagram">
        <span><span class="dim">for t in 1..T:</span></span>
        <span><span class="dim">  x_t   =</span> <span class="hl">[item_emb ‖ action_emb ‖ dwell_proj ‖ weight_proj]</span></span>
        <span><span class="dim">  h_GRU =</span> <span class="hl">GRU(x_t, h_prev)</span>                     <span class="dim">← standard recurrence</span></span>
        <span><span class="dim">  α     =</span> <span class="hl2">sigmoid(W·tanh(W·w_t)) × 3.0</span>         <span class="dim">← reinforcement gate</span></span>
        <span><span class="dim">  h_r   =</span> <span class="hl2">h_GRU × α</span>                             <span class="dim">← amplify strong signals</span></span>
        <span><span class="dim">  decay =</span> <span class="hl">exp(−λ/w_t)</span> <span class="dim">×</span> <span class="hl">σ(W·[h_r ‖ w_t])</span>   <span class="dim">← forgetting gate</span></span>
        <span><span class="dim">  h_new =</span> <span class="hl">decay ⊙ h_r + (1−decay) ⊙ h_prev</span>    <span class="dim">← selective retention</span></span>
        <span> </span>
        <span><span class="dim">logits =</span> <span class="hl">Linear(Dropout(h_T))</span>                   <span class="dim">← next-item prediction</span></span>
      </div>

      <!-- Key results -->
      <div class="section-title" style="margin-top:2rem;">Key Results</div>
      <div class="grid-4">
        ${[
          ['Accuracy@1', '4×', 'vs random baseline'],
          ['NDCG@20', '+21%', 'relative improvement'],
          ['HitRate@20', '+5.3pp', 'over GRU baseline'],
          ['Tests', '31 / 31', 'passing'],
        ].map(([label, val, sub]) => `
          <div class="metric-chip win">
            <div class="metric-chip-label">${label}</div>
            <div class="metric-chip-value">${val}</div>
            <div class="metric-chip-sub">${sub}</div>
          </div>`).join('')}
      </div>

      <!-- Team -->
      <div class="section-title" style="margin-top:2rem;">Team</div>
      <div class="grid-2">
        ${TEAM.map(t => `
          <div class="team-card">
            <div class="team-name">${t.name}</div>
            <div class="team-role" style="color:${t.color};">${t.role}</div>
            <div class="team-tasks">${t.tasks}</div>
          </div>`).join('')}
      </div>

      <!-- Stack -->
      <div class="section-title" style="margin-top:2rem;">Technology Stack</div>
      <div class="grid-4">
        ${[
          ['🐍 Python 3.11', 'Runtime'],
          ['⚡ FastAPI', 'Backend API'],
          ['🔥 PyTorch 2.x', 'Model inference'],
          ['📊 Chart.js 4', 'Frontend charts'],
          ['🎨 Vanilla CSS', 'Design system'],
          ['📦 Parquet', 'Data storage'],
          ['🧪 pytest', '31 unit tests'],
          ['📝 Streamlit', 'Original prototype'],
        ].map(([name, role]) => `
          <div class="card" style="padding:0.85rem 1rem;">
            <div style="font-weight:600; color:var(--text-0); font-size:0.9rem;">${name}</div>
            <div style="font-size:0.75rem; color:var(--text-3); margin-top:0.2rem;">${role}</div>
          </div>`).join('')}
      </div>

      <!-- Footer -->
      <div style="text-align:center; color:var(--text-3); font-size:0.8rem; margin-top:3rem; padding-bottom:2rem;">
        Neuro-CX Research Prototype · MSc Data Science · All sequences are synthetic/derived
      </div>

    </div>`;

    loadModelInfo();
  }

  async function loadModelInfo() {
    try {
      const data = await fetch('/api/models').then(r => r.json());
      Object.entries(data).forEach(([name, info]) => {
        console.log(`Model: ${name}`, info);
      });
    } catch(e) { /* non-critical */ }
  }

  return { render };
})();
