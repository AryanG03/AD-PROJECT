/* app.js — SPA router, API health check, toast system */
'use strict';

const App = (() => {

  // ── SPA Router ────────────────────────────────────────────────
  const pages = {
    explorer: { module: ExplorerPage, container: document.getElementById('page-explorer') },
    metrics:  { module: MetricsPage,  container: document.getElementById('page-metrics') },
    ablation: { module: AblationPage, container: document.getElementById('page-ablation') },
    about:    { module: AboutPage,    container: document.getElementById('page-about') },
  };

  const rendered = new Set();
  let currentPage = 'explorer';

  function navigateTo(name) {
    if (!pages[name]) return;

    // Hide all pages
    Object.entries(pages).forEach(([key, p]) => {
      p.container.classList.toggle('active', key === name);
    });

    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.page === name);
    });

    // Render page if not yet rendered (lazy)
    if (!rendered.has(name)) {
      pages[name].module.render(pages[name].container);
      rendered.add(name);
    }

    currentPage = name;
    window.location.hash = name;
  }

  // ── Toast system ──────────────────────────────────────────────
  function toast(message, type = 'info') {
    const tc   = document.getElementById('toast-container');
    const el   = document.createElement('div');
    const icon = type === 'err' ? '⚠️' : type === 'ok' ? '✅' : 'ℹ️';
    el.className = `toast ${type}`;
    el.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    tc.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.3s'; setTimeout(() => el.remove(), 300); }, 3500);
  }

  // ── Health check ──────────────────────────────────────────────
  async function checkHealth() {
    const dot  = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const d = await res.json();
        dot.className  = 'status-dot ok';
        text.textContent = `API ready · ${d.device}`;
      } else {
        throw new Error(res.statusText);
      }
    } catch(e) {
      dot.className  = 'status-dot err';
      text.textContent = 'API offline';
      toast('Cannot reach backend API', 'err');
    }
  }

  // ── Init ──────────────────────────────────────────────────────
  function init() {
    // Wire nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.addEventListener('click', () => navigateTo(btn.dataset.page));
    });

    // Handle hash routing
    const hash = window.location.hash.slice(1);
    navigateTo(hash && pages[hash] ? hash : 'explorer');

    // Health check
    checkHealth();
    setInterval(checkHealth, 30000);
  }

  document.addEventListener('DOMContentLoaded', init);

  return { toast, navigateTo };

})();
