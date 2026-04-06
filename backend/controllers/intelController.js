// ===== controllers/intelController.js =====
const { gatherIntelligence } = require('../services/intelligenceService');
const { spawn }              = require('child_process');
const path                   = require('path');
const config                 = require('../config/config');

const ML_DIR = path.join(__dirname, '..', '..', 'ml');

function runPython(script, inputObj) {
  return new Promise((resolve, reject) => {
    const proc = spawn(config.PYTHON_CMD, [path.join(ML_DIR, script)], {
      stdio: ['pipe', 'pipe', 'pipe']
    });
    let stdout = '', stderr = '';
    proc.stdout.on('data', d => { stdout += d.toString(); });
    proc.stderr.on('data', d => { stderr += d.toString(); });
    proc.on('close', code => {
      if (code !== 0) return reject(new Error(`Python (${script}) exited ${code}: ${stderr.slice(0, 300)}`));
      try { resolve(JSON.parse(stdout.trim())); }
      catch { reject(new Error(`Python invalid JSON: ${stdout.slice(0, 200)}`)); }
    });
    proc.on('error', err => reject(new Error(`Cannot spawn Python: ${err.message}`)));
    proc.stdin.write(JSON.stringify(inputObj));
    proc.stdin.end();
  });
}

async function health(req, res) {
  res.json({
    status: 'ok', version: '3.0.0',
    leakhunterKeySet: Boolean(config.LEAKHUNTER_API_KEY),
    pythonCmd: config.PYTHON_CMD,
    timestamp: new Date().toISOString(),
  });
}

async function getIntel(req, res) {
  const { identifier } = req.body;

  try {
    const intel = await gatherIntelligence(identifier);
    const { features, leakhunter, xposedornot, merged } = intel;

    // ── HARD ZERO GATE: unexposed email always gets score 0 ──────────────────
    const isExposed = leakhunter?.isExposed || merged.breachCount > 0;

    if (!isExposed) {
      return res.json({
        identifier,
        leakhunter,
        xposedornot,
        ml: {
          finalScore: 0,
          riskLevel:  'NO EXPOSURE',
          factors: [
            { icon: '🔑', name: 'PASSWORD LEAKS',  score: 0, barColor: 'var(--accent-red)' },
            { icon: '💀', name: 'BREACH SEVERITY', score: 0, barColor: 'var(--accent-orange)' },
            { icon: '🔁', name: 'EXPOSURE COUNT',  score: 0, barColor: 'var(--accent-yellow)' },
            { icon: '⚡', name: 'RECENT BREACHES', score: 0, barColor: 'var(--accent-cyan)' },
            { icon: '🌐', name: 'PUBLIC EXPOSURE', score: 0, barColor: 'var(--accent-blue)' },
          ],
          shapFactors: [],
          meta: { breachCount: 0, passwordLeaks: 0, avgSeverity: 0, criticalCount: 0, recentBreaches: 0 },
        },
        quota: leakhunter?.quota || null,
        apiCoverage: {
          leakhunter:  leakhunter  ? 'ok' : 'error',
          xposedornot: xposedornot ? 'ok' : 'error',
        },
        timestamp: new Date().toISOString(),
      });
    }

    // ── Run Python pipeline for exposed emails ───────────────────────────────
    let pipelineOut;
    try {
      pipelineOut = await runPython('data_pipeline.py', { features });
    } catch (pyErr) {
      console.warn('[Pipeline] Fallback:', pyErr.message);
      pipelineOut = heuristicPipeline(features);
    }

    let mlOut;
    try {
      mlOut = await runPython('model.py', { normalized_features: pipelineOut });
    } catch (pyErr) {
      console.warn('[Model] Fallback:', pyErr.message);
      mlOut = heuristicModel(pipelineOut, merged.signals, merged.breachCount);
    }

    return res.json({
      identifier,
      leakhunter,
      xposedornot,
      ml: {
        finalScore:  mlOut.score,
        riskLevel:   mlOut.risk_level,
        factors:     mlOut.factors,
        shapFactors: mlOut.shap_factors,
        meta: {
          breachCount:    merged.breachCount,
          passwordLeaks:  merged.passwordLeaks,
          avgSeverity:    parseFloat(features.avg_severity.toFixed(1)),
          criticalCount:  features.critical_count,
          recentBreaches: features.recent_breaches,
        },
      },
      quota: leakhunter?.quota || null,
      apiCoverage: {
        leakhunter:  leakhunter  ? 'ok' : 'error',
        xposedornot: xposedornot ? 'ok' : 'error',
      },
      timestamp: new Date().toISOString(),
    });

  } catch (err) {
    console.error('[IntelController]', err);
    return res.status(500).json({ error: err.message });
  }
}

// ── Heuristic fallbacks (Python unavailable) ─────────────────────────────────
function heuristicPipeline(features) {
  // Caps MUST match data_pipeline.py exactly (60, 40, 20, 20)
  // Old caps (200, 100, 50, 30) were compressing small breach counts → wrong low scores
  const log1p = x => Math.log(1 + x);
  return {
    breach_norm:         Math.min(log1p(features.breach_count)    / log1p(60),  1),
    password_norm:       Math.min(log1p(features.password_leaks)  / log1p(40),  1),
    severity_norm:       Math.min(features.avg_severity / 10, 1),
    critical_norm:       Math.min(log1p(features.critical_count)  / log1p(20),  1),
    recent_norm:         Math.min(log1p(features.recent_breaches) / log1p(20),  1),
    login_anomaly_score: Math.min(Math.max(features.login_anomaly_score, 0), 1),
    public_exposure:     Math.min(Math.max(features.public_exposure,     0), 1),
    social_risk_score:   Math.min(Math.max(features.social_risk_score,   0), 1),
    has_password_breach: features.has_password_breach,
  };
}

function heuristicModel(nf, signals, breachCount) {
  // Absolute zero guard — no breaches = score 0
  if (!breachCount || breachCount === 0) {
    return zeroResult();
  }

  // ── Direct scoring — mirrors model.py direct_score() ────────────────────
  const breach_n = nf.breach_norm   || 0;
  const pwd_n    = nf.password_norm  || 0;
  const sev_n    = nf.severity_norm  || 0;
  const crit_n   = nf.critical_norm  || 0;
  const recent_n = nf.recent_norm    || 0;
  const has_pwd  = nf.has_password_breach || 0;

  const volume     = Math.min(breach_n * 100 * 0.75, 100);
  const password   = Math.min(pwd_n * 100, 100);
  const sev_weight = Math.min(0.50 + 0.50 * (breach_n / 0.27), 1.0);
  const severity   = sev_n * 100 * sev_weight;
  const critical   = Math.min(crit_n  * 100, 100);
  const recent     = Math.min(recent_n * 100, 100);

  let score = (
    volume   * 0.35 +
    password * 0.25 +
    severity * 0.22 +
    critical * 0.12 +
    recent   * 0.06
  );

  if (has_pwd === 1) {
    score = breach_n > 0.27
      ? Math.min(score * 1.22, 100)
      : Math.min(score * 1.10, 100);
  }

  score = Math.max(score, 8.0);
  if (signals?.highSeverityBreach) score = Math.min(score + 6, 100);
  score = Math.min(Math.round(score), 100);

  const risk = score >= 75 ? 'CRITICAL' : score >= 50 ? 'HIGH RISK' : score >= 25 ? 'MEDIUM' : 'LOW RISK';

  return {
    score,
    risk_level: risk,
    factors: [
      { icon: '🔑', name: 'PASSWORD LEAKS',  score: Math.min(Math.round(nf.password_norm   * 10), 10), barColor: 'var(--accent-red)' },
      { icon: '💀', name: 'BREACH SEVERITY', score: Math.min(Math.round(nf.severity_norm   * 10), 10), barColor: 'var(--accent-orange)' },
      { icon: '🔁', name: 'EXPOSURE COUNT',  score: Math.min(Math.round(nf.breach_norm     * 10), 10), barColor: 'var(--accent-yellow)' },
      { icon: '⚡', name: 'RECENT BREACHES', score: Math.min(Math.round(nf.recent_norm     * 10), 10), barColor: 'var(--accent-cyan)' },
      { icon: '🌐', name: 'PUBLIC EXPOSURE', score: Math.min(Math.round(nf.public_exposure * 10), 10), barColor: 'var(--accent-blue)' },
    ],
    shap_factors: [
      { label: 'has_password_breach',   pts: Math.round(nf.has_password_breach * 28 + nf.password_norm * 10), pct: 80 },
      { label: 'breach_severity_score', pts: Math.round(nf.severity_norm * 22 + nf.critical_norm * 8),        pct: 65 },
      { label: 'breach_norm (log1p)',   pts: Math.round(nf.breach_norm * 18),                                 pct: 55 },
      { label: 'login_anomaly_score',   pts: Math.round(nf.login_anomaly_score * 15),                         pct: 40 },
      { label: 'public_exposure',       pts: Math.round(nf.public_exposure * 8),                              pct: 25 },
      { label: 'social_risk_score',     pts: Math.round(nf.social_risk_score * 6),                            pct: 20 },
      { label: 'recent_breach_norm',    pts: Math.round(nf.recent_norm * 10),                                 pct: 30 },
      { label: 'password_norm (log1p)', pts: Math.round(nf.password_norm * 14),                               pct: 45 },
    ].sort((a, b) => b.pts - a.pts),
  };
}

function zeroResult() {
  return {
    score: 0, risk_level: 'NO EXPOSURE',
    factors: [
      { icon: '🔑', name: 'PASSWORD LEAKS',  score: 0, barColor: 'var(--accent-red)' },
      { icon: '💀', name: 'BREACH SEVERITY', score: 0, barColor: 'var(--accent-orange)' },
      { icon: '🔁', name: 'EXPOSURE COUNT',  score: 0, barColor: 'var(--accent-yellow)' },
      { icon: '⚡', name: 'RECENT BREACHES', score: 0, barColor: 'var(--accent-cyan)' },
      { icon: '🌐', name: 'PUBLIC EXPOSURE', score: 0, barColor: 'var(--accent-blue)' },
    ],
    shap_factors: [],
  };
}

module.exports = { getIntel, health };