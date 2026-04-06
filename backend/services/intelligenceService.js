// ===== services/intelligenceService.js =====
const { checkLeakHunter } = require('./leakhunterService');
const { checkXON }         = require('./xonService');

async function gatherIntelligence(email) {
  const [lhResult, xonResult] = await Promise.allSettled([
    checkLeakHunter(email),
    checkXON(email),
  ]);

  const lh  = lhResult.status  === 'fulfilled' ? lhResult.value  : null;
  const xon = xonResult.status === 'fulfilled' ? xonResult.value : null;

  const exposures      = lh?.exposures || [];
  const signals        = lh?.signals   || {};
  const lhBreachCount  = lh?.exposureCount || exposures.length || 0;
  const xonBreachCount = xon?.breachCount || 0;
  const mergedBreachCount = Math.max(lhBreachCount, xonBreachCount);

  // ── If truly not exposed, return zeroed features immediately ────────────────
  const isExposed = lh?.isExposed || mergedBreachCount > 0;

  if (!isExposed) {
    return {
      leakhunter:  lh,
      xposedornot: xon,
      merged: { breachCount: 0, exposures: [], signals: {}, passwordLeaks: 0 },
      features: {
        breach_count:        0,
        password_leaks:      0,
        avg_severity:        0,
        critical_count:      0,
        recent_breaches:     0,
        login_anomaly_score: 0,
        public_exposure:     0,
        social_risk_score:   0,
        has_password_breach: 0,
      },
    };
  }

  const passwordLeaks = exposures.filter(e => e.passwordIncluded).length
    || Math.max(xon?.passwordLeakCount || 0, 0);

  const avgSeverity = exposures.length > 0
    ? exposures.reduce((s, e) => s + e.severity, 0) / exposures.length
    : 0;

  const criticalCount = exposures.filter(e => e.severity >= 9).length;

  const recentBreaches = exposures.filter(e => {
    const y = parseInt((e.date || '0000').substring(0, 4));
    return y >= 2020;
  }).length;

  const features = {
    breach_count:        mergedBreachCount,
    password_leaks:      passwordLeaks,
    avg_severity:        avgSeverity,
    critical_count:      criticalCount,
    recent_breaches:     recentBreaches,
    // Only non-zero when there IS exposure evidence
    login_anomaly_score: signals.recentBreach      ? 0.75 : (mergedBreachCount > 0 ? 0.1 : 0),
    public_exposure:     Math.min(mergedBreachCount / 200, 1),
    social_risk_score:   signals.multipleExposures ? 0.65 : (mergedBreachCount > 0 ? 0.1 : 0),
    has_password_breach: signals.passwordExposed   ? 1    : 0,
  };

  return {
    leakhunter:   lh,
    xposedornot:  xon,
    merged: { breachCount: mergedBreachCount, exposures, signals, passwordLeaks },
    features,
  };
}

module.exports = { gatherIntelligence };
