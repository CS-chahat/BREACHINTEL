// ===== routes/chatRoutes.js =====
// POST /api/chat  — ARIA AI breach advisor

const express        = require('express');
const router         = express.Router();
const { chatWithAria } = require('../services/groq');

router.post('/chat', async (req, res) => {
  const { message, context, history } = req.body;

  // ── Input validation ─────────────────────────────────────────────────────
  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid message.' });
  }

  const clean = message.trim().replace(/[\x00-\x08\x0b\x0e-\x1f]/g, '');
  if (clean.length === 0) {
    return res.status(400).json({ error: 'Message cannot be empty.' });
  }
  if (clean.length > 1000) {
    return res.status(400).json({ error: 'Message too long (max 1000 characters).' });
  }

  // ── Context validation ───────────────────────────────────────────────────
  if (!context || typeof context !== 'object') {
    return res.status(400).json({ error: 'Missing breach context.' });
  }

  // Sanitize context — only allow expected numeric/string fields
  const safeContext = {
    score:          Number(context.score)          || 0,
    riskLevel:      String(context.riskLevel       || 'UNKNOWN').slice(0, 30),
    breachCount:    Number(context.breachCount)     || 0,
    passwordLeaks:  Number(context.passwordLeaks)  || 0,
    avgSeverity:    Number(context.avgSeverity)     || 0,
    criticalCount:  Number(context.criticalCount)  || 0,
    recentBreaches: Number(context.recentBreaches) || 0,
    breachNames:    Array.isArray(context.breachNames)
      ? context.breachNames.slice(0, 15).map(n => String(n).slice(0, 50))
      : [],
  };

  // Sanitize history — array of {role, text} objects
  const safeHistory = Array.isArray(history)
    ? history.slice(-10).map(h => ({
        role: h.role === 'model' ? 'model' : 'user',
        text: String(h.text || '').slice(0, 500),
      }))
    : [];

  try {
    const reply = await chatWithAria(clean, safeContext, safeHistory);
    return res.json({ reply });
  } catch (err) {
    console.error('[ChatRoute] Error:', err.message);
    return res.status(500).json({ error: err.message });
  }
});

module.exports = router;