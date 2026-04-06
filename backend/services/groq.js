// ===== services/geminiService.js =====
// ARIA AI — breach-aware security advisor
// Using Groq API (India-friendly, free tier, ultra-fast)
// API key: GROQ_API_KEY in .env

const axios = require('axios');

const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_MODEL = 'llama-3.1-8b-instant'; // Free, fast, reliable

// ── Build system prompt based on user's breach context ───────────────────────
function buildSystemPrompt(context) {
  const {
    score,
    riskLevel,
    breachCount,
    passwordLeaks,
    avgSeverity,
    criticalCount,
    recentBreaches,
    breachNames,
  } = context;

  const breachList = breachNames && breachNames.length > 0
    ? breachNames.slice(0, 10).join(', ')
    : 'unknown sources';

  return `You are ARIA (AI Risk Intelligence Advisor), an expert cybersecurity assistant embedded in the Breach Intel platform.

## USER'S CURRENT BREACH PROFILE
- Risk Score: ${score}/100
- Risk Level: ${riskLevel}
- Total Breaches Found: ${breachCount}
- Password Leaks: ${passwordLeaks}
- Average Severity: ${avgSeverity}/10
- Critical Breaches (severity 9-10): ${criticalCount}
- Recent Breaches (2020+): ${recentBreaches}
- Known Breach Sources: ${breachList}

## YOUR ROLE & RULES
1. You are a helpful, concise cybersecurity expert. Always base your advice on the user's ACTUAL breach profile above.
2. Personalize every response — reference their specific score, breach count, or risk level when relevant.
3. Be direct and actionable. No vague generic advice. If their score is 0 (no exposure), reassure them but still give prevention tips.
4. Tone: Professional but approachable. Use security terminology correctly. Keep responses under 200 words unless the user asks for detail.
5. NEVER reveal, guess, or display the user's email address.
6. NEVER say you cannot help with security topics — this is your specialty.
7. If asked about non-security topics, politely redirect to their breach/security situation.
8. Format responses clearly: use short paragraphs or bullet points when listing steps.

## RISK-LEVEL BEHAVIOUR
${score === 0 ? `
- User has NO EXPOSURE. Congratulate them briefly, then shift to prevention best practices.
- Emphasize monitoring and password hygiene to stay safe.
` : score < 25 ? `
- LOW RISK (score ${score}). User has minimal exposure. Reassure but don't dismiss it.
- Focus on: changing passwords from the breached services, enabling 2FA.
` : score < 50 ? `
- MEDIUM RISK (score ${score}). User has real exposure that needs action.
- Focus on: password changes, 2FA, checking which specific services were breached.
` : score < 75 ? `
- HIGH RISK (score ${score}). User has significant exposure. Be direct about urgency.
- Focus on: immediate password changes, 2FA on ALL accounts, checking for unauthorized access.
` : `
- CRITICAL RISK (score ${score}). User has severe exposure. Treat with urgency.
- Focus on: emergency password rotation, credit monitoring, fraud alerts, full account audit.
`}

## CONVERSATION STYLE
- Start responses naturally (no "Sure!" or "Great question!")
- If score > 0 and user asks what to do: give ranked action list based on their specific breaches
- If user asks about a specific breach (e.g., "what is the Adobe breach"): explain it and relate to their profile
- If user asks "am I safe": answer based on their actual score, not generically`;
}

// ── Main chat function ────────────────────────────────────────────────────────
async function chatWithAria(userMessage, context, conversationHistory = []) {
  const apiKey = process.env.GROQ_API_KEY || '';

  if (!apiKey) {
    throw new Error('GROQ_API_KEY not set in .env file. Get free key at console.groq.com');
  }

  const systemPrompt = buildSystemPrompt(context);

  // Build messages array (OpenAI-compatible format used by Groq)
  const messages = [
    { role: 'system', content: systemPrompt },
  ];

  // Add conversation history (last 10 turns max)
  const recentHistory = conversationHistory.slice(-10);
  for (const turn of recentHistory) {
    messages.push({
      role: turn.role === 'model' ? 'assistant' : 'user',
      content: turn.text,
    });
  }

  // Add current user message
  messages.push({ role: 'user', content: userMessage });

  try {
    const response = await axios.post(
      GROQ_URL,
      {
        model: GROQ_MODEL,
        messages,
        max_tokens: 150,
        temperature: 0.7,
        top_p: 0.85,
      },
      {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        timeout: 20000,
      }
    );

    const text = response.data?.choices?.[0]?.message?.content || '';
    if (!text) throw new Error('Empty response from AI');

    return text.trim();

  } catch (err) {
    if (err.response?.status === 429) {
      throw new Error('AI rate limit reached. Please wait a moment and try again.');
    }
    if (err.response?.status === 401) {
      throw new Error('Invalid GROQ_API_KEY. Check your .env file.');
    }
    if (err.response?.status === 400) {
      throw new Error('Invalid request to AI. Please rephrase your question.');
    }
    throw new Error(`AI service error: ${err.message}`);
  }
}

module.exports = { chatWithAria };