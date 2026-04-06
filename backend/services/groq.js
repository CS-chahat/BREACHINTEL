// ===== services/groq.js =====
// ARIA AI — breach-aware security advisor
// Using Groq API (India-friendly, free tier, ultra-fast)
// API key: GROQ_API_KEY in .env

const axios = require('axios');

const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_MODEL = 'llama-3.1-8b-instant';

// ── Detect language of user message ──────────────────────────────────────────
function detectLanguage(text) {
  // Devanagari script = definitely hinglish
  if (/[\u0900-\u097F]/.test(text)) return 'hinglish';
  // Common Hinglish words
  const hinglishPattern = /\b(kya|hai|hain|karo|kaise|mera|meri|aap|apna|apne|isko|usko|toh|bhi|aur|nahi|nhi|kyun|kuch|sab|hua|hoga|kaisa|batao|bata|dekho|lagta|chahiye|wala|wali|teri|tera|mere|tumhara|accha|theek|bilkul|sirf|abhi|phir|fir|yeh|ye|wo|woh|lekin|par|pe|se|ko|ka|ki|ke|kab|kaun|kon|kitna|kitne|bahut|bohot|zaroor|karna|karein|karke|liye|mil|mila|dekh|samjha|samjho)\b/i;
  if (hinglishPattern.test(text)) return 'hinglish';
  return 'english';
}

// ── Build system prompt ───────────────────────────────────────────────────────
function buildSystemPrompt(context, lang) {
  const {
    score, riskLevel, breachCount, passwordLeaks,
    avgSeverity, criticalCount, recentBreaches, breachNames,
  } = context;

  const breachList = breachNames && breachNames.length > 0
    ? breachNames.slice(0, 10).join(', ')
    : 'unknown sources';

  const langInstruction = lang === 'hinglish'
    ? `CRITICAL LANGUAGE RULE: User is speaking Hinglish. You MUST respond in Hinglish (Hindi-English mix). Every sentence must have Hindi words mixed with English. Do NOT respond in pure English under any circumstance.`
    : `CRITICAL LANGUAGE RULE: User is speaking English. You MUST respond in pure English only. Do NOT use any Hindi or Hinglish words at all — not even a single Hindi word like "aapko", "karein", "hain" etc.`;

  return `You are ARIA (AI Risk Intelligence Advisor), a cybersecurity assistant in the Breach Intel platform.

## ${langInstruction}

## USER'S BREACH PROFILE
- Risk Score: ${score}/100 (${riskLevel})
- Total Breaches: ${breachCount}
- Password Leaks: ${passwordLeaks}
- Avg Severity: ${avgSeverity}/10
- Critical Breaches: ${criticalCount}
- Recent Breaches (2020+): ${recentBreaches}
- Breach Sources: ${breachList}

## OFF-TOPIC RULE
If user asks anything NOT related to their security or breaches, reply with ONLY this one line (in their detected language):
- English version: "I only handle security. Your score is ${score}/100 — ask me about your risks or how to fix them."
- Hinglish version: "Main sirf security ke liye hoon. Tera score ${score}/100 hai — risks ya fix ke baare mein pooch."

## ANSWER RULES
- Max 70 words per reply. Short and crisp.
- Always mention their actual score (${score}/100) or breach count in the answer.
- Use bullet points only for action steps (max 4 bullets).
- No filler openers like "Sure!", "Of course!", "Great question!".`;
}

// ── Main chat function ────────────────────────────────────────────────────────
async function chatWithAria(userMessage, context, conversationHistory = []) {
  const apiKey = process.env.GROQ_API_KEY || '';

  if (!apiKey) {
    throw new Error('GROQ_API_KEY not set in .env file. Get free key at console.groq.com');
  }

  // Detect language from actual user message
  const lang = detectLanguage(userMessage);
  const systemPrompt = buildSystemPrompt(context, lang);

  // Build messages array
  const messages = [{ role: 'system', content: systemPrompt }];

  // Add conversation history (last 6 turns)
  const recentHistory = conversationHistory.slice(-6);
  for (const turn of recentHistory) {
    messages.push({
      role: turn.role === 'model' ? 'assistant' : 'user',
      content: turn.text,
    });
  }

  // Inject language reminder right before the user message — this forces compliance
  const langReminder = lang === 'hinglish'
    ? '[SYSTEM REMINDER: Respond in Hinglish only] '
    : '[SYSTEM REMINDER: Respond in English only, no Hindi words] ';

  messages.push({ role: 'user', content: langReminder + userMessage });

  try {
    const response = await axios.post(
      GROQ_URL,
      {
        model: GROQ_MODEL,
        messages,
        max_tokens: 120,
        temperature: 0.5,
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
    if (err.response?.status === 429) throw new Error('AI rate limit reached. Please wait a moment.');
    if (err.response?.status === 401) throw new Error('Invalid GROQ_API_KEY. Check your .env file.');
    if (err.response?.status === 400) throw new Error('Invalid request. Please rephrase your question.');
    throw new Error(`AI service error: ${err.message}`);
  }
}

module.exports = { chatWithAria };