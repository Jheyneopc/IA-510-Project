const reviewEl = document.getElementById("review");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const resultEl = document.getElementById("result");
const pillEl = document.getElementById("pill");
const sourceEl = document.getElementById("source");
const confidenceEl = document.getElementById("confidence");
const errorEl = document.getElementById("error");
const hintTextEl = document.getElementById("hintText");

const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");

function setError(msg) {
  if (!msg) {
    errorEl.classList.add("hidden");
    errorEl.textContent = "";
    return;
  }
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}

function setResult({ sentiment, source, confidence }) {
  resultEl.classList.remove("hidden");

  pillEl.classList.remove("good", "bad", "mid");
  pillEl.textContent = (sentiment || "—").toUpperCase();

  if (sentiment === "positive") pillEl.classList.add("good");
  else if (sentiment === "negative") pillEl.classList.add("bad");
  else if (sentiment === "neutral") pillEl.classList.add("mid");

  sourceEl.textContent = source || "—";
  confidenceEl.textContent =
    (typeof confidence === "number") ? confidence.toFixed(3) : "—";

  if (sentiment === "negative") hintTextEl.textContent = "Try adding words like “crash”, “worst”, “broken”, or “scam”.";
  else if (sentiment === "positive") hintTextEl.textContent = "Try adding words like “amazing”, “great”, “best”, or “love”.";
  else hintTextEl.textContent = "Neutral often indicates mixed or low-confidence sentiment.";
}

async function predict(text) {
  // Add demo-safe threshold to request (so you can tune neutral sensitivity)
  const minConfidence = parseFloat(confSlider.value);

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, min_confidence: minConfidence }),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data?.error || "Request failed");
  return data;
}

analyzeBtn.addEventListener("click", async () => {
  setError(null);
  resultEl.classList.add("hidden");

  const text = reviewEl.value.trim();
  if (!text) {
    setError("Please enter some review text.");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";

  try {
    const out = await predict(text);
    if (out.error) setError(out.error);
    setResult(out);
  } catch (e) {
    setError(e.message || "Something went wrong.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze";
  }
});

clearBtn.addEventListener("click", () => {
  reviewEl.value = "";
  setError(null);
  resultEl.classList.add("hidden");
});

confSlider.addEventListener("input", () => {
  confValue.textContent = parseFloat(confSlider.value).toFixed(2);
});

// convenience: Ctrl+Enter runs analysis
reviewEl.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") analyzeBtn.click();
});