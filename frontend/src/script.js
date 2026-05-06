const API_BASE = window.location.origin.includes("localhost:5500")
  ? "http://localhost:8000"  // Live Server dev mode — backend runs separately
  : "";                      // Docker / nginx — same origin, proxied

const chatWindow = document.getElementById("chat-window");
const input = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const uploadBtn = document.getElementById("upload-btn");
const fileInput = document.getElementById("file-input");
const modelSelect = document.getElementById("model-select");
const usageBar = document.getElementById("usage-bar");
const docBar = document.getElementById("doc-bar");


// ── Session ID ──────────────────────────────────────────────────────────────

function getOrCreateSessionId() {
  let id = sessionStorage.getItem("chat_session_id");
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem("chat_session_id", id);
  }
  return id;
}

function newSessionId() {
  const id = crypto.randomUUID();
  sessionStorage.setItem("chat_session_id", id);
  return id;
}


// ── Document bar ────────────────────────────────────────────────────────────

let uploadedDocs = [];

function renderDocBar() {
  docBar.innerHTML = "";
  uploadedDocs.forEach(name => {
    const chip = document.createElement("span");
    chip.className = "doc-chip";
    chip.textContent = name;
    docBar.appendChild(chip);
  });
}

async function uploadFile(file) {
  const sessionId = getOrCreateSessionId();
  const form = new FormData();
  form.append("file", file);

  uploadBtn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/api/documents/upload?session_id=${sessionId}`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Upload failed");
    }
    const data = await res.json();
    uploadedDocs = data.documents;
    renderDocBar();
  } catch (err) {
    alert(`Upload error: ${err.message}`);
  } finally {
    uploadBtn.disabled = false;
    fileInput.value = "";
  }
}


// ── Rendering ──────────────────────────────────────────────────────────────

function appendMessage(role, content) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = role === "user" ? "You" : "Assistant";

  wrapper.appendChild(bubble);
  wrapper.appendChild(meta);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  return bubble;
}

function showTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";
  wrapper.id = "typing";

  const bubble = document.createElement("div");
  bubble.className = "bubble typing-indicator";
  bubble.innerHTML = "<span></span><span></span><span></span>";

  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeTypingIndicator() {
  document.getElementById("typing")?.remove();
}

function setUsage(usage, model, strategy) {
  usageBar.textContent =
    `Model: ${model} · Strategy: ${strategy} · Prompt: ${usage.prompt_tokens} tok · `+
    `Completion: ${usage.completion_tokens} tok · `+
    `Total: ${usage.total_tokens} tok`;
}


// ── API call ───────────────────────────────────────────────────────────────

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  appendMessage("user", text);

  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;
  showTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: getOrCreateSessionId(),
        model: modelSelect.value,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Request failed");
    }

    const data = await res.json();

    removeTypingIndicator();
    appendMessage("assistant", data.content);
    setUsage(data.usage, data.model, data.strategy);

  } catch (err) {
    removeTypingIndicator();
    appendMessage("assistant", `[ERROR]: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}


// ── Event listeners ────────────────────────────────────────────────────────

sendBtn.addEventListener("click", sendMessage);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-grow textarea
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) uploadFile(file);
});
