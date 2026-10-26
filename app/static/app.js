const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messagesEl = document.querySelector("#messages");
const welcomeEl = document.querySelector("#welcome");
const errorEl = document.querySelector("#error-banner");
const sendButton = document.querySelector("#send-button");
const charCount = document.querySelector("#char-count");
const conversationLabel = document.querySelector("#conversation-label");
const storageKey = "context-chat-conversation";
let conversationId = localStorage.getItem(storageKey);
let sending = false;

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
}

function renderMessages(messages) {
  messagesEl.replaceChildren();
  messagesEl.classList.toggle("has-messages", messages.length > 0);
  welcomeEl.hidden = messages.length > 0;
  conversationLabel.textContent = messages.length ? "Current conversation" : "New conversation";
  for (const message of messages) {
    const row = document.createElement("div");
    row.className = `message ${message.role}`;
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = message.role === "assistant" ? "✳" : "Y";
    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = message.content;
    row.append(avatar, body);
    messagesEl.append(row);
  }
  document.querySelector("#chat-scroll").scrollTop = document.querySelector("#chat-scroll").scrollHeight;
}

async function sendMessage(text) {
  const message = text.trim();
  if (!message || sending) return;
  sending = true;
  errorEl.hidden = true;
  input.value = "";
  input.dispatchEvent(new Event("input"));
  sendButton.disabled = true;
  const before = [...messagesEl.querySelectorAll(".message")].map((row) => ({
    role: row.classList.contains("user") ? "user" : "assistant",
    content: row.querySelector(".message-body").textContent,
  }));
  renderMessages([...before, { role: "user", content: message }]);
  const waiting = document.createElement("div");
  waiting.className = "message assistant pending";
  waiting.innerHTML = '<div class="message-avatar">✳</div><div class="message-body"><span class="typing-dots"><i></i><i></i><i></i></span></div>';
  messagesEl.append(waiting);
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, conversation_id: conversationId }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The message could not be sent.");
    conversationId = data.conversation_id;
    localStorage.setItem(storageKey, conversationId);
    renderMessages(data.messages);
  } catch (error) {
    renderMessages(before);
    showError(error.message || "Could not reach the chat service. Please try again.");
    input.value = message;
    input.dispatchEvent(new Event("input"));
    input.focus();
  } finally {
    sending = false;
    sendButton.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});
input.addEventListener("input", () => {
  charCount.textContent = `${input.value.length.toLocaleString()} / 4,000`;
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
document.querySelector("#new-chat").addEventListener("click", () => {
  conversationId = null;
  localStorage.removeItem(storageKey);
  renderMessages([]);
  errorEl.hidden = true;
  input.focus();
});
document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

if (conversationId) {
  fetch(`/api/conversations/${encodeURIComponent(conversationId)}`)
    .then(async (response) => {
      if (!response.ok) throw new Error("Conversation not found");
      return response.json();
    })
    .then((data) => renderMessages(data.messages))
    .catch(() => {
      localStorage.removeItem(storageKey);
      conversationId = null;
    });
}
