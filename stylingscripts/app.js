/* Rewiseorist frontend. Plain JavaScript, no external dependencies. */
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const STORAGE_PROFILE = "rewiseorist.profileId";
const STORAGE_PREFS = "rewiseorist.prefs";
const STORAGE_ONBOARDED = "rewiseorist.onboarded";

const state = {
  profileId: null,
  profile: null,
  profiles: [],
  settings: {},
  topics: [],
  sessions: [],
  currentSessionId: null,
  messages: [],
  resources: [],
  notes: [],
  quiz: [],
  quizIndex: 0,
  quizAnswers: 0,
  quizFeedback: null,
  hintUsed: false,
  mock: { active: false, timer: null, end: 0, results: [] },
  flashcards: [],
  planId: null,
  plan: null,
  plans: [],
  progress: null,
  progressPeriod: "all",
  store: { items: [], balance: { xp: 0, coins: 0, owned: [], badges: [] } },
  storeCategory: "all",
  modelStatus: null,
  onboardingStep: 0,
  pendingHomePrompt: "",
  drawing: { strokes: [], current: null, drawing: false },
  dismissal: {},
};

const PAGE_TITLES = {
  home: "Home",
  study: "Study",
  resources: "Resources",
  progress: "Progress",
  store: "Store",
  settings: "Settings",
};

const ONBOARDING_STEPS = [
  {
    title: "Welcome to Rewiseorist",
    html: `<p>This is your private revision workspace. Everything runs on this laptop with your own trained model.</p>
      <label for="onboarding-name">What should the tutor call you?</label>
      <input id="onboarding-name" type="text" maxlength="60" placeholder="Your first name" value="">`,
  },
  {
    title: "Bring your class notes",
    html: `<p>Open <strong>Resources</strong> to upload a PDF, Word document, slides, Markdown notes or a whole folder. The app keeps the file and page or slide reference so answers can cite it.</p>
      <ul><li>One file: up to 20 MB.</li><li>One folder: up to 50 files and about 200 MB.</li><li>Images need optional OCR; audio needs a transcript.</li></ul>`,
  },
  {
    title: "Study with sources",
    html: `<p>In <strong>Study</strong>, choose a topic, ask a question, and check the source chips under each answer. Then try a checked quiz. Hints are recorded so your progress reflects independent work.</p>`,
  },
  {
    title: "See real progress",
    html: `<p><strong>Progress</strong> shows accuracy, active minutes, topic levels and plan adherence. <strong>Store</strong> items are cosmetic and bought only with earned coins — there is no real money and no paywall.</p>`,
  },
  {
    title: "Local by default",
    html: `<p>Local mode answers with the pretrained SmolLM2 360M model running through Ollama on this laptop; the performance classifier is a transparent rule-based fallback when the trained model is absent; external API mode is optional and asks permission; notes, chats and sources stay in the local database.</p>
      <p class="fine-print">The model is small and can make mistakes. Check its cited source.</p>`,
  },
];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function readPrefs() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_PREFS) || "{}");
  } catch {
    return {};
  }
}

function savePrefs(extra = {}) {
  const prefs = { ...readPrefs(), ...extra };
  localStorage.setItem(STORAGE_PREFS, JSON.stringify(prefs));
  return prefs;
}

function applyAppearance() {
  const prefs = readPrefs();
  const settings = state.settings || {};
  const theme = settings.theme || prefs.theme || "light";
  const contrast = settings.contrast || prefs.contrast || "normal";
  document.body.dataset.theme = theme;
  document.body.dataset.contrast = contrast;
  document.body.dataset.reduceMotion = String(Boolean(prefs.reduceMotion));
  document.body.dataset.largeText = String(Boolean(prefs.largeText));
  const themeSelect = $("#theme-select");
  const contrastSelect = $("#contrast-select");
  if (themeSelect) themeSelect.value = theme;
  if (contrastSelect) contrastSelect.value = contrast;
  const reduce = $("#reduce-motion");
  const large = $("#large-text");
  if (reduce) reduce.checked = Boolean(prefs.reduceMotion);
  if (large) large.checked = Boolean(prefs.largeText);
}

function applyAiModeVisibility() {
  const wrap = $("#external-permission-wrap");
  if (!wrap) return;
  const isApi = (state.settings?.ai_mode || "local") === "api";
  wrap.hidden = !isApi;
  if (!isApi) {
    const box = $("#external-permission");
    if (box) box.checked = false;
  }
}

function toast(message, kind = "info", timeout = 4500) {  const region = $("#toast-region");
  if (!region) return;
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.textContent = message;
  region.appendChild(node);
  window.setTimeout(() => node.remove(), timeout);
}

function setBusy(button, busy, label = "Working…") {
  if (!button) return;
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.textContent = label;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
  }
}

async function api(path, options = {}) {
  const method = options.method || "GET";
  let url = path;
  const fetchOptions = { method, headers: { ...(options.headers || {}) } };
  if (options.form) {
    fetchOptions.body = options.form;
    if (state.profileId && !options.form.has("profile_id")) {
      options.form.append("profile_id", String(state.profileId));
    }
  } else if (options.body) {
    const body = { ...options.body };
    if (state.profileId && body.profile_id === undefined && options.profile !== false) {
      body.profile_id = state.profileId;
    }
    fetchOptions.headers["Content-Type"] = "application/json";
    fetchOptions.body = JSON.stringify(body);
  } else if (method === "GET" && state.profileId && options.profile !== false) {
    url += `${url.includes("?") ? "&" : "?"}profile_id=${encodeURIComponent(state.profileId)}`;
  }
  const response = await fetch(url, fetchOptions);
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { ok: false, error: text.slice(0, 240) || `Request failed (${response.status})` };
  }
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  return data;
}

function requireProfile() {
  if (state.profileId) return true;
  toast("Create or choose a profile first.", "error");
  go("settings");
  return false;
}

/* ------------------------------------------------------------------ */
/* Navigation                                                         */
/* ------------------------------------------------------------------ */

function go(page, options = {}) {
  if (!PAGE_TITLES[page]) page = "home";
  $$(".page").forEach((section) => section.classList.toggle("is-active", section.id === `page-${page}`));
  $$(".nav-item").forEach((button) => {
    const active = button.dataset.page === page;
    button.classList.toggle("is-active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $("#page-title").textContent = PAGE_TITLES[page];
  document.title = `${PAGE_TITLES[page]} — Rewiseorist`;
  if (options.tab) switchStudyTab(options.tab);
  if (options.focus) {
    const target = $(`#${options.focus}`);
    if (target) target.focus();
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
  const main = $("#main-content");
  if (main) main.focus({ preventScroll: true });
  if (page === "progress") loadProgress(state.progressPeriod);
  if (page === "store") loadStore();
  if (page === "settings") loadSettingsPage();
  if (page === "resources") loadResources();
  if (page === "study") {
    loadSessions();
    if (!state.topics.length) loadTopics();
  }
}

function switchStudyTab(tab) {
  $$(".tab").forEach((button) => {
    const active = button.dataset.studyTab === tab;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("is-active", panel.id === `study-tab-${tab}`));
  if (tab === "plans") loadPlanHistory();
  if (tab === "notes") loadNotes();
}

function bindNavigation() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => go(button.dataset.page)));
  $$("[data-goto]").forEach((button) => button.addEventListener("click", () => {
    go(button.dataset.goto, { tab: button.dataset.studyTab, focus: button.dataset.focus });
  }));
  $$(".tab").forEach((button) => button.addEventListener("click", () => switchStudyTab(button.dataset.studyTab)));
  $("#profile-button").addEventListener("click", () => go("settings"));
  window.addEventListener("hashchange", () => {
    const page = window.location.hash.replace("#", "");
    if (PAGE_TITLES[page]) go(page);
  });
}

/* ------------------------------------------------------------------ */
/* Profiles and settings                                              */
/* ------------------------------------------------------------------ */

async function loadProfiles() {
  const data = await api("/api/profiles", { profile: false });
  state.profiles = data.profiles || [];
  renderProfileOptions();
  return state.profiles;
}

function renderProfileOptions() {
  const select = $("#profile-select");
  if (!select) return;
  select.innerHTML = state.profiles
    .map((profile) => `<option value="${profile.id}">${escapeHtml(profile.name)}</option>`)
    .join("");
  if (state.profileId) select.value = String(state.profileId);
}

async function activateProfile(profileId) {
  try {
    const data = await api(`/api/profiles/${profileId}`, { profile: false });
    state.profileId = data.profile.id;
    state.profile = data.profile;
    state.settings = data.settings || {};
    localStorage.setItem(STORAGE_PROFILE, String(state.profileId));
    applyAppearance();
    applyAiModeVisibility();
    renderProfile();
    await Promise.allSettled([
      loadTopics(), loadSessions(), loadNotes(), loadResources(), loadStore(),
      loadProgress("all"), loadModelStatus(), loadSettingsPage(),
    ]);
    return true;
  } catch (error) {
    localStorage.removeItem(STORAGE_PROFILE);
    state.profileId = null;
    toast(error.message, "error");
    return false;
  }
}

function renderProfile() {
  if (!state.profile) return;
  const name = state.profile.name || "Student";
  $("#profile-name").textContent = name;
  $("#profile-level").textContent = `${state.profile.xp || 0} XP · ${state.profile.coins || 0} coins`;
  const owned = state.store.balance.owned || [];
  const student = owned.find((item) => item.equipped && String(item.item_key).startsWith("student-"));
  const avatarMap = {
    "student-default": "🧑‍🎓", "student-glasses": "🤓", "student-cap": "🎓",
  };
  $("#profile-avatar").textContent = student ? avatarMap[student.item_key] || "🧑‍🎓" : "🧑‍🎓";
  $("#chat-personality").textContent = state.settings.personality || "Friendly";
  $("#settings-version").textContent = `v1.0.0 · ${name}`;
}

async function createProfile(name) {
  const data = await api("/api/profiles", { method: "POST", body: { name }, profile: false });
  state.profiles = state.profiles.filter((item) => item.id !== data.profile.id).concat(data.profile);
  renderProfileOptions();
  await activateProfile(data.profile.id);
  return data.profile;
}

async function loadSettingsPage() {
  if (!state.profileId) return;
  try {
    const data = await api(`/api/settings?profile_id=${state.profileId}`, { profile: false });
    state.settings = data.settings || {};
    const s = state.settings;
    $("#personality-select").value = s.personality || "Friendly";
    $("#theme-select").value = s.theme || "light";
    $("#contrast-select").value = s.contrast || "normal";
    $("#api-base").value = s.api_base || "";
    $("#api-model").value = s.api_model || "";
    $("#web-permission-default").checked = Boolean(s.web_permission_default);
    $$('input[name="ai-mode"]').forEach((radio) => {
      radio.checked = radio.value === (s.ai_mode || "local");
    });
    $("#api-settings").hidden = (s.ai_mode || "local") !== "api";
    if (data.secrets && data.secrets.api_key_present) {
      $("#api-key").placeholder = "Key stored locally — enter a new key to replace";
    }
    renderProfileOptions();
    applyAppearance();
    applyAiModeVisibility();
    renderProfile();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function saveSettings() {
  if (!requireProfile()) return;
  const mode = $('input[name="ai-mode"]:checked')?.value || "local";
  const payload = {
    personality: $("#personality-select").value,
    theme: $("#theme-select").value,
    contrast: $("#contrast-select").value,
    ai_mode: mode,
    api_base: $("#api-base").value.trim(),
    api_model: $("#api-model").value.trim(),
    web_permission_default: $("#web-permission-default").checked,
  };
  try {
    const data = await api("/api/settings", { method: "POST", body: payload });
    state.settings = data.settings;
    savePrefs({ theme: payload.theme, contrast: payload.contrast });
    applyAppearance();
    applyAiModeVisibility();
    renderProfile();
    $("#settings-status").textContent = "Saved.";
    toast("Settings saved.", "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function saveApiKey() {
  if (!requireProfile()) return;
  const key = $("#api-key").value.trim();
  try {
    await api("/api/settings/api-key", { method: "POST", body: { api_key: key } });
    $("#api-key").value = "";
    $("#api-key").placeholder = key ? "Key stored locally" : "No key stored";
    toast(key ? "API key saved locally." : "API key removed.", "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function testApi() {
  if (!requireProfile()) return;
  try {
    const data = await api("/api/api/test", {
      method: "POST",
      body: {
        permission: true,
        api_base: $("#api-base").value.trim(),
        api_key: $("#api-key").value.trim(),
      },
    });
    toast(data.message || "API endpoint reachable.", "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

/* ------------------------------------------------------------------ */
/* Home                                                               */
/* ------------------------------------------------------------------ */

async function loadDashboard() {
  try {
    const [progress, sessions] = await Promise.all([
      api(`/api/progress?period=all&profile_id=${state.profileId}`, { profile: false }),
      api(`/api/sessions?limit=6&profile_id=${state.profileId}`, { profile: false }),
    ]);
    state.progress = progress.progress;
    const p = progress.progress;
    $("#home-xp").textContent = p.total_xp ?? 0;
    $("#home-coins").textContent = p.total_coins ?? 0;
    $("#home-accuracy").textContent = p.accuracy == null ? "—" : `${Math.round(p.accuracy * 100)}%`;
    $("#home-sessions").textContent = sessions.sessions.length;
    $("#home-plan-pill").textContent = p.current_plan ? "Accepted plan" : "No plan yet";
    const list = $("#home-sessions-list");
    list.innerHTML = sessions.sessions.length
      ? sessions.sessions.slice(0, 5).map((session) => `
        <button class="session-item" data-open-session="${session.id}">
          <strong>${escapeHtml(session.title)}</strong>
          <small>${escapeHtml(session.topic_name || "General")} · ${new Date(session.updated_at).toLocaleDateString()}</small>
        </button>`).join("")
      : '<div class="empty-state">No saved sessions yet. Start a chat in Study.</div>';
    $$("[data-open-session]", list).forEach((button) => button.addEventListener("click", () => openSession(Number(button.dataset.openSession))));
    const actions = $("#home-actions");
    actions.querySelectorAll("[data-goto]").forEach((button) => button.addEventListener("click", () => go(button.dataset.goto, { tab: button.dataset.studyTab })));
    renderModelBadge();
  } catch (error) {
    // A new profile may have no progress yet; keep the page usable.
    if (!String(error.message).includes("not found")) toast(error.message, "error");
  }
}

function renderModelBadge() {
  const badge = $("#home-model-badge");
  const sidebar = $("#sidebar-model-status");
  if (!badge || !sidebar) return;
  const ollama = state.ollamaStatus;
  if (ollama) {
    if (ollama.available && ollama.model_present) {
      badge.textContent = "Revision: SmolLM2 360M · Ollama";
      sidebar.textContent = "Revision ready · Ollama local";
    } else if (ollama.available) {
      badge.textContent = "Run ollama pull smollm2:360m";
      sidebar.textContent = "Model not installed";
    } else {
      badge.textContent = "Ollama not running";
      sidebar.textContent = "Ollama offline";
    }
    return;
  }
  const status = state.modelStatus || {};
  const lm = status.language_model || {};
  const clf = status.performance_model || {};
  if (status.ready) {
    badge.textContent = `Local model ${lm.version || "ready"} · classifier ${clf.version || "ready"}`;
    sidebar.textContent = status.loaded ? "Model loaded · local" : "Model ready · cold start";
  } else {
    const missing = (status.missing_files || []).join(", ");
    badge.textContent = missing ? `Model files missing: ${missing}` : "Checking local model…";
    sidebar.textContent = "Model not ready";
  }
}

/* ------------------------------------------------------------------ */
/* Topics                                                             */
/* ------------------------------------------------------------------ */

async function loadTopics() {
  if (!state.profileId) return;
  try {
    const data = await api(`/api/topics?profile_id=${state.profileId}`, { profile: false });
    state.topics = data.topics || [];
    renderTopicOptions();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderTopicOptions() {
  const selects = [$("#chat-topic"), $("#practice-topic")];
  selects.forEach((select) => {
    if (!select) return;
    const prior = select.value;
    select.innerHTML = '<option value="">Choose a topic…</option>' + state.topics
      .map((topic) => `<option value="${topic.id}">${escapeHtml(topic.name)}</option>`)
      .join("");
    if (state.topics.some((topic) => String(topic.id) === prior)) select.value = prior;
    else if (state.topics.length === 1) select.value = String(state.topics[0].id);
  });
}

async function ensureTopic(name = "General") {
  if (!requireProfile()) throw new Error("No profile selected");
  const clean = (name || "General").trim() || "General";
  const existing = state.topics.find((topic) => topic.name.toLowerCase() === clean.toLowerCase());
  if (existing) return existing;
  const data = await api("/api/topics", { method: "POST", body: { name: clean, difficulty: "medium" } });
  state.topics.push(data.topic);
  renderTopicOptions();
  const select = $("#chat-topic");
  if (select) select.value = String(data.topic.id);
  return data.topic;
}

async function addTopic() {
  const name = $("#new-topic-name").value.trim();
  if (!name) return toast("Type a topic name first.", "error");
  try {
    const topic = await ensureTopic(name);
    $("#new-topic-name").value = "";
    const select = $("#chat-topic");
    if (select) select.value = String(topic.id);
    if ($("#practice-topic")) $("#practice-topic").value = String(topic.id);
    toast(`Topic “${topic.name}” ready.`, "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

function currentTopicId() {
  const value = $("#chat-topic")?.value || $("#practice-topic")?.value || "";
  return value ? Number(value) : null;
}

function currentTopicName() {
  const id = currentTopicId();
  return state.topics.find((topic) => topic.id === id)?.name || "General";
}

/* ------------------------------------------------------------------ */
/* Sessions and chat                                                  */
/* ------------------------------------------------------------------ */

async function loadSessions() {
  if (!state.profileId) return;
  try {
    const data = await api(`/api/sessions?limit=100&profile_id=${state.profileId}`, { profile: false });
    state.sessions = data.sessions || [];
    renderSessions();
    $("#session-count").textContent = String(state.sessions.length);
    loadDashboard();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderSessions() {
  const list = $("#session-list");
  if (!list) return;
  list.innerHTML = state.sessions.length
    ? state.sessions.map((session) => `
      <button class="session-item ${state.currentSessionId === session.id ? "is-active" : ""}" data-session="${session.id}">
        <strong>${escapeHtml(session.title)}</strong>
        <small>${escapeHtml(session.topic_name || "General")} · ${new Date(session.updated_at).toLocaleDateString()}</small>
      </button>`).join("")
    : '<div class="empty-state">No sessions yet.</div>';
  $$("[data-session]", list).forEach((button) => button.addEventListener("click", () => openSession(Number(button.dataset.session))));
}

async function openSession(sessionId) {
  try {
    const data = await api(`/api/sessions/${sessionId}?profile_id=${state.profileId}`, { profile: false });
    state.currentSessionId = data.session.id;
    state.messages = data.messages || [];
    if (data.session.topic_id) $("#chat-topic").value = String(data.session.topic_id);
    $("#chat-input").value = data.session.draft || "";
    renderMessages();
    renderSessions();
    go("study", { tab: "chat" });
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderMessages() {
  const area = $("#chat-messages");
  if (!area) return;
  if (!state.messages.length) {
    area.innerHTML = '<div class="empty-state">Ask a question to start this session. Answers cite the sources they use.</div>';
    return;
  }
  area.innerHTML = state.messages.map((message) => {
    const sources = (message.sources || []).map((source) => {
      const label = source.original_name || source.file || "source";
      const ref = source.source_ref || source.page_ref || "";
      return `<span class="source-chip">${escapeHtml(label)}${ref ? ` · ${escapeHtml(ref)}` : ""}</span>`;
    }).join("");
    const model = message.model_version ? ` · ${escapeHtml(message.model_version)}` : "";
    const unverified = message.role === "assistant" && (message.verified === false || message.verified === 0)
      ? '<span class="unverified-chip">Unverified</span>'
      : "";
    return `<article class="message ${message.role}">
      <div class="message-meta"><strong>${message.role === "user" ? "You" : "Rewiseorist"}</strong><span>${new Date(message.created_at).toLocaleString()}${model}${unverified}</span></div>
      <p>${escapeHtml(message.content)}</p>
      ${sources ? `<div>${sources}</div>` : ""}
    </article>`;
  }).join("");
  area.scrollTop = area.scrollHeight;
}

function newSession() {
  state.currentSessionId = null;
  state.messages = [];
  $("#chat-input").value = "";
  $("#chat-attachments").value = "";
  $("#draft-status").textContent = "";
  renderMessages();
  renderSessions();
  switchStudyTab("chat");
  $("#chat-input").focus();
}

let draftTimer = null;
function scheduleDraftSave() {
  const status = $("#draft-status");
  if (status) status.textContent = "Writing draft…";
  clearTimeout(draftTimer);
  draftTimer = setTimeout(async () => {
    const draft = $("#chat-input").value;
    if (!state.currentSessionId) {
      if (status) status.textContent = draft ? "Draft kept in this tab" : "";
      return;
    }
    try {
      await api(`/api/sessions/${state.currentSessionId}/draft`, { method: "POST", body: { draft } });
      if (status) status.textContent = draft ? "Draft saved" : "Draft cleared";
    } catch {
      if (status) status.textContent = "Draft not saved";
    }
  }, 900);
}

async function sendChat(event) {
  event.preventDefault();
  if (!requireProfile()) return;
  const input = $("#chat-input");
  const message = input.value.trim();
  if (!message) return;
  const button = $("#send-chat-button");
  setBusy(button, true, "Thinking…");
  try {
    const topic = await ensureTopic(currentTopicName());
    $("#chat-topic").value = String(topic.id);
    const files = Array.from($("#chat-attachments").files || []);
    let attachments = [];
    if (files.length) {
      await uploadFileList(files, topic.name);
      attachments = files.map((file) => file.name);
    }
    const data = await api("/api/chat", {
      method: "POST",
      body: {
        message,
        topic_id: topic.id,
        session_id: state.currentSessionId,
        attachments,
        api_permission: Boolean($("#external-permission")?.checked),
      },
    });
    state.currentSessionId = data.session_id;
    input.value = "";
    $("#chat-attachments").value = "";
    $("#draft-status").textContent = "";
    await openSession(data.session_id);
    toast("Answered by SmolLM2 360M (Ollama).", "success", 2500);
    loadSessions();
    loadProgress(state.progressPeriod);
  } catch (error) {
    toast(error.message, "error", 7000);
  } finally {
    setBusy(button, false);
  }
}

/* ------------------------------------------------------------------ */
/* Uploads and resources                                              */
/* ------------------------------------------------------------------ */

async function uploadFileList(files, topicName = "General") {
  const form = new FormData();
  files.forEach((file) => {
    form.append("files", file, file.name);
    form.append("relative_paths", file.webkitRelativePath || file.name);
  });
  const topic = await ensureTopic(topicName);
  form.append("topic_id", String(topic.id));
  form.append("topic_name", topic.name);
  return api("/api/resources/upload", { method: "POST", form });
}

async function loadResources() {
  if (!state.profileId) return;
  try {
    const data = await api(`/api/resources?profile_id=${state.profileId}`, { profile: false });
    state.resources = data.resources || [];
    renderResources();
    renderPlugins(data.plugins || []);
  } catch (error) {
    toast(error.message, "error");
  }
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function renderResources() {
  const list = $("#resource-list");
  if (!list) return;
  $("#resource-count").textContent = String(state.resources.length);
  list.innerHTML = state.resources.length
    ? state.resources.map((resource) => `
      <div class="resource-item">
        <div>
          <strong>${escapeHtml(resource.original_name)}</strong>
          <div class="resource-meta">${escapeHtml(resource.relative_path || "")} · ${formatBytes(resource.size_bytes)} · ${escapeHtml(resource.status_label || resource.status)}</div>
          ${resource.error ? `<div class="resource-meta">${escapeHtml(resource.error)}</div>` : ""}
        </div>
        <div class="resource-actions">
          <button class="button ghost" data-resource-note="${resource.id}" data-resource-name="${escapeHtml(resource.original_name)}">Note</button>
        </div>
      </div>`).join("")
    : '<div class="empty-state">No sources imported yet. Upload a file or folder above.</div>';
  $$("[data-resource-note]", list).forEach((button) => button.addEventListener("click", () => {
    $("#note-target").value = button.dataset.resourceName;
    go("study", { tab: "notes" });
    loadNotes();
  }));
}

function renderPlugins(plugins) {
  const list = $("#plugin-list");
  if (!list) return;
  list.innerHTML = plugins.map((plugin) => `
    <div class="plugin-row">
      <div><strong>${escapeHtml(plugin.name)}</strong><small>${escapeHtml(plugin.status)}</small></div>
      <button class="button ghost" data-plugin="${escapeHtml(plugin.plugin_key)}" data-enabled="${plugin.enabled ? "0" : "1"}">${plugin.enabled ? "Disable" : "Enable"}</button>
    </div>`).join("");
  $$("[data-plugin]", list).forEach((button) => button.addEventListener("click", async () => {
    try {
      const enabled = button.dataset.enabled === "1";
      await api(`/api/plugins/${button.dataset.plugin}`, { method: "POST", body: { enabled } });
      await loadResources();
    } catch (error) {
      toast(error.message, "error");
    }
  }));
}

async function submitUpload(event) {
  event.preventDefault();
  if (!requireProfile()) return;
  const topicName = $("#upload-topic").value.trim() || "General";
  const folderFiles = Array.from($("#upload-folder").files || []);
  const fileFiles = Array.from($("#upload-files").files || []);
  const files = folderFiles.length ? folderFiles : fileFiles;
  if (!files.length) return toast("Choose files or a folder first.", "error");
  const button = event.submitter || $("#upload-form button[type=submit]");
  setBusy(button, true, "Importing…");
  $("#upload-status").textContent = "";
  try {
    const data = await uploadFileList(files, topicName);
    $("#upload-status").textContent = `${data.count} file(s) imported into ${data.topic.name}.`;
    toast(`${data.count} source(s) imported.`, "success");
    $("#upload-files").value = "";
    $("#upload-folder").value = "";
    $("#upload-topic").value = "";
    await loadResources();
    await loadDashboard();
  } catch (error) {
    $("#upload-status").textContent = error.message;
    toast(error.message, "error", 7000);
  } finally {
    setBusy(button, false);
  }
}

async function submitResearch(event) {
  event.preventDefault();
  if (!requireProfile()) return;
  if (!$("#research-permission").checked) return toast("Tick the permission box first.", "error");
  const url = $("#research-url").value.trim();
  if (!url) return toast("Enter a URL.", "error");
  try {
    await api("/api/research", { method: "POST", body: { url, permission: true } });
    $("#research-url").value = "";
    $("#research-permission").checked = false;
    toast("Page saved as a source.", "success");
    await loadResources();
  } catch (error) {
    toast(error.message, "error", 7000);
  }
}

/* ------------------------------------------------------------------ */
/* Quizzes, mock exams, flashcards, classification                    */
/* ------------------------------------------------------------------ */

async function fetchQuiz() {
  const select = $("#practice-topic");
  let topic = state.topics.find((item) => String(item.id) === select?.value);
  if (!topic) topic = await ensureTopic(currentTopicName());
  if (select) select.value = String(topic.id);
  const data = await api(`/api/quiz?topic_id=${topic.id}&count=5&profile_id=${state.profileId}`, { profile: false });
  return data.questions || [];
}

async function loadQuizForPractice(mock = false) {
  if (!requireProfile()) return;
  try {
    state.quiz = await fetchQuiz();
    state.quizIndex = 0;
    state.quizAnswers = 0;
    state.quizFeedback = null;
    state.hintUsed = false;
    state.mock = { active: mock, timer: null, end: mock ? Date.now() + 600000 : 0, results: [] };
    renderQuiz();
    if (mock) startMockTimer();
    else $("#mock-timer").hidden = true;
    if (!state.quiz.length) $("#quiz-area").innerHTML = '<div class="empty-state">No questions are available yet. Upload a source for this topic first.</div>';
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderQuiz() {
  const area = $("#quiz-area");
  if (!area) return;
  const question = state.quiz[state.quizIndex];
  if (!question) {
    area.innerHTML = `<div class="empty-state">Quiz complete. ${state.mock.active ? "Mock exam still running." : "Load questions to try again."}</div>`;
    return;
  }
  $("#quiz-progress").textContent = `${state.quizIndex + 1} / ${state.quiz.length}`;
  if (state.quizFeedback) {
    const result = state.quizFeedback.result || {};
    const kind = result.correct ? "correct" : result.marks > 0 ? "partial" : "incorrect";
    area.innerHTML = `
      <div class="question-card">
        <h4>${escapeHtml(question.prompt)}</h4>
        <div class="feedback ${kind}">
          <strong>${result.correct ? "Correct" : result.marks > 0 ? "Partly correct" : "Not yet"}</strong>
          <p>${escapeHtml(result.feedback || "")}</p>
          <p><strong>Model answer:</strong> ${escapeHtml(state.quizFeedback.model_answer || "")}</p>
          ${state.quizFeedback.explanation ? `<p>${escapeHtml(state.quizFeedback.explanation)}</p>` : ""}
        </div>
        <button class="button primary" id="next-question-button" type="button">${state.quizIndex + 1 < state.quiz.length ? "Next question" : "Finish quiz"}</button>
      </div>`;
    $("#next-question-button").addEventListener("click", () => {
      state.quizFeedback = null;
      state.hintUsed = false;
      state.quizIndex += 1;
      renderQuiz();
    });
    return;
  }
  const options = (question.options || []).map((option, index) => `
    <label><input type="radio" name="quiz-option" value="${escapeHtml(option)}"> ${String.fromCharCode(65 + index)}. ${escapeHtml(option)}</label>
  `).join("");
  area.innerHTML = `
    <form class="question-card" id="quiz-form">
      <h4>${escapeHtml(question.prompt)}</h4>
      <p class="fine-print">${escapeHtml(question.source_ref || "No source reference")}</p>
      ${question.type === "mcq" ? `<div class="options">${options}</div>` : '<label for="quiz-answer">Your answer</label><textarea id="quiz-answer" rows="4" required></textarea>'}
      <div class="form-actions">
        <button class="button ghost" id="hint-button" type="button">Show hint</button>
        <button class="button primary" type="submit">${state.mock.active ? "Save and next" : "Submit answer"}</button>
        <span class="fine-print" id="hint-text"></span>
      </div>
    </form>`;
  $("#hint-button").addEventListener("click", () => {
    state.hintUsed = true;
    $("#hint-text").textContent = question.hint || "Re-read the linked source and identify the key term.";
  });
  $("#quiz-form").addEventListener("submit", submitQuizAnswer);
}

async function submitQuizAnswer(event) {
  event.preventDefault();
  const question = state.quiz[state.quizIndex];
  if (!question) return;
  let answer = "";
  if (question.type === "mcq") answer = $('input[name="quiz-option"]:checked')?.value || "";
  else answer = $("#quiz-answer").value.trim();
  if (!answer) return toast("Choose or type an answer first.", "error");
  try {
    const data = await api("/api/quiz/answer", {
      method: "POST",
      body: {
        question_id: question.id,
        answer,
        hint_used: state.hintUsed,
        active_seconds: 30,
        session_id: state.currentSessionId,
      },
    });
    state.quizAnswers += 1;
    $("#quiz-progress").textContent = `${state.quizAnswers} answered`;
    if (state.mock.active) {
      state.mock.results.push({ question, answer, data });
      state.quizIndex += 1;
      if (state.quizIndex >= state.quiz.length) finishMock();
      else renderQuiz();
    } else {
      state.quizFeedback = data;
      renderQuiz();
    }
    loadProgress(state.progressPeriod);
    loadDashboard();
  } catch (error) {
    toast(error.message, "error", 7000);
  }
}

function startMockTimer() {
  clearInterval(state.mock.timer);
  $("#mock-timer").hidden = false;
  const tick = () => {
    if (!state.mock.active) return;
    const left = Math.max(0, state.mock.end - Date.now());
    const minutes = Math.floor(left / 60000);
    const seconds = Math.floor((left % 60000) / 1000);
    $("#mock-timer").innerHTML = `Mock exam: <strong>${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}</strong>`;
    if (left <= 0) finishMock(true);
  };
  tick();
  state.mock.timer = setInterval(tick, 1000);
}

function finishMock(timeUp = false) {
  clearInterval(state.mock.timer);
  state.mock.active = false;
  state.mock.timer = null;
  $("#mock-timer").hidden = true;
  const marks = state.mock.results.reduce((sum, item) => sum + (item.data.result?.marks || 0), 0);
  const possible = state.mock.results.length;
  $("#quiz-area").innerHTML = `
    <div class="result-box">
      <h3>${timeUp ? "Time is up" : "Mock exam complete"}</h3>
      <p><strong>${marks.toFixed(1)} / ${possible}</strong> from ${possible} question(s).</p>
      <p>Check the feedback in each question next time, then target the weakest topic in Progress.</p>
      <button class="button secondary" id="restart-mock-button" type="button">Start another mock</button>
    </div>`;
  $("#restart-mock-button").addEventListener("click", () => loadQuizForPractice(true));
  loadProgress(state.progressPeriod);
}

async function startFlashcards() {
  if (!requireProfile()) return;
  try {
    const questions = state.quiz.length ? state.quiz : await fetchQuiz();
    state.flashcards = questions;
    const area = $("#flashcard-area");
    area.hidden = false;
    area.innerHTML = questions.map((question, index) => `
      <button class="flashcard" data-card="${index}" type="button">
        <div class="front"><strong>${escapeHtml(question.prompt)}</strong><p class="fine-print">Click to reveal</p></div>
        <div class="back"><strong>Answer</strong><p>${escapeHtml(question.answer || "See the checked quiz.")}</p>${question.hint ? `<p class="fine-print">${escapeHtml(question.hint)}</p>` : ""}</div>
      </button>`).join("");
    $$("[data-card]", area).forEach((card) => card.addEventListener("click", () => card.classList.toggle("is-flipped")));
  } catch (error) {
    toast(error.message, "error");
  }
}

async function classifyPerformance() {
  if (!requireProfile()) return;
  const topicId = currentTopicId();
  if (!topicId) return toast("Choose a topic first.", "error");
  const button = $("#classify-button");
  setBusy(button, true, "Checking…");
  try {
    const data = await api("/api/classify", { method: "POST", body: { topic_id: topicId } });
    $("#classify-result").classList.remove("empty-state");
    $("#classify-result").innerHTML = `<strong>${escapeHtml(data.category)}</strong><p>${escapeHtml(data.reason || "")}</p><p class="fine-print">Model: ${escapeHtml(data.model_version || "local classifier")}</p>`;
    loadProgress(state.progressPeriod);
  } catch (error) {
    toast(error.message, "error", 7000);
  } finally {
    setBusy(button, false);
  }
}

async function loadImprovement() {
  if (!requireProfile()) return;
  const topicId = currentTopicId();
  if (!topicId) return toast("Choose a topic first.", "error");
  try {
    const data = await api(`/api/quiz/improvement?topic_id=${topicId}&profile_id=${state.profileId}`, { profile: false });
    const improvement = data.improvement || {};
    const list = $("#improvement-list");
    list.innerHTML = improvement.ready
      ? (improvement.steps || []).map((step) => `<li>${escapeHtml(step)}</li>`).join("")
      : `<li>${escapeHtml(improvement.message || "Complete more checked practice first.")}</li>`;
  } catch (error) {
    toast(error.message, "error");
  }
}

/* ------------------------------------------------------------------ */
/* Plans                                                              */
/* ------------------------------------------------------------------ */

async function generatePlan() {
  if (!requireProfile()) return;
  const button = $("#generate-plan-button");
  setBusy(button, true, "Building…");
  try {
    const data = await api("/api/plans/propose", {
      method: "POST",
      body: {
        horizon_days: Number($("#plan-days").value) || 7,
        daily_minutes: Number($("#plan-minutes").value) || 45,
        title: "Revision plan",
      },
    });
    state.plan = data.plan;
    state.planId = data.stored.id;
    renderPlan(data.plan);
    $("#download-plan-button").disabled = false;
    await loadPlanHistory();
    toast("Draft plan generated. Review it before accepting.", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(button, false);
  }
}

function renderPlan(plan) {
  const output = $("#plan-output");
  if (!plan) {
    output.className = "plan-output empty-state";
    output.textContent = "Generate a plan to see priority and scheduled blocks.";
    return;
  }
  output.className = "plan-output";
  const blocks = plan.blocks || [];
  const days = {};
  blocks.forEach((block) => {
    if (!days[block.date]) days[block.date] = [];
    days[block.date].push(block);
  });
  const dayHtml = Object.entries(days).map(([day, items]) => `
    <section class="plan-day">
      <h4>${escapeHtml(new Date(`${day}T00:00:00`).toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "short" }))}</h4>
      ${items.map((block) => {
        const hour = 9 + Math.floor((block.start_minute || 0) / 60);
        const minute = String((block.start_minute || 0) % 60).padStart(2, "0");
        return `<div class="plan-block"><span class="time">${String(hour).padStart(2, "0")}:${minute}</span><span><strong>${escapeHtml(block.topic)}</strong> — ${escapeHtml(block.activity)} (${block.minutes} min)<br><small>${escapeHtml(block.reason || "")}</small></span></div>`;
      }).join("")}
    </section>`).join("");
  const uncovered = (plan.uncovered || []).length
    ? `<div class="feedback partial"><strong>Needs extra time</strong><ul>${plan.uncovered.map((item) => `<li>${escapeHtml(item.topic)} — ${item.minutes} min</li>`).join("")}</ul></div>`
    : "";
  output.innerHTML = `
    <p><strong>${escapeHtml(plan.summary || "")}</strong></p>
    <div class="form-actions">
      <button class="button primary" id="accept-plan-button" type="button">Accept this draft</button>
      <button class="button ghost" id="reject-plan-button" type="button">Reject</button>
      <button class="button ghost" id="revise-plan-button" type="button">Generate another</button>
    </div>
    ${dayHtml || '<div class="empty-state">No blocks were scheduled. Add topics or more available minutes.</div>'}
    ${uncovered}`;
  $("#accept-plan-button").addEventListener("click", () => acceptCurrentPlan());
  $("#reject-plan-button").addEventListener("click", () => rejectCurrentPlan());
  $("#revise-plan-button").addEventListener("click", generatePlan);
}

async function loadPlanHistory() {
  if (!state.profileId) return;
  try {
    const data = await api(`/api/plans?profile_id=${state.profileId}`, { profile: false });
    state.plans = data.plans || [];
    if (!state.planId && state.plans.length) state.planId = state.plans[0].id;
    if (state.planId) {
      const full = await api(`/api/plans/${state.planId}?profile_id=${state.profileId}`, { profile: false });
      renderPlanVersions(full.plan);
      if (!state.plan) {
        const accepted = (full.plan.versions || []).find((version) => version.accepted);
        if (accepted) {
          state.plan = accepted.plan;
          renderPlan(accepted.plan);
        }
      }
    }
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderPlanVersions(plan) {
  const list = $("#plan-versions");
  if (!list || !plan) return;
  const versions = plan.versions || [];
  list.innerHTML = versions.length
    ? versions.slice().reverse().map((version) => `
      <div class="plan-version">
        <div class="meta"><strong>Version ${version.version} ${version.accepted ? "· accepted" : ""}</strong><small>${escapeHtml(version.reason || "")} · ${new Date(version.created_at).toLocaleString()}</small></div>
        <div class="resource-actions">
          ${version.accepted ? "" : `<button class="button ghost" data-accept-version="${version.version}">Accept</button>`}
          <button class="button ghost" data-restore-version="${version.version}">Restore</button>
        </div>
      </div>`).join("")
    : '<div class="empty-state">No plan versions yet.</div>';
  $$("[data-accept-version]", list).forEach((button) => button.addEventListener("click", () => acceptPlanVersion(Number(button.dataset.acceptVersion))));
  $$("[data-restore-version]", list).forEach((button) => button.addEventListener("click", () => restorePlanVersion(Number(button.dataset.restoreVersion))));
}

async function acceptCurrentPlan() {
  if (!state.planId) return;
  try {
    const full = await api(`/api/plans/${state.planId}?profile_id=${state.profileId}`, { profile: false });
    const latest = full.plan.versions[full.plan.versions.length - 1];
    await api(`/api/plans/${state.planId}/accept`, { method: "POST", body: { version: latest.version } });
    toast("Plan accepted.", "success");
    await loadPlanHistory();
    await loadProgress(state.progressPeriod);
  } catch (error) {
    toast(error.message, "error");
  }
}

async function acceptPlanVersion(version) {
  try {
    await api(`/api/plans/${state.planId}/accept`, { method: "POST", body: { version } });
    toast(`Version ${version} accepted.`, "success");
    await loadPlanHistory();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function rejectCurrentPlan() {
  if (!state.planId) return;
  try {
    await api(`/api/plans/${state.planId}/reject`, { method: "POST", body: {} });
    toast("Draft rejected. The previous accepted plan is unchanged.", "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function restorePlanVersion(version) {
  if (!state.planId) return;
  if (!window.confirm(`Restore plan version ${version}? This creates a new version and keeps the old one.`)) return;
  try {
    await api(`/api/plans/${state.planId}/restore`, { method: "POST", body: { version, confirm: true } });
    state.plan = null;
    toast(`Version ${version} restored.`, "success");
    await loadPlanHistory();
    await loadProgress(state.progressPeriod);
  } catch (error) {
    toast(error.message, "error");
  }
}

function downloadPlan() {
  if (!state.planId) return toast("Accept or generate a plan first.", "error");
  window.location.href = `/api/plans/${state.planId}/download?profile_id=${state.profileId}`;
}

/* ------------------------------------------------------------------ */
/* Notes and drawing                                                  */
/* ------------------------------------------------------------------ */

let canvasContext = null;
let canvasDrawing = false;
let currentStroke = null;

function setupCanvas() {
  const canvas = $("#drawing-canvas");
  if (!canvas) return;
  canvasContext = canvas.getContext("2d");
  canvasContext.lineCap = "round";
  canvasContext.lineJoin = "round";
  const position = (event) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (canvas.width / rect.width),
      y: (event.clientY - rect.top) * (canvas.height / rect.height),
    };
  };
  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    canvasDrawing = true;
    currentStroke = { color: $("#draw-colour").value, width: Number($("#draw-width").value), points: [position(event)] };
    state.drawing.strokes.push(currentStroke);
    drawStrokes();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!canvasDrawing || !currentStroke) return;
    currentStroke.points.push(position(event));
    drawStrokes();
  });
  const stop = () => { canvasDrawing = false; currentStroke = null; };
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointerleave", stop);
  $("#clear-canvas-button").addEventListener("click", () => {
    state.drawing.strokes = [];
    drawStrokes();
  });
  $("#save-note-button").addEventListener("click", saveNote);
}

function drawStrokes() {
  const canvas = $("#drawing-canvas");
  if (!canvas || !canvasContext) return;
  canvasContext.clearRect(0, 0, canvas.width, canvas.height);
  canvasContext.fillStyle = "#ffffff";
  canvasContext.fillRect(0, 0, canvas.width, canvas.height);
  for (const stroke of state.drawing.strokes) {
    if (!stroke.points.length) continue;
    canvasContext.strokeStyle = stroke.color || "#1d4ed8";
    canvasContext.lineWidth = stroke.width || 3;
    canvasContext.beginPath();
    canvasContext.moveTo(stroke.points[0].x, stroke.points[0].y);
    for (const point of stroke.points.slice(1)) canvasContext.lineTo(point.x, point.y);
    canvasContext.stroke();
  }
}

async function saveNote() {
  if (!requireProfile()) return;
  const target = $("#note-target").value.trim() || "study";
  const canvas = $("#drawing-canvas");
  try {
    await api("/api/notes", {
      method: "POST",
      body: {
        target_type: "study",
        target_id: target,
        text: $("#note-text").value,
        strokes: state.drawing.strokes,
        image_data: canvas.toDataURL("image/png"),
      },
    });
    toast("Note saved.", "success");
    await loadNotes();
    await loadProgress(state.progressPeriod);
  } catch (error) {
    toast(error.message, "error");
  }
}

async function loadNotes() {
  if (!state.profileId) return;
  try {
    const data = await api(`/api/notes?profile_id=${state.profileId}`, { profile: false });
    state.notes = data.notes || [];
    renderNotes();
    renderProgressNotes();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderNotes() {
  const list = $("#notes-list");
  if (!list) return;
  list.innerHTML = state.notes.length
    ? state.notes.map((note) => `
      <button class="note-item" data-note-id="${note.id}">
        <strong>${escapeHtml(note.target_id)}</strong>
        <p>${escapeHtml((note.text || "").slice(0, 140)) || "Drawing note"}</p>
        <small>${new Date(note.updated_at).toLocaleString()}</small>
      </button>`).join("")
    : '<div class="empty-state">No notes saved.</div>';
  $$("[data-note-id]", list).forEach((button) => button.addEventListener("click", () => {
    const note = state.notes.find((item) => String(item.id) === button.dataset.noteId);
    if (!note) return;
    $("#note-target").value = note.target_id;
    $("#note-text").value = note.text || "";
    state.drawing.strokes = note.strokes || [];
    drawStrokes();
  }));
}

function renderProgressNotes() {
  const list = $("#progress-notes");
  if (!list) return;
  list.innerHTML = state.notes.length
    ? state.notes.slice(0, 5).map((note) => `<div class="note-item"><strong>${escapeHtml(note.target_id)}</strong><p>${escapeHtml((note.text || "").slice(0, 180)) || "Drawing note"}</p></div>`).join("")
    : '<div class="empty-state">No notes yet.</div>';
}

/* ------------------------------------------------------------------ */
/* Progress                                                           */
/* ------------------------------------------------------------------ */

async function loadProgress(period = "all") {
  if (!state.profileId) return;
  state.progressPeriod = period;
  $$("#page-progress .segmented button").forEach((button) => button.classList.toggle("is-active", button.dataset.period === period));
  try {
    const data = await api(`/api/progress?period=${period}&profile_id=${state.profileId}`, { profile: false });
    state.progress = data.progress;
    renderProgress(data.progress);
    renderDashboardStats(data.progress);
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderDashboardStats(progress) {
  if (!progress) return;
  $("#home-xp").textContent = progress.total_xp ?? 0;
  $("#home-coins").textContent = progress.total_coins ?? 0;
  $("#home-accuracy").textContent = progress.accuracy == null ? "—" : `${Math.round(progress.accuracy * 100)}%`;
}

function renderProgress(progress) {
  $("#progress-accuracy").textContent = progress.accuracy == null ? "—" : `${Math.round(progress.accuracy * 100)}%`;
  $("#progress-attempts").textContent = progress.attempts ?? 0;
  $("#progress-minutes").textContent = progress.active_minutes ?? 0;
  $("#progress-xp").textContent = progress.xp ?? 0;
  $("#progress-adherence").textContent = progress.adherence_label || "—";
  $("#progress-period-label").textContent = progress.period === "all" ? "All time" : progress.period;
  const topics = $("#progress-topics");
  topics.innerHTML = (progress.topics || []).length
    ? progress.topics.map((topic) => `
      <div class="topic-row">
        <div><strong>${escapeHtml(topic.topic)}</strong><br><small>${topic.attempts} checked · ${topic.active_seconds ? Math.round(topic.active_seconds / 60) : 0} min</small></div>
        <span>${topic.accuracy == null ? "—" : `${Math.round(topic.accuracy * 100)}%`}</span>
        <span class="pill soft">${escapeHtml(topic.level || "Unknown")}</span>
      </div>`).join("")
    : '<div class="empty-state">No topic evidence yet. Complete a checked quiz.</div>';
  drawProgressChart(progress.chart || []);
  renderProgressNotes();
}

function drawProgressChart(chart) {
  const canvas = $("#progress-chart");
  if (!canvas) return;
  const context = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 720;
  const height = 260;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const styles = getComputedStyle(document.body);
  const text = styles.getPropertyValue("--muted").trim() || "#46556b";
  const border = styles.getPropertyValue("--border").trim() || "#c9d4e2";
  const accent = styles.getPropertyValue("--accent").trim() || "#1d4ed8";
  context.font = "12px Segoe UI, sans-serif";
  context.fillStyle = text;
  if (!chart.length) {
    context.fillText("No activity recorded yet.", 16, 40);
    return;
  }
  const max = Math.max(1, ...chart.map((item) => Math.max(item.minutes || 0, (item.xp || 0) / 10)));
  const left = 14;
  const bottom = height - 28;
  const plotWidth = width - 28;
  const barWidth = Math.max(4, Math.min(28, (plotWidth / chart.length) * 0.6));
  context.strokeStyle = border;
  context.beginPath();
  context.moveTo(left, bottom);
  context.lineTo(width - 10, bottom);
  context.stroke();
  chart.forEach((item, index) => {
    const x = left + (index + 0.5) * (plotWidth / chart.length) - barWidth / 2;
    const heightValue = Math.max(2, ((item.minutes || 0) / max) * (bottom - 20));
    context.fillStyle = accent;
    context.fillRect(x, bottom - heightValue, barWidth, heightValue);
    if (chart.length <= 14 || index % Math.ceil(chart.length / 10) === 0) {
      context.save();
      context.translate(x + barWidth / 2, bottom + 6);
      context.rotate(-0.45);
      context.fillStyle = text;
      context.fillText(String(item.date).slice(5), 0, 8);
      context.restore();
    }
  });
  context.fillStyle = text;
  context.fillText("Active minutes per day", left, 14);
  $("#progress-chart-legend").innerHTML = `<span><span class="legend-dot" style="background:${accent}"></span>Active minutes</span><span>Total XP: ${state.progress?.total_xp ?? 0}</span>`;
}

/* ------------------------------------------------------------------ */
/* Store                                                              */
/* ------------------------------------------------------------------ */

async function loadStore() {
  if (!state.profileId) return;
  try {
    const data = await api(`/api/store?profile_id=${state.profileId}`, { profile: false });
    state.store = data;
    $("#store-coins").textContent = data.balance.coins ?? 0;
    renderStore();
    renderProfile();
  } catch (error) {
    toast(error.message, "error");
  }
}

const ITEM_ART = {
  "student-default": "🧑‍🎓", "student-glasses": "🤓", "student-cap": "🎓",
  "ai-default": "🤖", "ai-owl": "🦉", "ai-robot": "🤖",
  "theme-midnight": "🌙", "theme-ocean": "🌊",
  "background-grid": "▦", "background-paper": "📄",
};

function renderStore() {
  const grid = $("#store-grid");
  if (!grid) return;
  const owned = new Map((state.store.balance.owned || []).map((item) => [item.item_key, item]));
  const items = state.store.items.filter((item) => state.storeCategory === "all" || item.category === state.storeCategory);
  grid.innerHTML = items.map((item) => {
    const ownedItem = owned.get(item.item_key);
    const equipped = Boolean(ownedItem?.equipped);
    return `<article class="store-card ${ownedItem ? "is-owned" : ""} ${equipped ? "is-equipped" : ""}">
      <div class="item-art" aria-hidden="true">${ITEM_ART[item.item_key] || "✦"}</div>
      <h3>${escapeHtml(item.name)}</h3>
      <p>${escapeHtml(item.description)}</p>
      <div class="price">${item.price ? `${item.price} coins` : "Free"}</div>
      ${ownedItem
        ? `<button class="button ${equipped ? "secondary" : "ghost"}" data-equip="${escapeHtml(item.item_key)}" ${equipped ? "disabled" : ""}>${equipped ? "Equipped" : "Equip"}</button>`
        : `<button class="button primary" data-buy="${escapeHtml(item.item_key)}">Buy</button>`}
    </article>`;
  }).join("") || '<div class="empty-state">No items in this category.</div>';
  $$("[data-buy]", grid).forEach((button) => button.addEventListener("click", () => buyItem(button.dataset.buy)));
  $$("[data-equip]", grid).forEach((button) => button.addEventListener("click", () => equipItem(button.dataset.equip)));
}

async function buyItem(itemKey) {
  try {
    const data = await api("/api/store/buy", { method: "POST", body: { item_key: itemKey } });
    toast(`Purchased ${data.item.name}.`, "success");
    await loadStore();
    await loadProgress(state.progressPeriod);
  } catch (error) {
    toast(error.message, "error");
  }
}

async function equipItem(itemKey) {
  try {
    await api("/api/store/equip", { method: "POST", body: { item_key: itemKey } });
    toast("Item equipped.", "success");
    await loadStore();
    await loadSettingsPage();
    applyAppearance();
  } catch (error) {
    toast(error.message, "error");
  }
}

/* ------------------------------------------------------------------ */
/* Models, help and onboarding                                        */
/* ------------------------------------------------------------------ */

async function loadModelStatus() {
  try {
    const data = await api("/api/models/status", { profile: false });
    state.modelStatus = data.runtime;
    state.ollamaStatus = data.ollama;
    renderModelBadge();
    const box = $("#model-status");
    if (box) {
      const runtime = data.runtime || {};
      const ollama = data.ollama || {};
      const clf = runtime.performance_model || {};
      const clfMissing = (runtime.missing_files || []).includes("performance_model.joblib");
      box.innerHTML = `<strong>${escapeHtml(ollama.label || "SmolLM2 360M")}</strong>
        <p>Ollama: ${escapeHtml(ollama.model || "smollm2:360m")} · ${escapeHtml(ollama.base_url || "http://127.0.0.1:11434")}</p>
        <p>Context ${ollama.num_ctx ?? 2048} · max tokens ${ollama.max_tokens ?? 200} · temperature ${ollama.temperature ?? 0.2}</p>
        <p>Available: ${ollama.available ? "yes" : "no"} · model present: ${ollama.model_present ? "yes" : "no"}${ollama.error ? ` · ${escapeHtml(ollama.error)}` : ""}</p>
        <p>${clfMissing ? "Classifier: rule-based fallback (trained model absent)" : `Classifier: ${escapeHtml(clf.version || "loaded")}`}</p>
        <p class="fine-print">Loaded: ${runtime.loaded ? "yes" : "no (cold start on first request)"}</p>`;
    }
  } catch (error) {
    const box = $("#model-status");
    if (box) box.textContent = error.message;
  }
}

async function warmupModel() {
  try {
    const data = await api("/api/models/warmup", { method: "POST", body: {} });
    const ollama = data.result?.ollama;
    toast(ollama?.warm ? "SmolLM2 360M warmed up." : (ollama?.error || "SmolLM2 360M warmup failed."), "success");
    await loadModelStatus();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function unloadModel() {
  try {
    const data = await api("/api/models/unload", { method: "POST", body: {} });
    const ollama = data.result?.ollama;
    toast(ollama?.unloaded ? "SmolLM2 360M released from memory." : (ollama?.error || "SmolLM2 360M could not be released."), "success");
    await loadModelStatus();
  } catch (error) {
    toast(error.message, "error");
  }
}

function openHelp() {
  $("#help-drawer").hidden = false;
  $("#help-input").focus();
}

function closeHelp() {
  $("#help-drawer").hidden = true;
  $("#help-button").focus();
}

async function sendHelp(event) {
  event.preventDefault();
  const input = $("#help-input");
  const message = input.value.trim();
  if (!message) return;
  const messages = $("#help-messages");
  messages.insertAdjacentHTML("beforeend", `<div class="message user"><p>${escapeHtml(message)}</p></div>`);
  input.value = "";
  try {
    const data = await api("/api/help", { method: "POST", body: { message, page: $(".page.is-active")?.id || "home" } });
    messages.insertAdjacentHTML("beforeend", `<div class="message assistant"><p>${escapeHtml(data.answer)}</p>${data.action ? `<button class="button ghost" id="help-action-button">Take me there</button>` : ""}</div>`);
    if (data.action) {
      $("#help-action-button").addEventListener("click", () => {
        closeHelp();
        go(data.action.page || "home", { tab: data.action.tab, focus: data.action.focus });
        if (data.action.action === "new-chat") newSession();
      });
    }
    messages.scrollTop = messages.scrollHeight;
  } catch (error) {
    toast(error.message, "error");
  }
}

function showOnboarding() {
  state.onboardingStep = 0;
  $("#onboarding-modal").hidden = false;
  renderOnboarding();
}

function renderOnboarding() {
  const step = ONBOARDING_STEPS[state.onboardingStep];
  $("#onboarding-title").textContent = step.title;
  $("#onboarding-content").innerHTML = step.html;
  $("#onboarding-step-label").textContent = `Step ${state.onboardingStep + 1} of ${ONBOARDING_STEPS.length}`;
  $("#onboarding-bar").style.width = `${((state.onboardingStep + 1) / ONBOARDING_STEPS.length) * 100}%`;
  $("#onboarding-next").textContent = state.onboardingStep === ONBOARDING_STEPS.length - 1 ? "Start studying" : "Next";
  if (state.onboardingStep === 0) {
    const input = $("#onboarding-name");
    if (input) input.value = state.profile?.name || "";
  }
}

async function nextOnboarding() {
  const button = $("#onboarding-next");
  setBusy(button, true, "Saving…");
  try {
    if (!state.profileId) {
      const name = ($("#onboarding-name")?.value || "Student").trim() || "Student";
      await createProfile(name);
    }
    if (state.onboardingStep < ONBOARDING_STEPS.length - 1) {
      state.onboardingStep += 1;
      renderOnboarding();
    } else {
      localStorage.setItem(STORAGE_ONBOARDED, "1");
      $("#onboarding-modal").hidden = true;
      toast("You are ready. Add a source or start a chat.", "success");
      go("home");
    }
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(button, false);
  }
}

async function skipOnboarding() {
  try {
    if (!state.profileId) await createProfile("Student");
    localStorage.setItem(STORAGE_ONBOARDED, "1");
    $("#onboarding-modal").hidden = true;
    go("home");
  } catch (error) {
    toast(error.message, "error");
  }
}

/* ------------------------------------------------------------------ */
/* Inactivity and activity logging                                    */
/* ------------------------------------------------------------------ */

let lastActivitySent = 0;
function markActivity() {
  if (!state.profileId) return;
  const now = Date.now();
  if (now - lastActivitySent < 15000) return;
  lastActivitySent = now;
  fetch("/api/activity", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_id: state.profileId }),
  }).catch(() => {});
}

function pauseMedia() {
  $$("audio, video").forEach((media) => {
    try { media.pause(); } catch { /* ignore */ }
  });
  toast("Media paused after 10 minutes of inactivity. Your saved work is safe.", "info", 7000);
}

async function checkInactivity() {
  if (!state.profileId) return;
  try {
    const data = await api(`/api/inactivity?profile_id=${state.profileId}`, { profile: false });
    if ((data.actions || []).includes("media_stopped")) pauseMedia();
    if ((data.actions || []).length) {
      loadModelStatus();
      const actionText = data.actions.join(", ").replaceAll("_", " ");
      toast(`Inactivity: ${actionText}.`, "info", 5000);
    }
  } catch {
    /* Keep the website usable if the timer check fails. */
  }
}

function setupInactivity() {
  ["click", "keydown", "pointerdown", "scroll", "touchstart"].forEach((eventName) => {
    window.addEventListener(eventName, markActivity, { passive: true });
  });
  window.setInterval(checkInactivity, 30000);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) markActivity();
  });
}

/* ------------------------------------------------------------------ */
/* Event wiring and startup                                           */
/* ------------------------------------------------------------------ */

function bindEvents() {
  bindNavigation();
  $("#home-prompt-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const prompt = $("#home-prompt").value.trim();
    if (!prompt) return;
    state.pendingHomePrompt = prompt;
    go("study", { tab: "chat" });
    $("#chat-input").value = prompt;
    $("#chat-input").focus();
  });
  $("#new-session-button").addEventListener("click", newSession);
  $("#save-draft-button").addEventListener("click", async () => {
    if (!state.currentSessionId) return toast("Send the first message to create a session.", "error");
    try {
      await api(`/api/sessions/${state.currentSessionId}/draft`, { method: "POST", body: { draft: $("#chat-input").value } });
      toast("Draft saved.", "success");
    } catch (error) { toast(error.message, "error"); }
  });
  $("#chat-input").addEventListener("input", scheduleDraftSave);
  $("#chat-form").addEventListener("submit", sendChat);
  $("#add-topic-button").addEventListener("click", addTopic);
  $("#upload-form").addEventListener("submit", submitUpload);
  $("#research-form").addEventListener("submit", submitResearch);
  $("#load-quiz-button").addEventListener("click", () => loadQuizForPractice(false));
  $("#start-mock-button").addEventListener("click", () => loadQuizForPractice(true));
  $("#start-flashcards-button").addEventListener("click", startFlashcards);
  $("#classify-button").addEventListener("click", classifyPerformance);
  $("#improvement-button").addEventListener("click", loadImprovement);
  $("#generate-plan-button").addEventListener("click", generatePlan);
  $("#download-plan-button").addEventListener("click", downloadPlan);
  $("#save-note-button").addEventListener("click", saveNote);
  $$("#page-progress .segmented button").forEach((button) => button.addEventListener("click", () => loadProgress(button.dataset.period)));
  $$("#store-filters button").forEach((button) => button.addEventListener("click", () => {
    state.storeCategory = button.dataset.category;
    $$("#store-filters button").forEach((item) => item.classList.toggle("is-active", item === button));
    renderStore();
  }));
  $("#profile-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = $("#new-profile-name").value.trim();
    if (!name) return toast("Enter a profile name.", "error");
    try {
      await createProfile(name);
      $("#new-profile-name").value = "";
      toast("Profile created.", "success");
    } catch (error) { toast(error.message, "error"); }
  });
  $("#profile-select").addEventListener("change", (event) => activateProfile(Number(event.target.value)));
  $("#refresh-profiles-button").addEventListener("click", async () => { await loadProfiles(); toast("Profiles refreshed.", "success"); });
  $("#save-settings-button").addEventListener("click", saveSettings);
  $("#save-api-key-button").addEventListener("click", saveApiKey);
  $("#test-api-button").addEventListener("click", testApi);
  $$('input[name="ai-mode"]').forEach((radio) => radio.addEventListener("change", () => {
    $("#api-settings").hidden = $('input[name="ai-mode"]:checked')?.value !== "api";
  }));
  $("#theme-select").addEventListener("change", () => {
    state.settings.theme = $("#theme-select").value;
    savePrefs({ theme: state.settings.theme });
    applyAppearance();
  });
  $("#contrast-select").addEventListener("change", () => {
    state.settings.contrast = $("#contrast-select").value;
    savePrefs({ contrast: state.settings.contrast });
    applyAppearance();
  });
  $("#reduce-motion").addEventListener("change", (event) => { savePrefs({ reduceMotion: event.target.checked }); applyAppearance(); });
  $("#large-text").addEventListener("change", (event) => { savePrefs({ largeText: event.target.checked }); applyAppearance(); });
  $("#warmup-model-button").addEventListener("click", warmupModel);
  $("#unload-model-button").addEventListener("click", unloadModel);
  $("#help-button").addEventListener("click", openHelp);
  $("#help-close").addEventListener("click", closeHelp);
  $("#help-form").addEventListener("submit", sendHelp);
  $("#onboarding-next").addEventListener("click", nextOnboarding);
  $("#onboarding-skip").addEventListener("click", skipOnboarding);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (!$("#help-drawer").hidden) closeHelp();
    }
  });
  window.setInterval(() => {
    if (state.profileId && (!state.modelStatus || !state.modelStatus.ready)) loadModelStatus();
  }, 60000);
}

async function init() {
  applyAppearance();
  bindEvents();
  setupCanvas();
  setupInactivity();
  drawStrokes();
  try {
    await loadModelStatus();
    await loadProfiles();
    const stored = Number(localStorage.getItem(STORAGE_PROFILE) || 0);
    const chosen = state.profiles.find((profile) => profile.id === stored) || state.profiles[0];
    if (chosen) {
      await activateProfile(chosen.id);
      if (!localStorage.getItem(STORAGE_ONBOARDED)) showOnboarding();
    } else {
      showOnboarding();
    }
    const hash = window.location.hash.replace("#", "");
    if (PAGE_TITLES[hash]) go(hash);
  } catch (error) {
    toast(error.message, "error", 8000);
  }
}

document.addEventListener("DOMContentLoaded", init);
