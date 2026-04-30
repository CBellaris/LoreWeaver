const state = {
  commands: {},
  currentJobId: null,
  eventSource: null,
  inspectorMode: "windows",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

document.addEventListener("DOMContentLoaded", async () => {
  bindNavigation();
  bindActions();
  await loadCommands();
  await loadOverview();
  await loadNeo4jStatus();
});

function bindNavigation() {
  $$(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      $$(".nav-item").forEach((item) => item.classList.remove("active"));
      $$(".view").forEach((view) => view.classList.remove("active"));
      button.classList.add("active");
      $(`#view-${button.dataset.view}`).classList.add("active");
    });
  });

}

function bindActions() {
  $("#refresh-overview").addEventListener("click", loadOverview);
  $("#run-command").addEventListener("click", runSelectedCommand);
  $("#cancel-job").addEventListener("click", cancelCurrentJob);
  $("#run-retrieve").addEventListener("click", runRetrievalWorkbench);
  $("#neo4j-refresh").addEventListener("click", loadNeo4jStatus);
  $("#neo4j-start").addEventListener("click", startNeo4j);
  $("#command-select").addEventListener("change", renderCommandFields);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text);
  }
  return response.json();
}

async function loadCommands() {
  state.commands = await api("/api/commands");
  const select = $("#command-select");
  select.innerHTML = Object.entries(state.commands)
    .map(([name, spec]) => `<option value="${escapeHtml(name)}">${escapeHtml(spec.label)}</option>`)
    .join("");
  select.value = "retrieve";
  renderCommandFields();
}

async function loadOverview() {
  const content = $("#overview-content");
  content.innerHTML = loading();
  try {
    const data = await api("/api/overview");
    $("#stage-line").textContent = `${data.project.name || "LoreWeaver"} ${data.project.stage || ""}`;
    const counts = data.counts || {};
    content.innerHTML = `
      <section class="panel">
        <div class="metric-grid">
          ${metric("Documents", counts.documents)}
          ${metric("Chapters", counts.chapters)}
          ${metric("Windows", counts.candidate_windows)}
          ${metric("Spans", counts.spans)}
          ${metric("Clusters", counts.center_span_clusters)}
          ${metric("Evidence", counts.evidence_packs)}
        </div>
      </section>
      <section class="panel">
        <div class="panel-title"><h3>Paths</h3></div>
        ${keyValueTable(data.paths)}
      </section>
      <section class="panel">
        <div class="panel-title"><h3>Environment</h3></div>
        <div class="pill-row">
          ${Object.entries(data.env || {})
            .map(([key, ok]) => `<span class="pill ${ok ? "good" : "bad"}">${escapeHtml(key)} ${ok ? "set" : "missing"}</span>`)
            .join("")}
        </div>
      </section>
      <section class="panel">
        <div class="panel-title"><h3>Recent Reports</h3></div>
        ${reportsTable(data.recent_reports || [])}
      </section>
    `;
  } catch (error) {
    content.innerHTML = errorBox(error);
  }
}

function renderCommandFields() {
  const command = $("#command-select").value;
  const spec = state.commands[command] || { fields: [] };
  $("#command-fields").innerHTML = spec.fields.map(fieldHtml).join("");
  $$("#command-fields input, #command-fields select").forEach((element) => {
    element.addEventListener("input", updateCliPreview);
    element.addEventListener("change", updateCliPreview);
  });
  seedCommandDefaults(command);
  updateCliPreview();
}

function fieldHtml(name) {
  if (isBooleanField(name)) {
    return `
      <label class="check-field">
        <input id="field-${name}" data-field="${name}" type="checkbox">
        <span>${escapeHtml(name.replaceAll("_", " "))}</span>
      </label>
    `;
  }
  if (name === "only") {
    return `
      <label class="field">
        <span>only</span>
        <select id="field-only" data-field="only">
          <option value="all">all</option>
          <option value="pending">pending</option>
          <option value="extracted">extracted</option>
        </select>
      </label>
    `;
  }
  const wide = ["question", "query", "source", "window_id", "window_range"].includes(name) ? " wide" : "";
  return `
    <label class="field${wide}">
      <span>${escapeHtml(name.replaceAll("_", " "))}</span>
      <input id="field-${name}" data-field="${name}" placeholder="${placeholderFor(name)}">
    </label>
  `;
}

function seedCommandDefaults(command) {
  if (command === "retrieve") {
    setField("question", "塞西尔家族为什么衰落？");
    setCheckbox("mock_embeddings", true);
    setCheckbox("mock_reranker", true);
  }
  if (command === "evidence" || command === "ask") {
    setField("question", "塞西尔家族为什么衰落？");
  }
  if (command === "search-vector" || command === "search-bm25") {
    setField("query", "塞西尔家族为什么衰落？");
  }
  if (command === "extract") {
    setCheckbox("mock", true);
    setField("limit", "3");
  }
  if (command === "index") {
    setCheckbox("mock_embeddings", true);
  }
}

function setField(name, value) {
  const element = $(`#field-${name}`);
  if (element && !element.value) element.value = value;
}

function setCheckbox(name, value) {
  const element = $(`#field-${name}`);
  if (element) element.checked = value;
}

function updateCliPreview() {
  const command = $("#command-select").value;
  const payload = collectCommandPayload();
  const args = Object.entries(payload).flatMap(([key, value]) => {
    const flag = `--${key.replaceAll("_", "-")}`;
    if (typeof value === "boolean") return value ? [flag] : [];
    return value === "" || value === null || value === undefined ? [] : [flag, quoteArg(String(value))];
  });
  const positional = [];
  if (payload.question) positional.push(quoteArg(payload.question));
  if (payload.query) positional.push(quoteArg(payload.query));
  const filteredArgs = args.filter((item) => !["--question", quoteArg(payload.question || ""), "--query", quoteArg(payload.query || "")].includes(item));
  $("#cli-preview").textContent = `python -m loreweaver.cli ${command} ${positional.concat(filteredArgs).join(" ")}`.trim();
}

function collectCommandPayload() {
  const payload = {};
  $$("#command-fields [data-field]").forEach((element) => {
    const name = element.dataset.field;
    if (element.type === "checkbox") {
      payload[name] = element.checked;
    } else if (element.value.trim()) {
      payload[name] = element.value.trim();
    }
  });
  return payload;
}

async function runSelectedCommand() {
  const command = $("#command-select").value;
  const payload = collectCommandPayload();
  attachOneRunApiKey(payload, "#runner-api-env", "#runner-api-key");
  $("#job-result").innerHTML = "";
  await startJob(command, payload, $("#job-result"));
}

async function runRetrievalWorkbench() {
  const payload = {
    question: $("#retrieve-question").value.trim(),
    document_id: $("#retrieve-document-id").value.trim(),
    mock_embeddings: $("#retrieve-mock-embeddings").checked,
    mock_reranker: $("#retrieve-mock-reranker").checked,
    no_reranker: $("#retrieve-no-reranker").checked,
  };
  attachOneRunApiKey(payload, "#retrieve-api-env", "#retrieve-api-key");
  Object.keys(payload).forEach((key) => {
    if (payload[key] === "") delete payload[key];
  });
  $("#retrieval-result").innerHTML = "";
  await startJob("retrieve", payload, $("#retrieval-result"));
}

async function startJob(command, payload, resultTarget) {
  closeEventSource();
  $("#event-log").innerHTML = "";
  $("#job-status").textContent = "starting";
  $("#cancel-job").disabled = false;
  try {
    const job = await api(`/api/jobs/${command}`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.currentJobId = job.job_id;
    $("#job-status").textContent = `${job.command} ${job.job_id}`;
    state.eventSource = new EventSource(`/api/jobs/${job.job_id}/events`);
    state.eventSource.onmessage = (message) => {
      const event = JSON.parse(message.data);
      appendEvent(event);
      if (event.event === "completed") {
        renderResult(event.payload.result, resultTarget);
      }
      if (event.event === "terminal") {
        $("#cancel-job").disabled = true;
        $("#job-status").textContent = `${command} ${event.payload.status}`;
        closeEventSource();
        loadOverview();
      }
    };
    state.eventSource.onerror = () => {
      $("#job-status").textContent = "event stream disconnected";
      closeEventSource();
    };
  } catch (error) {
    resultTarget.innerHTML = errorBox(error);
    $("#job-status").textContent = "failed to start";
    $("#cancel-job").disabled = true;
  }
}

function attachOneRunApiKey(payload, envSelector, keySelector) {
  const envName = $(envSelector).value;
  const apiKey = $(keySelector).value;
  if (envName && apiKey) {
    payload._env = { [envName]: apiKey };
  }
}

async function cancelCurrentJob() {
  if (!state.currentJobId) return;
  await api(`/api/jobs/${state.currentJobId}/cancel`, { method: "POST", body: "{}" });
}

function appendEvent(event) {
  const log = $("#event-log");
  const item = document.createElement("div");
  item.className = "event";
  item.innerHTML = `
    <strong>${escapeHtml(event.event)}</strong>
    <code>${escapeHtml(JSON.stringify(event.payload || {}))}</code>
  `;
  log.appendChild(item);
  log.scrollTop = log.scrollHeight;
}

function closeEventSource() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
}

function renderResult(result, target) {
  if (!target) return;
  const parts = [];
  if (result && result.retrieval && result.top_results) {
    parts.push(renderRetrievalReport(result));
  }
  if (result && result.evidence_pack) {
    parts.push(renderEvidenceReport(result));
  }
  if (result && result.answer) {
    parts.push(renderAnswerReport(result));
  }
  parts.push(`<pre class="json-view">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`);
  target.innerHTML = parts.join("");
}

function renderRetrievalReport(report) {
  const retrieval = report.retrieval || {};
  const graph = retrieval.graph || {};
  const vector = retrieval.vector || {};
  const bm25 = retrieval.bm25 || {};
  const union = retrieval.union || {};
  return `
    <div class="retrieval-section">
      <div class="metric-grid">
        ${metric("Graph Hits", graph.count || 0)}
        ${metric("Vector Hits", vector.count || 0)}
        ${metric("BM25 Hits", bm25.count || 0)}
        ${metric("Union", union.candidate_count || 0)}
        ${metric("Multi Source", union.multi_source_count || 0)}
      </div>
    </div>
    <div class="retrieval-section">
      <h4>Graph Clusters</h4>
      ${simpleTable(graph.clusters || [], ["cluster_id", "cluster_name", "cluster_type", "score"])}
    </div>
    <div class="retrieval-section">
      <h4>Top Results</h4>
      ${simpleTable(report.top_results || [], [
        "rank",
        "span_id",
        "rerank_score",
        "fused_score",
        "sources",
        "source_scores",
        "normalized_scores",
        "micro_topic",
        "span_start_idx",
        "span_end_idx",
      ])}
    </div>
    <div class="retrieval-section">
      <h4>Union Candidates</h4>
      ${simpleTable((report.candidates || []).slice(0, 40), [
        "span_id",
        "fused_score",
        "sources",
        "source_scores",
        "normalized_scores",
        "cluster_ids",
        "micro_topic",
      ])}
    </div>
  `;
}

function renderEvidenceReport(report) {
  const pack = report.evidence_pack || {};
  const blocks = pack.evidence_blocks || [];
  return `
    <div class="retrieval-section">
      <h4>Evidence Blocks</h4>
      ${simpleTable(blocks, [
        "citation_id",
        "chapter_title",
        "start_idx",
        "end_idx",
        "source_span_ids",
        "retrieval_sources",
        "rerank_score",
        "text",
      ])}
    </div>
  `;
}

function renderAnswerReport(report) {
  return `
    <div class="retrieval-section">
      <h4>Answer</h4>
      <div class="status-line">${escapeHtml((report.answer_validation || {}).ok ? "citations ok" : "citation validation failed")}</div>
      <div class="panel text-preview">${escapeHtml(report.answer || "")}</div>
    </div>
  `;
}

async function loadNeo4jStatus() {
  const target = $("#neo4j-status");
  target.innerHTML = loading();
  try {
    const status = await api("/api/neo4j/status");
    target.innerHTML = neo4jHtml(status);
  } catch (error) {
    target.innerHTML = errorBox(error);
  }
}

async function startNeo4j() {
  const button = $("#neo4j-start");
  button.disabled = true;
  try {
    const status = await api("/api/neo4j/start", { method: "POST", body: "{}" });
    $("#neo4j-status").innerHTML = neo4jHtml(status);
    window.open(status.url, "_blank", "noopener,noreferrer");
  } catch (error) {
    $("#neo4j-status").innerHTML = errorBox(error);
  } finally {
    button.disabled = false;
  }
}

function neo4jHtml(status) {
  return `
    <div class="metric-grid">
      ${metric("Docker", status.docker_available ? "ok" : "missing")}
      ${metric("Container", status.exists ? "exists" : "new")}
      ${metric("Running", status.running ? "yes" : "no")}
    </div>
    <div class="retrieval-section">
      ${keyValueTable({
        container: status.container,
        status: status.status || "not created",
        url: status.url,
        bolt_url: status.bolt_url,
        username: status.username,
        password: status.password,
        error: status.error || "",
      })}
    </div>
  `;
}

function reportsTable(reports) {
  return simpleTable(reports, ["kind", "name", "mtime", "bytes"]);
}

function simpleTable(rows, columns, options = {}) {
  if (!rows.length) return `<div class="status-line">no rows</div>`;
  return `
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              ${columns.map((column) => {
                const value = row[column];
                const rendered = renderCell(value);
                if (options.clickable === column) {
                  return `<td><button class="button" data-detail="${options.onClick}" data-id="${escapeHtml(String(value || ""))}">${rendered}</button></td>`;
                }
                return `<td>${rendered}</td>`;
              }).join("")}
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderCell(value) {
  if (Array.isArray(value)) {
    return `<div class="pill-row">${value.slice(0, 8).map((item) => `<span class="pill">${escapeHtml(String(item))}</span>`).join("")}</div>`;
  }
  if (value && typeof value === "object") {
    return `<code>${escapeHtml(JSON.stringify(value))}</code>`;
  }
  const text = value === null || value === undefined ? "" : String(value);
  const short = text.length > 180 ? `${text.slice(0, 177)}...` : text;
  return `<span class="text-preview">${escapeHtml(short)}</span>`;
}

function keyValueTable(object) {
  const rows = Object.entries(object || {}).map(([key, value]) => ({ key, value }));
  return simpleTable(rows, ["key", "value"]);
}

function metric(label, value) {
  return `
    <div class="metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value ?? 0))}</strong>
    </div>
  `;
}

function loading() {
  return `<div class="status-line">loading</div>`;
}

function errorBox(error) {
  return `<div class="status-line">${escapeHtml(error.message || String(error))}</div>`;
}

function isBooleanField(name) {
  return [
    "by_chapter",
    "list_windows",
    "mock",
    "batch",
    "batch_wait",
    "no_progress",
    "mock_embeddings",
    "mock_reranker",
    "no_reranker",
    "mock_answer",
    "sync_neo4j",
    "no_neo4j",
    "no_embeddings",
    "list",
  ].includes(name);
}

function placeholderFor(name) {
  const values = {
    document_id: "latest",
    limit: "10",
    offset: "0",
    top_k: "5",
    top_salience: "30",
    window_range: "21-40",
    batch_poll_interval: "30",
  };
  return values[name] || "";
}

function quoteArg(value) {
  if (!value) return "";
  if (/^[A-Za-z0-9_./:=,-]+$/.test(value)) return value;
  return `"${value.replaceAll('"', '\\"')}"`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
