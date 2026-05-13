const state = {
  commands: {},
  currentJobId: null,
  eventSource: null,
  inspectorMode: "windows",
  spanReviewWindows: [],
  selectedReviewWindowId: null,
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
      if (button.dataset.view === "span-review" && state.spanReviewWindows.length === 0) {
        loadSpanReview();
      }
    });
  });

}

function bindActions() {
  $("#refresh-overview").addEventListener("click", loadOverview);
  $("#run-command").addEventListener("click", runSelectedCommand);
  $("#cancel-job").addEventListener("click", cancelCurrentJob);
  $("#run-retrieve").addEventListener("click", runRetrievalWorkbench);
  $("#span-review-refresh").addEventListener("click", loadSpanReview);
  $("#neo4j-refresh").addEventListener("click", loadNeo4jStatus);
  $("#neo4j-start").addEventListener("click", startNeo4j);
  $("#command-select").addEventListener("change", renderCommandFields);
  $("#span-review-min-gap").addEventListener("change", loadSpanReview);
  $("#span-review-limit").addEventListener("change", loadSpanReview);
  $("#span-review-window-range").addEventListener("change", loadSpanReview);
  $("#span-review-with-spans").addEventListener("change", loadSpanReview);
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
  if (name === "profile") {
    return `
      <label class="field">
        <span>profile</span>
        <select id="field-profile" data-field="profile">
          <option value="broad">broad</option>
          <option value="pinpoint">pinpoint</option>
          <option value="mixed">mixed</option>
        </select>
      </label>
    `;
  }
  const wide = [
    "question",
    "query",
    "source",
    "window_id",
    "window_range",
    "corpus",
    "questions",
    "predictions",
    "output",
  ].includes(name) ? " wide" : "";
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
  if (command === "eval-build-corpus") {
    setField("chapter_start", "1");
    setField("chapter_end", "100");
  }
  if (command === "eval-generate") {
    setField("corpus", "data/eval/corpora/doc_59331b17113e_ch001_100.json");
    setField("question_count", "50");
    setField("max_output_tokens", "384000");
  }
  if (command === "eval-run") {
    setField("questions", "data/eval/question_sets/doc_59331b17113e_ch001_100_broad_v001.jsonl");
    setCheckbox("no_reranker", true);
  }
  if (command === "eval-report") {
    setField("predictions", "data/eval/runs/<run_id>_predictions.jsonl");
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
  const cli = cliCommandParts(command, payload);
  const positionalKeys = new Set(cli.positionals);
  const args = Object.entries(payload).flatMap(([key, value]) => {
    if (positionalKeys.has(key)) return [];
    const flag = `--${key.replaceAll("_", "-")}`;
    if (typeof value === "boolean") return value ? [flag] : [];
    return value === "" || value === null || value === undefined ? [] : [flag, quoteArg(String(value))];
  });
  const positional = cli.positionals
    .map((key) => payload[key])
    .filter((value) => value !== "" && value !== null && value !== undefined)
    .map((value) => quoteArg(String(value)));
  $("#cli-preview").textContent = `python -m loreweaver.cli ${cli.command} ${positional.concat(args).join(" ")}`.trim();
}

function cliCommandParts(command, payload) {
  const evalCommands = {
    "eval-build-corpus": { command: "eval build-corpus", positionals: [] },
    "eval-generate": { command: "eval generate", positionals: ["corpus"] },
    "eval-run": { command: "eval run", positionals: ["questions"] },
    "eval-report": { command: "eval report", positionals: ["predictions"] },
  };
  if (evalCommands[command]) return evalCommands[command];
  if (payload.question) return { command, positionals: ["question"] };
  if (payload.query) return { command, positionals: ["query"] };
  return { command, positionals: [] };
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

async function loadSpanReview() {
  const list = $("#span-review-list");
  const summary = $("#span-review-summary");
  list.innerHTML = loading();
  summary.innerHTML = "";
  const params = new URLSearchParams();
  const documentId = $("#span-review-document-id").value.trim();
  const windowRange = $("#span-review-window-range").value.trim();
  const withSpansOnly = $("#span-review-with-spans").checked;
  const minGap = $("#span-review-min-gap").value.trim();
  const limit = $("#span-review-limit").value.trim();
  if (documentId) params.set("document_id", documentId);
  if (windowRange) params.set("window_range", windowRange);
  if (withSpansOnly) params.set("with_spans_only", "true");
  if (minGap) params.set("min_gap_chars", minGap);
  if (limit) params.set("limit", limit);
  try {
    const data = await api(`/api/span-review?${params.toString()}`);
    state.spanReviewWindows = data.windows || [];
    state.selectedReviewWindowId = null;
    summary.innerHTML = renderSpanReviewSummary(data.summary || {});
    list.innerHTML = renderSpanReviewList(state.spanReviewWindows);
    $$("#span-review-list [data-window-id]").forEach((button) => {
      button.addEventListener("click", () => loadSpanReviewWindow(button.dataset.windowId));
    });
    $("#span-review-detail").innerHTML = state.spanReviewWindows.length
      ? "Select a window to inspect coverage."
      : "No windows match the current filters.";
  } catch (error) {
    list.innerHTML = errorBox(error);
  }
}

function renderSpanReviewSummary(summary) {
  return `
    <div class="metric-grid compact-metrics">
      ${metric("Windows", summary.window_count || 0)}
      ${metric("Coverage", `${percent(summary.coverage_ratio)}%`)}
      ${metric("Max Gap", summary.max_gap_chars || 0)}
      ${metric("Failed", summary.failed_window_count || 0)}
    </div>
  `;
}

function renderSpanReviewList(windows) {
  if (!windows.length) return `<div class="status-line">no windows</div>`;
  return windows.map((item) => `
    <button class="list-item review-window-item" data-window-id="${escapeHtml(item.window_id)}">
      <strong>#${escapeHtml(item.global_window_index ?? "?")} ${escapeHtml(item.window_id)}</strong>
      <span>${escapeHtml(item.chapter_title || item.chapter_id)} · window ${escapeHtml(item.window_index)}</span>
      <div class="review-meter" title="${escapeHtml(percent(item.coverage_ratio))}% covered">
        <div style="width: ${escapeHtml(percent(item.coverage_ratio))}%"></div>
      </div>
      <span>${escapeHtml(item.uncovered_chars)} uncovered · max gap ${escapeHtml(item.max_gap_chars)} · ${escapeHtml(item.failed_count)} failed</span>
      <div class="pill-row">${(item.hint_tags || []).map(tagPill).join("")}</div>
    </button>
  `).join("");
}

async function loadSpanReviewWindow(windowId) {
  state.selectedReviewWindowId = windowId;
  $$("#span-review-list [data-window-id]").forEach((button) => {
    button.classList.toggle("active", button.dataset.windowId === windowId);
  });
  const detail = $("#span-review-detail");
  detail.innerHTML = loading();
  try {
    const data = await api(`/api/span-review/${encodeURIComponent(windowId)}`);
    detail.innerHTML = renderSpanReviewDetail(data);
    bindSpanHoverCards(data);
  } catch (error) {
    detail.innerHTML = errorBox(error);
  }
}

function renderSpanReviewDetail(data) {
  const audit = data.audit || {};
  const spans = data.spans || [];
  const gaps = data.gaps || [];
  return `
    <div class="panel-title">
      <h3>${escapeHtml(audit.window_id || "")}</h3>
      <div class="pill-row">${(audit.hint_tags || []).map(tagPill).join("")}</div>
    </div>
    <div class="metric-grid compact-metrics">
      ${metric("Coverage", `${percent(audit.coverage_ratio)}%`)}
      ${metric("Spans", `${audit.located_count || 0}/${audit.span_count || 0}`)}
      ${metric("Gaps", audit.gap_count || 0)}
      ${metric("Max Gap", audit.max_gap_chars || 0)}
    </div>
    <div class="review-section">
      <h4>Text Overlay</h4>
      <div class="review-text">${renderCoverageSegments(data.segments || [])}</div>
    </div>
    <div class="review-section">
      <h4>Spans</h4>
      ${simpleTable(spans, [
        "span_index_in_window",
        "span_id",
        "locator_status",
        "span_type",
        "salience_score",
        "locator_confidence",
        "span_start_idx",
        "span_end_idx",
        "micro_summary",
        "failure_reasons",
      ])}
    </div>
    <div class="review-section">
      <h4>Gaps</h4>
      ${simpleTable(gaps, [
        "gap_index",
        "start_idx",
        "end_idx",
        "char_count",
        "left_span_id",
        "right_span_id",
        "text_preview",
      ])}
    </div>
  `;
}

function renderCoverageSegments(segments) {
  if (!segments.length) return "";
  return segments.map((segment) => {
    const spanIds = segment.span_ids || [];
    if (!spanIds.length) {
      return `<span class="coverage-gap" title="${segmentTitle(segment)}">${escapeHtml(segment.text)}</span>`;
    }
    const depth = Math.min(spanIds.length, 4);
    const color = hashIndex(spanIds[0], 6);
    return `<span class="coverage-span coverage-color-${color} coverage-depth-${depth}" data-span-ids="${escapeHtml(spanIds.join(","))}" title="${segmentTitle(segment)}">${escapeHtml(segment.text)}</span>`;
  }).join("");
}

function bindSpanHoverCards(data) {
  const spansById = new Map((data.spans || []).map((span) => [span.span_id, span]));
  $$(".coverage-span").forEach((element) => {
    const spanIds = (element.dataset.spanIds || "").split(",").filter(Boolean);
    const lines = spanIds.map((spanId) => spanHoverText(spansById.get(spanId))).filter(Boolean);
    if (lines.length) element.title = lines.join("\n\n");
  });
}

function spanHoverText(span) {
  if (!span) return "";
  return [
    span.span_id,
    `${span.span_type} · salience ${span.salience_score} · confidence ${span.locator_confidence}`,
    `[${span.span_start_idx}, ${span.span_end_idx})`,
    span.micro_summary,
    `start: ${span.start_anchor_quote}`,
    `end: ${span.end_anchor_quote}`,
  ].filter(Boolean).join("\n");
}

function segmentTitle(segment) {
  const ids = segment.span_ids && segment.span_ids.length ? `spans: ${segment.span_ids.join(", ")}` : "gap";
  return `${ids}\n[${segment.start_idx}, ${segment.end_idx})`;
}

function tagPill(tag) {
  const kind = tag.includes("failed") || tag.includes("rejected")
    ? "bad"
    : tag.includes("review") || tag.includes("cap")
      ? "warn"
      : "good";
  return `<span class="pill ${kind}">${escapeHtml(tag)}</span>`;
}

function percent(value) {
  return (Number(value || 0) * 100).toFixed(1);
}

function hashIndex(value, modulo) {
  let hash = 0;
  for (const char of String(value || "")) {
    hash = (hash * 31 + char.charCodeAt(0)) % 9973;
  }
  return hash % modulo;
}

async function startJob(command, payload, resultTarget) {
  closeEventSource();
  $("#event-log").innerHTML = "";
  resetProgress();
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
      updateProgress(event);
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

function resetProgress() {
  $("#job-progress-label").textContent = "waiting";
  $("#job-progress-meta").textContent = "";
  $("#job-progress-fill").style.width = "0%";
}

function updateProgress(event) {
  const payload = event.payload || {};
  if (!payload.stage) return;
  $("#job-progress-label").textContent = `${payload.stage}: ${payload.label || event.event}`;
  const parts = [];
  if (payload.current !== null && payload.current !== undefined && payload.total !== null && payload.total !== undefined) {
    parts.push(`${payload.current}/${payload.total}${payload.unit ? ` ${payload.unit}` : ""}`);
  }
  if (payload.percent !== null && payload.percent !== undefined) {
    const percent = Math.max(0, Math.min(100, Number(payload.percent)));
    parts.push(`${percent.toFixed(1)}%`);
    $("#job-progress-fill").style.width = `${percent}%`;
  }
  if (payload.status && payload.status !== "running") parts.push(payload.status);
  $("#job-progress-meta").textContent = parts.join(" · ");
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
  if (result && result.metrics) {
    parts.push(renderEvalReport(result));
  }
  if (result && result.question_set_path) {
    parts.push(renderEvalGenerationReport(result));
  }
  if (result && result.corpus_path && result.chapter_count) {
    parts.push(renderEvalCorpusReport(result));
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
        "micro_summary",
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
        "micro_summary",
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

function renderEvalReport(report) {
  const overall = (report.metrics || {}).overall || {};
  return `
    <div class="retrieval-section">
      <div class="metric-grid">
        ${metric("Questions", report.question_count || 0)}
        ${metric("Recall@3", formatMetric(overall.weighted_recall_at_3))}
        ${metric("Recall@20", formatMetric(overall.weighted_recall_at_20))}
        ${metric("NDCG@20", formatMetric(overall.ndcg_at_20))}
        ${metric("Facet@20", formatMetric(overall.facet_coverage_at_20))}
        ${metric("MRR", formatMetric(overall.mrr))}
      </div>
    </div>
    <div class="retrieval-section">
      ${keyValueTable({
        predictions_path: report.predictions_path || "",
        summary_path: report.report_path || "",
        failures_path: report.failures_path || "",
      })}
    </div>
  `;
}

function renderEvalGenerationReport(report) {
  return `
    <div class="retrieval-section">
      <div class="metric-grid">
        ${metric("Requested", report.requested_question_count || 0)}
        ${metric("Generated", report.generated_question_count || 0)}
        ${metric("Profile", report.profile || "")}
      </div>
    </div>
    <div class="retrieval-section">
      ${keyValueTable({
        corpus_path: report.corpus_path || "",
        question_set_path: report.question_set_path || "",
        report_path: report.report_path || "",
      })}
    </div>
  `;
}

function renderEvalCorpusReport(report) {
  return `
    <div class="retrieval-section">
      <div class="metric-grid">
        ${metric("Chapters", report.chapter_count || 0)}
        ${metric("Chars", report.char_count || 0)}
        ${metric("Range", `${report.chapter_start || ""}-${report.chapter_end || ""}`)}
      </div>
    </div>
    <div class="retrieval-section">
      ${keyValueTable({
        document_id: (report.document || {}).document_id || "",
        corpus_path: report.corpus_path || "",
      })}
    </div>
  `;
}

function formatMetric(value) {
  return value === null || value === undefined ? "0.0000" : Number(value).toFixed(4);
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
    "repair_failed",
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
    span_chars_min: "1",
    span_chars_max: "2000",
    chapter_start: "1",
    chapter_end: "100",
    question_count: "50",
    max_output_tokens: "384000",
    corpus: "data/eval/corpora/...",
    questions: "data/eval/question_sets/...",
    predictions: "data/eval/runs/...",
    output: "optional output path",
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
