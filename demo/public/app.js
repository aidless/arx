"use strict";

const state = {
  payload: null,
  selectedId: null,
  mode: "static",
  config: null,
  reportUrl: null,
};

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function verdictClass(verdict) {
  return `verdict-${String(verdict || "").toLowerCase()}`;
}

function fmt(value) {
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

function num(value, digits = 4) {
  if (value === null || value === undefined) return "—";
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : fmt(value);
}

function renderSummary(payload) {
  const summary = payload.summary;
  const verdicts = summary.verdicts || {};
  const cards = [
    {
      value: summary.cases,
      label: "案例",
      hint: "构建时全部重算",
    },
    {
      value: verdicts.READABLE || 0,
      label: "READABLE",
      hint: "差距大于测量误差",
    },
    {
      value: verdicts.MARGINAL || 0,
      label: "MARGINAL",
      hint: "差距在分辨率内",
    },
    {
      value: verdicts.UNREADABLE || 0,
      label: "UNREADABLE",
      hint: "领先者没清过地板",
    },
    {
      value: verdicts.UNDECIDABLE || 0,
      label: "UNDECIDABLE",
      hint: "前提不成立，拒绝给数",
    },
  ];
  const root = $("summary");
  root.replaceChildren();
  for (const card of cards) {
    const node = el("div", "metric-card");
    node.append(el("span", "metric-value", card.value));
    node.append(el("span", "metric-label", card.label));
    node.append(el("span", "metric-hint", card.hint));
    root.append(node);
  }
}

function renderCases() {
  const root = $("case-list");
  root.replaceChildren();
  for (const item of state.payload.cases) {
    const button = el(
      "button",
      "case-button" + (item.id === state.selectedId ? " is-selected" : "")
    );
    button.type = "button";
    button.dataset.caseId = item.id;
    const top = el("div", "case-top");
    top.append(el("span", "case-id", item.id));
    top.append(el("span", `verdict-badge ${verdictClass(item.verdict)}`, item.verdict));
    button.append(top);
    button.append(el("p", "case-label", item.title));
    button.addEventListener("click", () => {
      state.selectedId = item.id;
      renderCases();
      renderDetail();
    });
    root.append(button);
  }
}

function selectedCase() {
  if (!state.payload || !state.payload.cases || !state.payload.cases.length) {
    return null;
  }
  return (
    state.payload.cases.find((item) => item.id === state.selectedId) ||
    state.payload.cases[0]
  );
}

function renderMetrics(item) {
  const root = $("metrics");
  root.replaceChildren();
  for (const metric of item.metrics || []) {
    const node = el("div", "metric");
    node.append(el("span", "label", metric.label));
    node.append(el("span", "value", fmt(metric.value)));
    node.append(el("span", "source", metric.source));
    root.append(node);
  }
}

function renderReasons(item) {
  const root = $("reasons");
  root.replaceChildren();
  root.append(el("h3", null, "拒绝理由"));
  if (!item.reasons || !item.reasons.length) {
    root.append(el("p", "verdict-meaning", "没有拒绝理由：这个判决是 READABLE。"));
    return;
  }
  const list = el("ul");
  for (const reason of item.reasons) {
    list.append(el("li", null, reason));
  }
  root.append(list);
}

function renderChance(item) {
  const root = $("chance");
  root.replaceChildren();
  if (!item.chance) return;
  root.append(el("h3", null, "地板 vs 随机水平"));
  const chance = item.chance;
  const z = Number(chance.z);
  const critical = Number(chance.critical_z);
  const text = [
    `above_chance=${fmt(chance.above_chance)}`,
    `z=${Number.isFinite(z) ? z.toFixed(2) : fmt(chance.z)}`,
    `critical_z=${Number.isFinite(critical) ? critical.toFixed(2) : fmt(chance.critical_z)}`,
    `n_items_needed=${fmt(chance.n_items_needed)}`,
  ].join(" · ");
  root.append(el("p", null, text));
  root.append(el("p", "source", chance.source));
}

function renderNotes(item) {
  const root = $("notes");
  root.replaceChildren();
  if (!item.notes || !item.notes.length) return;
  root.append(el("h3", null, "诊断备注"));
  const list = el("ul");
  for (const note of item.notes) {
    list.append(el("li", null, note));
  }
  root.append(list);
}

function renderBoard(item) {
  const table = $("board");
  table.replaceChildren();
  const board = item.board || {};
  const rows = [
    ["generators", board.n_generators],
    ["items", board.n_items],
    ["agreement", board.agreement],
    ["seed", board.seed],
    ["delta_declared", board.delta_declared],
    ["scores", JSON.stringify(board.scores)],
    ["judge_ids", JSON.stringify(board.judge_ids)],
    ["contestant_ids", JSON.stringify(board.contestant_ids)],
    ["votes_sha256_16", board.votes_sha256_16],
  ];
  if (board.note) rows.push(["note", board.note]);
  for (const [key, value] of rows) {
    const tr = el("tr");
    tr.append(el("th", null, key));
    tr.append(el("td", null, fmt(value)));
    table.append(tr);
  }
}

function renderReport(item) {
  const frame = $("report-frame");
  frame.onload = () => {
    try {
      const doc = frame.contentDocument;
      if (doc && doc.documentElement) {
        frame.style.height =
          Math.max(520, doc.documentElement.scrollHeight + 8) + "px";
      }
    } catch (error) {
      // Keep the CSS fallback height if the report cannot be measured.
    }
  };
  frame.srcdoc = item.report_html || "";
  if (state.reportUrl) {
    URL.revokeObjectURL(state.reportUrl);
    state.reportUrl = null;
  }
}

function renderDetail() {
  const item = selectedCase();
  if (!item) return;
  state.selectedId = item.id;
  $("case-title").textContent = item.title;
  $("case-id").textContent = item.id;
  $("case-question").textContent = item.question;

  const verdict = $("verdict");
  verdict.replaceChildren();
  verdict.append(
    el("span", `verdict-badge ${verdictClass(item.verdict)}`, item.verdict)
  );
  verdict.append(el("p", "verdict-meaning", item.meaning || ""));

  renderMetrics(item);
  renderReasons(item);
  renderChance(item);
  renderNotes(item);
  renderBoard(item);
  $("raw-json").textContent = JSON.stringify(item.report, null, 2);
  renderReport(item);
}

function renderRealCase(payload) {
  const section = document.querySelector(".real-case");
  const data = payload.real_case;
  if (!data) {
    if (section) section.hidden = true;
    return;
  }
  $("real-case-note").textContent = data.note || "";
  const link = $("real-case-link");
  link.href = data.url;
  const root = $("real-case-cards");
  root.replaceChildren();
  for (const audit of data.audits || []) {
    const card = el("article", "real-case-card");
    const head = el("div", "real-case-head");
    head.append(el("h3", null, audit.label));
    head.append(
      el("span", `verdict-badge ${verdictClass(audit.verdict)}`, audit.verdict)
    );
    card.append(head);
    const metrics = el("div", "real-case-metrics");
    const rows = [
      ["floor", num(audit.floor)],
      ["leader", `${fmt(audit.leader)} = ${num(audit.leader_score)}`],
      ["margin", audit.margin === null || audit.margin === undefined
        ? "—" : `${audit.margin >= 0 ? "+" : ""}${num(audit.margin)}`],
      ["MDE", num(audit.mde)],
      ["items", fmt(audit.n_items)],
      ["floor source", fmt(audit.floor_source)],
    ];
    for (const [label, value] of rows) {
      const metric = el("div", "real-case-metric");
      metric.append(el("span", "label", label));
      metric.append(el("span", "value", value));
      metrics.append(metric);
    }
    card.append(metrics);
    card.append(el("p", "real-case-source", audit.source));
    root.append(card);
  }
}

function renderSensitivity(payload) {
  const section = document.querySelector(".sensitivity");
  const data = payload.sensitivity;
  if (!data) {
    if (section) section.hidden = true;
    return;
  }
  const table = $("sensitivity-table");
  table.replaceChildren();
  const thead = el("thead");
  const headRow = el("tr");
  headRow.append(el("th", null, "items \\ delta"));
  for (const delta of data.deltas || []) {
    headRow.append(el("th", null, `delta=${delta}`));
  }
  thead.append(headRow);
  table.append(thead);
  const body = el("tbody");
  for (const row of data.rows || []) {
    const tr = el("tr");
    tr.append(el("th", null, fmt(row.n_items)));
    for (const cell of row.cells || []) {
      const td = el("td");
      td.append(
        el("span", `verdict-badge ${verdictClass(cell.verdict)}`, cell.verdict)
      );
      td.append(
        el(
          "div",
          "sensitivity-detail",
          `floor ${num(cell.floor)} · margin ${num(cell.margin)} · mde ${num(cell.mde)}`
        )
      );
      td.title = `n=${cell.n_items}, delta=${cell.delta}, floor=${cell.floor}, margin=${cell.margin}, mde=${cell.mde}`;
      tr.append(td);
    }
    body.append(tr);
  }
  table.append(body);
  $("sensitivity-note").textContent = data.note || "";
}

function renderBoundaries(payload) {
  const root = $("boundaries");
  root.replaceChildren();
  for (const boundary of payload.boundaries || []) {
    root.append(el("li", null, boundary));
  }
  const caseStudy = $("case-study");
  caseStudy.replaceChildren();
  if (payload.case_study) {
    caseStudy.append(document.createTextNode("真实数据案例："));
    const link = el("a", null, payload.case_study.title);
    link.href = payload.case_study.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    caseStudy.append(link);
    caseStudy.append(document.createTextNode(` — ${payload.case_study.note}`));
  }
}

function renderMeta(payload) {
  const meta = payload.meta || {};
  const verification = payload.verification || {};
  const rows = [
    ["仓库", meta.repo],
    ["Demo", meta.demo],
    ["提交", meta.commit],
    ["工作区", meta.worktree_dirty ? "有未提交改动" : "干净（cases.json 除外）"],
    ["生成时间 (UTC)", meta.generated_at],
    ["模式", meta.mode],
    ["seed", meta.seed],
    ["Python", meta.python],
    ["NumPy", meta.numpy],
    ["判决对账", `${verification.status || "—"} (${verification.cases || 0} cases)`],
  ];
  if (meta.dirty_paths && meta.dirty_paths.length) {
    rows.push(["未提交路径", meta.dirty_paths.join(", ")]);
  }
  const dl = $("meta");
  dl.replaceChildren();
  for (const [key, value] of rows) {
    dl.append(el("dt", null, key));
    dl.append(el("dd", null, value === undefined || value === null ? "—" : value));
  }
}

function renderMode() {
  const badge = $("mode-badge");
  badge.className = "badge " + (state.mode === "live" ? "badge-live" : "badge-static");
  badge.textContent = state.mode === "live" ? "本地实时" : "静态回放";
  $("mode-note").textContent =
    state.mode === "live"
      ? "点击“重跑全部案例”会重新执行六个审计；判决与报告随之更新。"
      : "这是提交进仓库的一次确定性构建。运行 python3 demo/server.py 可实时重算。";
  $("rerun").disabled = state.mode !== "live";
}

function renderAll() {
  renderSummary(state.payload);
  renderRealCase(state.payload);
  renderSensitivity(state.payload);
  renderCases();
  renderDetail();
  renderBoundaries(state.payload);
  renderMeta(state.payload);
  renderMode();
}

async function loadConfig() {
  try {
    const response = await fetch("./config.json", { cache: "no-store" });
    if (response.ok) return await response.json();
  } catch (error) {
    // file:// or a static host without config.json: stay in replay mode.
  }
  return { mode: "static" };
}

async function rerun() {
  const button = $("rerun");
  const status = $("rerun-status");
  button.disabled = true;
  status.textContent = "正在重跑六个审计…";
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    state.payload = payload;
    if (!payload.cases.some((item) => item.id === state.selectedId)) {
      state.selectedId = payload.cases[0] ? payload.cases[0].id : null;
    }
    renderAll();
    status.textContent = `重跑完成 · ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    status.textContent = `重跑失败：${error.message}`;
  } finally {
    button.disabled = state.mode !== "live";
  }
}

function openReport() {
  const item = selectedCase();
  if (!item) return;
  state.reportUrl = URL.createObjectURL(
    new Blob([item.report_html], { type: "text/html;charset=utf-8" })
  );
  window.open(state.reportUrl, "_blank", "noopener,noreferrer");
}

function downloadReport() {
  const item = selectedCase();
  if (!item) return;
  const url = URL.createObjectURL(
    new Blob([item.report_html], { type: "text/html;charset=utf-8" })
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `arx-audit-${item.id}.html`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function init() {
  try {
    state.config = await loadConfig();
    state.mode = state.config.mode === "live" ? "live" : "static";
    const response = await fetch("./cases.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`cases.json HTTP ${response.status}`);
    state.payload = await response.json();
    state.selectedId = state.payload.cases[0] ? state.payload.cases[0].id : null;
    renderAll();

    $("rerun").addEventListener("click", rerun);
    $("open-report").addEventListener("click", openReport);
    $("download-report").addEventListener("click", downloadReport);
    $("raw-toggle").addEventListener("click", () => {
      const pre = $("raw-json");
      const button = $("raw-toggle");
      const wasHidden = pre.hidden;
      pre.hidden = !wasHidden;
      button.setAttribute("aria-expanded", String(wasHidden));
      button.textContent = wasHidden ? "收起原始 AuditReport JSON" : "查看原始 AuditReport JSON";
    });
  } catch (error) {
    const box = $("error");
    box.hidden = false;
    box.textContent =
      `加载失败：${error.message}。如果你在用 file:// 打开页面，` +
      "请运行 python3 demo/server.py 或 python3 -m http.server -d demo/public。";
  }
}

document.addEventListener("DOMContentLoaded", init);
