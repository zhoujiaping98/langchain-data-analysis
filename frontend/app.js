const chatList = document.getElementById("chatList");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const statusPill = document.getElementById("statusPill");
const routeSnapshot = document.getElementById("routeSnapshot");
const sqlSnapshot = document.getElementById("sqlSnapshot");
const riskSnapshot = document.getElementById("riskSnapshot");

let isStreaming = false;
let lastQuery = "";
let sessionId = null;
let pendingAction = null;
let pendingHint = "";

function setStatus(text, tone = "ready") {
  statusPill.textContent = text;
  statusPill.style.background =
    tone === "error"
      ? "rgba(255, 107, 107, 0.18)"
      : tone === "busy"
        ? "rgba(247, 178, 103, 0.2)"
        : "rgba(96, 211, 148, 0.18)";
  statusPill.style.color =
    tone === "error" ? "#ffb3b3" : tone === "busy" ? "#f7b267" : "#60d394";
}

function addMessage(role, content, title = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  if (title) {
    const h4 = document.createElement("h4");
    h4.textContent = title;
    wrapper.appendChild(h4);
  }
  const body = document.createElement("div");
  body.innerHTML = content;
  wrapper.appendChild(body);
  chatList.appendChild(wrapper);
  chatList.scrollTop = chatList.scrollHeight;
  return body;
}

function addCodeBlock(role, title, code) {
  return addMessage(role, `<pre class="code-block">${escapeHtml(code)}</pre>`, title);
}

function addTable(role, title, rows) {
  if (!rows || !rows.length) {
    return addMessage(role, "没有返回数据。", title);
  }
  const cols = Object.keys(rows[0]);
  const thead = cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("");
  const tbody = rows
    .slice(0, 20)
    .map(
      (row) =>
        `<tr>${cols.map((c) => `<td>${escapeHtml(String(row[c] ?? ""))}</td>`).join("")}</tr>`
    )
    .join("");
  const table = `
    <table class="result-table">
      <thead><tr>${thead}</tr></thead>
      <tbody>${tbody}</tbody>
    </table>
  `;
  return addMessage(role, table, title);
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function typeText(target, text, speed = 14) {
  target.textContent = "";
  let i = 0;
  const timer = setInterval(() => {
    target.textContent += text.charAt(i);
    i += 1;
    chatList.scrollTop = chatList.scrollHeight;
    if (i >= text.length) {
      clearInterval(timer);
    }
  }, speed);
}

function renderOptions(payload, originQuery) {
  const options = payload.options || [];
  if (payload.input_mode === "template") {
    renderTemplateForm(payload, originQuery);
    return;
  }
  if (payload.input_mode === "filters") {
    renderFilterForm(payload, originQuery);
    return;
  }
  const hasOptions = options.length > 0;
  const hasSkip =
    (payload.input_mode === "intent" && payload.skip_action) ||
    (payload.input_mode === "filters" && payload.skip_action);
  if (!hasOptions && !hasSkip) return;

  const container = document.createElement("div");
  container.className = "option-bar";

  if (payload.selection_mode === "multi") {
    const selected = new Set();
    options.forEach((opt) => {
      const btn = document.createElement("button");
      btn.className = "option-btn";
      btn.textContent = opt.label;
      btn.onclick = () => {
        if (selected.has(opt.value)) {
          selected.delete(opt.value);
          btn.style.borderColor = "rgba(255, 255, 255, 0.1)";
        } else {
          selected.add(opt.value);
          btn.style.borderColor = "#f7b267";
        }
      };
      container.appendChild(btn);
    });

    const confirmBtn = document.createElement("button");
    confirmBtn.className = "option-btn";
    confirmBtn.textContent = payload.confirm_label || "确认";
    confirmBtn.onclick = () => {
      streamQuery(originQuery, payload.confirm_action || "choose_fields", Array.from(selected));
    };
    container.appendChild(confirmBtn);

    if (payload.skip_action) {
      const skipBtn = document.createElement("button");
      skipBtn.className = "option-btn";
      skipBtn.textContent = payload.skip_label || "跳过";
      skipBtn.onclick = () => {
        streamQuery(originQuery, payload.skip_action, null);
      };
      container.appendChild(skipBtn);
    }
  } else {
    options.forEach((opt) => {
      const btn = document.createElement("button");
      btn.className = "option-btn";
      btn.textContent = opt.label;
      btn.onclick = () => {
        streamQuery(originQuery, opt.action, opt.value);
      };
      container.appendChild(btn);
    });
  }

  if (hasSkip) {
    const skipBtn = document.createElement("button");
    skipBtn.className = "option-btn";
    skipBtn.textContent = payload.skip_label || "跳过";
    skipBtn.onclick = () => {
      streamQuery(originQuery, payload.skip_action, null);
    };
    container.appendChild(skipBtn);
  }

  const msg = addMessage("agent", "", "请选择补充信息");
  msg.appendChild(container);
}

function renderTemplateForm(payload, originQuery) {
  const tpl = payload.template || {};
  const container = document.createElement("div");
  container.className = "option-bar";

  const aggSelect = buildSelect("统计方式", tpl.aggregations || []);
  const metricSelect = buildSelect("指标字段", tpl.metric_fields || []);
  const timeSelect = buildSelect("时间字段", tpl.time_fields || []);
  const grainSelect = buildSelect("时间粒度", tpl.grains || []);
  const rangeSelect = buildSelect("时间范围", tpl.time_ranges || []);

  container.appendChild(aggSelect.wrapper);
  container.appendChild(metricSelect.wrapper);
  container.appendChild(timeSelect.wrapper);
  container.appendChild(grainSelect.wrapper);
  container.appendChild(rangeSelect.wrapper);

  const goBtn = document.createElement("button");
  goBtn.className = "option-btn";
  goBtn.textContent = "生成SQL";
  goBtn.onclick = () => {
    const selection = {
      aggregation: aggSelect.select.value,
      metric_field: metricSelect.select.value,
      time_field: timeSelect.select.value,
      grain: grainSelect.select.value,
      time_range: rangeSelect.select.value,
    };
    streamQuery(originQuery, "choose_template", selection);
  };
  container.appendChild(goBtn);

  const msg = addMessage("agent", "", "统计口径模板");
  msg.appendChild(container);
}

function buildSelect(label, options) {
  const wrapper = document.createElement("div");
  wrapper.style.display = "flex";
  wrapper.style.flexDirection = "column";
  wrapper.style.gap = "6px";
  wrapper.style.minWidth = "140px";

  const span = document.createElement("span");
  span.textContent = label;
  span.style.fontSize = "12px";
  span.style.color = "#9aa2b1";

  const select = document.createElement("select");
  select.style.background = "rgba(255,255,255,0.05)";
  select.style.border = "1px solid rgba(255,255,255,0.1)";
  select.style.borderRadius = "8px";
  select.style.color = "white";
  select.style.padding = "6px";
  options.forEach((opt) => {
    const o = document.createElement("option");
    o.value = opt;
    o.textContent = opt;
    select.appendChild(o);
  });

  wrapper.appendChild(span);
  wrapper.appendChild(select);
  return { wrapper, select };
}

function renderFilterForm(payload, originQuery) {
  const fields = payload.filter_fields || [];
  const container = document.createElement("div");
  container.className = "option-bar";

  if (!fields.length) {
    const msg = addMessage("agent", "未提供可选字段，请跳过过滤条件。", "过滤条件");
    msg.appendChild(container);
    if (payload.skip_action) {
      const skipBtn = document.createElement("button");
      skipBtn.className = "option-btn";
      skipBtn.textContent = payload.skip_label || "跳过";
      skipBtn.onclick = () => {
        streamQuery(originQuery, payload.skip_action, null);
      };
      container.appendChild(skipBtn);
    }
    return;
  }

  const opOptions = ["=", "!=", ">", "<", ">=", "<=", "contains", "in"];
  const fieldSelect = buildSelect("字段", fields);
  const opSelect = buildSelect("运算符", opOptions);

  const valueWrapper = document.createElement("div");
  valueWrapper.style.display = "flex";
  valueWrapper.style.flexDirection = "column";
  valueWrapper.style.gap = "6px";
  valueWrapper.style.minWidth = "160px";
  const valueLabel = document.createElement("span");
  valueLabel.textContent = "值";
  valueLabel.style.fontSize = "12px";
  valueLabel.style.color = "#9aa2b1";
  const valueInput = document.createElement("input");
  valueInput.placeholder = "例如: 已支付 / 2025-01-01";
  valueInput.style.background = "rgba(255,255,255,0.05)";
  valueInput.style.border = "1px solid rgba(255,255,255,0.1)";
  valueInput.style.borderRadius = "8px";
  valueInput.style.color = "white";
  valueInput.style.padding = "6px";
  valueWrapper.appendChild(valueLabel);
  valueWrapper.appendChild(valueInput);

  const list = document.createElement("div");
  list.style.display = "flex";
  list.style.flexDirection = "column";
  list.style.gap = "6px";
  list.style.width = "100%";

  const filters = [];

  const addBtn = document.createElement("button");
  addBtn.className = "option-btn";
  addBtn.textContent = "添加过滤";
  addBtn.onclick = () => {
    const f = fieldSelect.select.value;
    const op = opSelect.select.value;
    const val = valueInput.value.trim();
    if (!f || !op || !val) return;
    filters.push({ field: f, op, value: val });
    valueInput.value = "";
    renderFilterList();
  };

  const renderFilterList = () => {
    list.innerHTML = "";
    filters.forEach((f, idx) => {
      const row = document.createElement("div");
      row.style.display = "flex";
      row.style.alignItems = "center";
      row.style.gap = "8px";
      row.style.fontSize = "12px";
      row.textContent = `${f.field} ${f.op} ${f.value}`;
      const del = document.createElement("button");
      del.className = "option-btn";
      del.textContent = "移除";
      del.onclick = () => {
        filters.splice(idx, 1);
        renderFilterList();
      };
      row.appendChild(del);
      list.appendChild(row);
    });
  };

  const confirmBtn = document.createElement("button");
  confirmBtn.className = "option-btn";
  confirmBtn.textContent = "确认过滤条件";
  confirmBtn.onclick = () => {
    streamQuery(originQuery, "choose_filters", filters);
  };

  container.appendChild(fieldSelect.wrapper);
  container.appendChild(opSelect.wrapper);
  container.appendChild(valueWrapper);
  container.appendChild(addBtn);
  container.appendChild(confirmBtn);
  if (payload.skip_action) {
    const skipBtn = document.createElement("button");
    skipBtn.className = "option-btn";
    skipBtn.textContent = payload.skip_label || "跳过";
    skipBtn.onclick = () => {
      streamQuery(originQuery, payload.skip_action, null);
    };
    container.appendChild(skipBtn);
  }
  container.appendChild(list);

  const msg = addMessage("agent", "", "过滤条件");
  msg.appendChild(container);
}

async function streamQuery(query, action = null, selection = null) {
  if (isStreaming) return;
  isStreaming = true;
  sendBtn.disabled = true;
  setStatus("运行中", "busy");
  lastQuery = query;
  pendingAction = null;
  pendingHint = "";

  addMessage("user", escapeHtml(query));
  const loading = addMessage("agent", "正在进入四层流水线…", "Agent");

  const payload = {
    user_id: "u_demo",
    role: "analyst",
    query,
    action,
    selection,
    session_id: sessionId,
  };

  try {
    const res = await fetch("/agent/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok || !res.body) {
      throw new Error("服务不可用，请检查后端状态。");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";

      chunks.forEach((chunk) => {
        const lines = chunk.split("\n");
        let event = "message";
        let data = "";
        lines.forEach((line) => {
          if (line.startsWith("event:")) {
            event = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            data += line.slice(5).trim();
          }
        });
        if (!data) return;
        const payload = JSON.parse(data);
        handleEvent(event, payload, loading);
      });
    }
  } catch (err) {
    addMessage("system", err.message || "请求失败");
    setStatus("异常", "error");
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    setStatus("就绪", "ready");
  }
}

function handleEvent(event, payload, loadingNode) {
  if (!payload) return;

  switch (event) {
    case "session": {
      sessionId = payload.session_id;
      break;
    }
    case "status": {
      loadingNode.innerHTML = escapeHtml(payload.message || "处理中…");
      break;
    }
    case "route": {
      routeSnapshot.textContent = JSON.stringify(payload, null, 2);
      break;
    }
    case "ask": {
      const msg = payload.message || "需要补充信息。";
      const node = addMessage("agent", escapeHtml(msg), "Agent");
      if (payload.input_mode === "intent") {
        pendingAction = "provide_intent";
        pendingHint = "请输入统计口径、过滤条件和时间范围...";
        queryInput.placeholder = pendingHint;
        queryInput.focus();
      }
      renderOptions(payload, lastQuery);
      node.scrollIntoView({ behavior: "smooth" });
      break;
    }
    case "sql": {
      sqlSnapshot.textContent = payload.sql || "";
      if (payload.post_risk) {
        riskSnapshot.textContent = JSON.stringify(payload.post_risk, null, 2);
      }
      addCodeBlock("agent", "SQL", payload.sql || "");
      if (payload.tags) {
        const tagText = Object.entries(payload.tags)
          .map(([k, v]) => `${k}: ${v}`)
          .join("\n");
        addCodeBlock("agent", "SQL 结构预览", tagText);
      }
      break;
    }
    case "rows": {
      addTable("agent", `结果（共 ${payload.row_count || 0} 行）`, payload.rows || []);
      break;
    }
    case "analysis": {
      const body = addMessage("agent", "", "分析");
      typeText(body, payload.text || "暂无分析");
      break;
    }
    case "error": {
      addMessage("system", escapeHtml(payload.message || "出错了"));
      break;
    }
    case "block": {
      addMessage("system", escapeHtml(payload.message || "请求被阻止"));
      break;
    }
    case "done": {
      break;
    }
    default:
      break;
  }
}

sendBtn.addEventListener("click", () => {
  const query = queryInput.value.trim();
  if (!query) return;
  queryInput.value = "";
  if (pendingAction === "provide_intent") {
    streamQuery(lastQuery, "provide_intent", query);
  } else {
    streamQuery(query);
  }
});

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendBtn.click();
  }
});

clearBtn.addEventListener("click", () => {
  chatList.innerHTML = "";
  routeSnapshot.textContent = "等待请求…";
  sqlSnapshot.textContent = "等待生成…";
  riskSnapshot.textContent = "等待评估…";
  sessionId = null;
  pendingAction = null;
  pendingHint = "";
  queryInput.placeholder = "例如：上周GMV按渠道统计";
});
