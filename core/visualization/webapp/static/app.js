let state = {
  data: null,
  solutionKey: null,
  campaignId: null,
  externalId: null,
};

async function loadData() {
  const r = await fetch("/api/data");
  if (!r.ok) throw new Error(await r.text());
  state.data = await r.json();
  document.getElementById("runInfo").textContent =
    state.data.run_dir
      ? `Run: ${state.data.run_dir}`
      : "";
  const leg = document.getElementById("simLegend");
  if (leg) {
    if (state.data.attribute_similarity_error) {
      leg.textContent = `Similarity unavailable: ${state.data.attribute_similarity_error}`;
    } else if (!state.data.attribute_similarity) {
      leg.textContent =
        "No per-attribute similarity in this export (disabled in config or not computed).";
    }
  }
  buildTabs();
}

function solutionKeys() {
  const sols = state.data && state.data.solutions;
  if (!sols) return [];
  return Object.keys(sols);
}

function buildTabs() {
  const tabs = document.getElementById("tabs");
  tabs.innerHTML = "";
  const keys = solutionKeys();

  if (keys.length === 0) {
    tabs.innerHTML =
      '<p class="muted" style="padding:1rem">No campaigns*.json files found in this run.</p>';
    state.solutionKey = null;
    renderCampaigns();
    return;
  }

  keys.forEach((key, i) => {
    const sol = state.data.solutions[key];
    const b = document.createElement("button");
    b.className = i === 0 ? "tab active" : "tab";
    b.textContent = sol.label || key;
    b.title = key;
    b.dataset.sol = key;
    b.addEventListener("click", () => selectTab(key));
    tabs.appendChild(b);
  });

  state.solutionKey = keys[0];
  renderCampaigns();
}

function selectTab(key) {
  state.solutionKey = key;
  state.campaignId = null;
  state.externalId = null;
  document.querySelectorAll(".tabs .tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.sol === key);
  });
  renderCampaigns();
  document.getElementById("emailList").innerHTML = "";
  document.getElementById("emailDetail").innerHTML =
    '<p class="muted">Select a campaign and an email.</p>';
}

function currentPayload() {
  if (!state.data || !state.solutionKey) return null;
  return state.data.solutions ? state.data.solutions[state.solutionKey] : null;
}

function renderCampaigns() {
  const ul = document.getElementById("campaignList");
  ul.innerHTML = "";
  const payload = currentPayload();
  if (!payload || !payload.campaigns) {
    ul.innerHTML = '<li class="muted">No campaigns</li>';
    return;
  }
  payload.campaigns.forEach((c) => {
    const li = document.createElement("li");
    const id = c.id;
    const btn = document.createElement("button");
    btn.textContent = `Campaign ${id} (${c.size})`;
    btn.addEventListener("click", () => {
      state.campaignId = id;
      state.externalId = null;
      document.querySelectorAll("#campaignList button").forEach((x) =>
        x.classList.remove("active"),
      );
      btn.classList.add("active");
      renderEmails(c);
    });
    li.appendChild(btn);
    ul.appendChild(li);
  });
}

function renderEmails(campaign) {
  const ul = document.getElementById("emailList");
  ul.innerHTML = "";
  const emails = state.data.emails || {};
  (campaign.member_external_ids || []).forEach((eid) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    const subj = emails[eid]?.subject || eid;
    btn.textContent = subj.length > 60 ? subj.slice(0, 57) + "…" : subj;
    btn.title = eid;
    btn.addEventListener("click", () => {
      state.externalId = eid;
      document.querySelectorAll("#emailList button").forEach((x) =>
        x.classList.remove("active"),
      );
      btn.classList.add("active");
      renderEmailDetail(eid);
    });
    li.appendChild(btn);
    ul.appendChild(li);
  });
}

function esc(s) {
  if (s == null) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}

function similarityFor(eid) {
  const sol = state.solutionKey;
  const cid = state.campaignId;
  if (sol == null || cid == null || !state.data.attribute_similarity) return null;
  const block =
    state.data.attribute_similarity[sol]?.[String(cid)]?.[eid];
  return block || null;
}

function scoreToBg(score) {
  if (score == null || Number.isNaN(Number(score))) return "transparent";
  const t = Math.max(0, Math.min(1, Number(score)));
  const hue = t * 120;
  /* red (0) → green (120); darker mid-tone so hue differences stay visible */
  return `hsl(${hue}, 88%, 78%)`;
}

function attrRow(label, value, score) {
  const bg = scoreToBg(score);
  const badge =
    score != null && !Number.isNaN(Number(score))
      ? ` <span class="sim-badge">${(Number(score) * 100).toFixed(0)}%</span>`
      : "";
  return `<div class="row attr-row" style="background:${bg}"><span class="label">${esc(
    label,
  )}${badge}</span><br/>${esc(value)}</div>`;
}

function renderEmailDetail(eid) {
  const emails = state.data.emails || {};
  const e = emails[eid];
  const el = document.getElementById("emailDetail");

  if (!e) {
    el.innerHTML = `<p class="muted">No email body in MISP for <code>${esc(
      eid,
    )}</code></p>`;
    return;
  }
  const toList = Array.isArray(e.receivers)
    ? e.receivers.join(", ")
    : e.receivers || "";
  const fromList = Array.isArray(e.senders)
    ? e.senders.join(", ")
    : e.senders || "";
  const sim = similarityFor(eid);
  const s = (k) => (sim && sim[k] != null ? sim[k] : null);

  const bodyBg = scoreToBg(s("body"));
  el.innerHTML = `
    <div class="row"><span class="label">External ID</span><br/>${esc(e.external_id || eid)}</div>
    ${attrRow("From", fromList, s("senders"))}
    ${attrRow("To", toList, s("receivers"))}
    ${attrRow("Date", e.date, s("date"))}
    ${attrRow("Subject", e.subject, s("subject"))}
    <div class="row"><span class="label">Body</span>${s("body") != null ? ` <span class="sim-badge">${(Number(s("body")) * 100).toFixed(0)}%</span>` : ""}</div>
    <pre class="body" style="background:${bodyBg}">${esc(e.body)}</pre>
  `;
}

loadData().catch((err) => {
  document.body.innerHTML = `<p style="padding:2rem;color:#c00">Failed to load: ${esc(
    err.message,
  )}</p>`;
});
