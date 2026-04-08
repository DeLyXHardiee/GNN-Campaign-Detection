let state = {
  data: null,
  solution: null, // 'gnn' | 'featureset'
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

function buildTabs() {
  const tabs = document.getElementById("tabs");
  tabs.innerHTML = "";
  const gnn = state.data.gnn;
  const fs = state.data.featureset;

  if (!gnn && !fs) {
    tabs.innerHTML =
      '<p class="muted" style="padding:1rem">No GNN or featureset campaign data in this run.</p>';
    return;
  }

  if (gnn) {
    const b = document.createElement("button");
    b.className = "tab active";
    b.textContent = "GNN";
    b.dataset.sol = "gnn";
    b.addEventListener("click", () => selectTab("gnn"));
    tabs.appendChild(b);
  }
  if (fs) {
    const b = document.createElement("button");
    b.className = gnn ? "tab" : "tab active";
    b.textContent = "Feature set";
    b.dataset.sol = "featureset";
    b.addEventListener("click", () => selectTab("featureset"));
    tabs.appendChild(b);
  }

  state.solution = gnn ? "gnn" : "featureset";
  renderCampaigns();
}

function selectTab(sol) {
  state.solution = sol;
  state.campaignId = null;
  state.externalId = null;
  document.querySelectorAll(".tabs .tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.sol === sol);
  });
  renderCampaigns();
  document.getElementById("emailList").innerHTML = "";
  document.getElementById("emailDetail").innerHTML =
    '<p class="muted">Select a campaign and an email.</p>';
}

function currentPayload() {
  if (!state.data) return null;
  return state.solution === "gnn" ? state.data.gnn : state.data.featureset;
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
  const sol = state.solution;
  const cid = state.campaignId;
  if (cid == null || !state.data.attribute_similarity) return null;
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
