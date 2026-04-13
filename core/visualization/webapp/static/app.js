let state = {
  data: null,
  solution: null, // 'gnn' | 'featureset'
  campaignId: null,
  externalId: null,
};

function renderUmap() {
  const section = document.getElementById("umapSection");
  const cap = document.getElementById("umapCaption");
  const canvas = document.getElementById("umapCanvas");
  const leg = document.getElementById("umapLegend");
  const greyNote = document.getElementById("umapGreyNote");
  const tooltip = document.getElementById("umapTooltip");
  if (!section || !canvas || !state.data) return;

  const umap = state.data.umap;
  if (!umap) {
    section.hidden = true;
    return;
  }

  section.hidden = false;
  if (umap.error) {
    cap.textContent = umap.error;
    leg.innerHTML = "";
    if (greyNote) greyNote.textContent = "";
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "#f5f5f5";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#666";
      ctx.font = "14px system-ui";
      ctx.fillText("UMAP not available", 16, 32);
    }
    return;
  }

  const pts = umap.points || [];
  const params = umap.params || {};
  cap.textContent = `n=${umap.n_emails ?? pts.length}, dim=${umap.embedding_dim ?? "?"}, n_neighbors=${params.n_neighbors ?? "?"}, min_dist=${params.min_dist ?? "?"}`;

  leg.innerHTML = "";
  (umap.legend || []).forEach((row) => {
    const li = document.createElement("li");
    const sw = document.createElement("span");
    sw.className = "umap-swatch";
    sw.style.background = row.color;
    li.appendChild(sw);
    li.appendChild(
      document.createTextNode(`Campaign ${row.campaign}`),
    );
    leg.appendChild(li);
  });
  if (greyNote) {
    greyNote.textContent = `Grey points: not in ground_truth.json (${umap.no_ground_truth_color || "#b0b0b0"}).`;
  }

  const dpr = window.devicePixelRatio || 1;
  const cssW = 900;
  const cssH = 420;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#fafafa";
  ctx.fillRect(0, 0, cssW, cssH);

  if (pts.length === 0) return;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  pts.forEach((p) => {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  });
  const pad = 24;
  const dx = maxX - minX || 1e-9;
  const dy = maxY - minY || 1e-9;
  const plotW = cssW - 2 * pad;
  const plotH = cssH - 2 * pad;

  function tx(x) {
    return pad + ((x - minX) / dx) * plotW;
  }
  function ty(y) {
    return pad + ((maxY - y) / dy) * plotH;
  }

  pts.forEach((p) => {
    ctx.beginPath();
    ctx.arc(tx(p.x), ty(p.y), 3, 0, Math.PI * 2);
    ctx.fillStyle = p.color || "#999";
    ctx.fill();
  });

  const HIT_RADIUS_SQ = 100;

  function pickPoint(mx, my) {
    let best = -1;
    let bestD = HIT_RADIUS_SQ;
    pts.forEach((p, i) => {
      const px = tx(p.x);
      const py = ty(p.y);
      const d = (mx - px) * (mx - px) + (my - py) * (my - py);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    if (best < 0 || bestD > HIT_RADIUS_SQ) return null;
    return pts[best];
  }

  canvas.onmousemove = (ev) => {
    const mx = ev.offsetX;
    const my = ev.offsetY;
    const hit = pickPoint(mx, my);
    canvas.style.cursor = hit ? "pointer" : "default";
    if (!hit) {
      tooltip.hidden = true;
      return;
    }
    tooltip.hidden = false;
    tooltip.textContent = `${hit.external_id}\nGT campaign: ${hit.has_ground_truth ? String(hit.ground_truth_campaign) : "(none)"}\n(Click to open in list)`;
    tooltip.style.left = `${Math.min(mx + 12, cssW - 200)}px`;
    tooltip.style.top = `${Math.min(my + 12, cssH - 48)}px`;
  };
  canvas.onmouseleave = () => {
    tooltip.hidden = true;
    canvas.style.cursor = "default";
  };

  canvas.onclick = (ev) => {
    const hit = pickPoint(ev.offsetX, ev.offsetY);
    if (hit) openEmailFromUmap(hit.external_id);
  };
}

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
  renderUmap();
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
    btn.type = "button";
    btn.dataset.campaignId = String(id);
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
    btn.type = "button";
    btn.dataset.externalId = eid;
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

/**
 * Find the GNN campaign dict that contains this external_id, or null.
 */
function findGnnCampaignForEmail(externalId) {
  const gnn = state.data && state.data.gnn;
  if (!gnn || !gnn.campaigns) return null;
  const eid = String(externalId);
  for (const c of gnn.campaigns) {
    const members = c.member_external_ids || [];
    if (members.some((m) => String(m) === eid)) return c;
  }
  return null;
}

/**
 * From UMAP: switch to GNN tab, open the cluster that contains the email, select it, show detail.
 * If the email is not in any GNN campaign, still show email detail when present in the catalog.
 */
function openEmailFromUmap(externalId) {
  const eid = String(externalId);
  if (!state.data) return;

  if (state.data.gnn) {
    state.solution = "gnn";
    document.querySelectorAll(".tabs .tab").forEach((t) => {
      t.classList.toggle("active", t.dataset.sol === "gnn");
    });
  }

  const campaign = findGnnCampaignForEmail(eid);
  if (campaign) {
    state.campaignId = campaign.id;
    state.externalId = eid;
    renderCampaigns();
    document.querySelectorAll("#campaignList button").forEach((b) => {
      b.classList.toggle("active", b.dataset.campaignId === String(campaign.id));
    });
    renderEmails(campaign);
    document.querySelectorAll("#emailList button").forEach((x) =>
      x.classList.remove("active"),
    );
    document.querySelectorAll("#emailList button").forEach((b) => {
      if (b.dataset.externalId === eid) b.classList.add("active");
    });
    renderEmailDetail(eid);
    return;
  }

  state.campaignId = null;
  state.externalId = eid;
  if (state.data.gnn) {
    renderCampaigns();
  }
  document.getElementById("emailList").innerHTML =
    '<li class="muted">This email is not in any GNN campaign cluster.</li>';
  renderEmailDetail(eid);
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
