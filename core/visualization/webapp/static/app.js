const EVAL_METRICS_STORAGE_KEY = "campaignViewer.showEvalMetrics";

let state = {
  data: null,
  solutionKey: null,
  campaignId: null,
  externalId: null,
  showEvalMetrics: false,
  gtIdSet: null,
};

async function loadData() {
  const r = await fetch("/api/data");
  if (!r.ok) throw new Error(await r.text());
  state.data = await r.json();
  const gtIds = state.data.ground_truth_ids;
  state.gtIdSet =
    Array.isArray(gtIds) && gtIds.length > 0
      ? new Set(gtIds.map((id) => String(id)))
      : null;
  const info = [];
  if (state.data.run_dir) info.push(`Run: ${state.data.run_dir}`);
  if (state.data.misp_json_path) info.push(`MISP: ${state.data.misp_json_path}`);
  document.getElementById("runInfo").textContent = info.join(" · ");
  const leg = document.getElementById("simLegend");
  if (leg) {
    if (state.data.attribute_similarity_error) {
      leg.textContent = `Similarity unavailable: ${state.data.attribute_similarity_error}`;
    } else if (!state.data.attribute_similarity) {
      leg.textContent =
        "No per-campaign similarity in this export (disabled in config or not computed).";
    }
  }
  setupEvalMetricsToggle();
  buildTabs();
}

function setupEvalMetricsToggle() {
  const wrap = document.getElementById("evalMetricsToggleWrap");
  const input = document.getElementById("showEvalMetrics");
  if (!wrap || !input) return;

  const available = Boolean(state.data?.campaign_eval_metrics_available);
  wrap.hidden = !available;
  if (!available) {
    state.showEvalMetrics = false;
    return;
  }

  const stored = localStorage.getItem(EVAL_METRICS_STORAGE_KEY);
  state.showEvalMetrics = stored === "1";
  input.checked = state.showEvalMetrics;

  input.addEventListener("change", () => {
    state.showEvalMetrics = input.checked;
    localStorage.setItem(EVAL_METRICS_STORAGE_KEY, state.showEvalMetrics ? "1" : "0");
    renderCampaigns();
  });
}

function formatEvalMetric(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(2);
}

function hasGroundTruth(eid, row) {
  if (row?.ground_truth_label != null) return true;
  if (!state.gtIdSet) return false;
  const key = String(eid);
  return state.gtIdSet.has(key);
}

function campaignEvalMetricsSuffix(campaign) {
  if (!state.showEvalMetrics) return "";
  const nGt = Number(campaign.n_eval);
  const gtPart =
    Number.isFinite(nGt) && nGt >= 0 ? ` GT${nGt}` : "";
  const h = formatEvalMetric(campaign.homogeneity);
  const c = formatEvalMetric(campaign.completeness);
  const v = formatEvalMetric(campaign.v_measure);
  if (!gtPart && h === "—" && c === "—" && v === "—") return "";
  return `${gtPart} H${h} C${c} V${v}`;
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
    '<p class="muted">Select a campaign, then an email.</p>';
}

function currentPayload() {
  if (!state.data || !state.solutionKey) return null;
  return state.data.solutions ? state.data.solutions[state.solutionKey] : null;
}

function campaignEmailCount(campaign) {
  const n = Number(campaign.size);
  if (Number.isFinite(n) && n >= 0) return n;
  return (campaign.member_external_ids || []).length;
}

function campaignsBySizeDesc(campaigns) {
  return [...campaigns].sort(
    (a, b) => campaignEmailCount(b) - campaignEmailCount(a),
  );
}

function isCampaignNoise(campaign) {
  if (campaign?.is_noise === true) return true;
  return campaignEmailCount(campaign) === 1;
}

function campaignNoiseTotal(payload, campaigns) {
  const total = Number(payload?.n_noise_total);
  if (Number.isFinite(total) && total >= 0) return total;
  const labeled = Number(payload?.n_noise);
  const labeledNoise = Number.isFinite(labeled) && labeled >= 0 ? labeled : 0;
  const singletons = campaigns.filter((c) => isCampaignNoise(c)).length;
  return labeledNoise + singletons;
}

function campaignNonNoiseCount(payload, campaigns) {
  const n = Number(payload?.n_campaigns_non_noise);
  if (Number.isFinite(n) && n >= 0) return n;
  return campaigns.filter((c) => !isCampaignNoise(c)).length;
}

function renderCampaignSummary() {
  const el = document.getElementById("campaignSummary");
  if (!el) return;
  const payload = currentPayload();
  const campaigns = payload?.campaigns;
  if (!campaigns?.length) {
    el.textContent = "";
    el.hidden = true;
    return;
  }
  const nCampaigns = campaignNonNoiseCount(payload, campaigns);
  const noise = campaignNoiseTotal(payload, campaigns);
  const nonNoise = campaigns.filter((c) => !isCampaignNoise(c));
  const forAvg = nonNoise.length ? nonNoise : campaigns;
  const sizes = forAvg.map(campaignEmailCount);
  const avg = sizes.reduce((a, b) => a + b, 0) / sizes.length;
  el.hidden = false;
  el.textContent = `${nCampaigns} campaigns · ${noise} noise · avg size ${avg.toFixed(1)}`;
}

function renderCampaigns() {
  const ul = document.getElementById("campaignList");
  ul.innerHTML = "";
  renderCampaignSummary();
  const payload = currentPayload();
  if (!payload || !payload.campaigns) {
    ul.innerHTML = '<li class="muted">No campaigns</li>';
    return;
  }
  campaignsBySizeDesc(payload.campaigns).forEach((c) => {
    const li = document.createElement("li");
    const id = c.id;
    const btn = document.createElement("button");
    const count = campaignEmailCount(c);
    const noise = isCampaignNoise(c);
    const metrics = campaignEvalMetricsSuffix(c);
    const noiseTag = noise ? " · noise" : "";
    btn.textContent = `Campaign ${id} (${count})${noiseTag}${metrics}`;
    if (noise) {
      btn.classList.add("campaign-noise");
    }
    const nGt = Number(c.n_eval);
    const gtNote =
      Number.isFinite(nGt) && nGt >= 0 ? ` · ${nGt} with ground truth` : "";
    btn.title = noise
      ? `Noise (singleton): campaign ${id}, 1 email${gtNote}`
      : metrics
        ? `Campaign ${id}: ${count} emails${gtNote} · H homogeneity · C completeness · V v-measure`
        : `Campaign ${id}: ${count} emails${gtNote}`;
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
    const row = getEmailRow(eid);
    const subj = row?.subject || eid;
    const subjShort = subj.length > 56 ? subj.slice(0, 53) + "…" : subj;
    const isGt = hasGroundTruth(eid, row);
    if (isGt) {
      btn.classList.add("email-gt");
      const gtLabel = row?.ground_truth_label;
      btn.textContent =
        gtLabel != null ? `GT ${gtLabel} · ${subjShort}` : `GT · ${subjShort}`;
      btn.title =
        gtLabel != null
          ? `${eid} · ground truth campaign ${gtLabel}`
          : `${eid} · ground truth labeled`;
    } else {
      btn.textContent = subjShort;
      btn.title = eid;
    }
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

function getEmailRow(eid) {
  const emails = state.data?.emails;
  if (!emails) return null;
  if (emails[eid]) return emails[eid];
  const key = String(eid);
  return emails[key] || null;
}

function refangUrl(text) {
  return String(text)
    .replace(/\bhxxps:\/\//gi, "https://")
    .replace(/\bhxxp:\/\//gi, "http://");
}

function defangUrl(text) {
  return String(text)
    .replace(/\bhttps:\/\//gi, "hxxps://")
    .replace(/\bhttp:\/\//gi, "hxxp://")
    .replace(/\bftp:\/\//gi, "fxp://")
    .replace(/\bmailto:/gi, "mailt0:");
}

/** Defanged key aligned with ``url_similarity`` in data.json (refang then defang). */
function canonicalUrlKey(url) {
  return defangUrl(refangUrl(String(url).trim()));
}

function extractUrlsFromText(text) {
  if (!text) return [];
  const re = /(?:https?:\/\/|hxxps?:\/\/|www\.)[^\s'"<>]+/gi;
  const out = [];
  const seen = new Set();
  for (const m of refangUrl(text).matchAll(re)) {
    const u = refangUrl(m[0].replace(/[.,;:!?)]+$/, ""));
    if (!u || seen.has(u)) continue;
    seen.add(u);
    out.push(u);
  }
  return out;
}

function expandUrlValue(raw) {
  if (raw == null) return [];
  if (Array.isArray(raw)) {
    return raw.flatMap((item) => expandUrlValue(item));
  }
  const text = String(raw).trim();
  if (!text) return [];
  if (text.startsWith("[") && text.endsWith("]")) {
    try {
      const parsed = JSON.parse(text.replace(/'/g, '"'));
      if (Array.isArray(parsed)) {
        return parsed.flatMap((item) => expandUrlValue(item));
      }
    } catch (_) {
      /* fall through to regex extraction */
    }
  }
  return extractUrlsFromText(text);
}

function emailUrlsForRow(e) {
  if (!e) return [];
  const seen = new Set();
  const out = [];
  const add = (raw) => {
    for (const u of expandUrlValue(raw)) {
      const norm = canonicalUrlKey(u);
      if (!norm || seen.has(norm)) continue;
      seen.add(norm);
      out.push(norm);
    }
  };
  for (const u of e.urls || []) add(u);
  for (const u of e.email_urls || []) add(u);
  if (e.body) add(e.body);
  if (e.email_info) add(e.email_info);
  return out.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

function parseScore(raw) {
  if (raw == null) return null;
  if (typeof raw === "number" && !Number.isNaN(raw)) return Number(raw);
  if (typeof raw === "object" && raw.score != null && !Number.isNaN(Number(raw.score))) {
    return Number(raw.score);
  }
  return null;
}

function scoreToBg(score) {
  if (score == null || Number.isNaN(Number(score))) return "transparent";
  const t = Math.max(0, Math.min(1, Number(score)));
  const hue = t * 120;
  return `hsl(${hue}, 88%, 78%)`;
}

function simBadge(score) {
  if (score == null || Number.isNaN(Number(score))) return "";
  return ` <span class="sim-badge" title="Similarity to other members in this campaign (SBERT, scaled per campaign)">${(Number(score) * 100).toFixed(0)}%</span>`;
}

function attrScore(sim, key) {
  if (!sim) return null;
  return parseScore(sim[key]);
}

function similarityFor(eid) {
  const sol = state.solutionKey;
  const cid = state.campaignId;
  if (sol == null || cid == null || !state.data?.attribute_similarity) return null;
  return state.data.attribute_similarity[sol]?.[String(cid)]?.[eid] || null;
}

function urlScore(url) {
  const sol = state.solutionKey;
  const cid = state.campaignId;
  if (sol == null || cid == null || !state.data?.url_similarity) return null;
  const block = state.data.url_similarity[sol]?.[String(cid)];
  if (!block) return null;
  const u = String(url).trim();
  if (!u) return null;
  const candidates = [canonicalUrlKey(u), u, refangUrl(u), defangUrl(u)];
  const tried = new Set();
  for (const key of candidates) {
    if (!key || tried.has(key)) continue;
    tried.add(key);
    const s = parseScore(block[key]);
    if (s != null) return s;
  }
  return null;
}

function urlsAttrRow(label, urls) {
  const body = urls.length
    ? `<ul class="url-list">${urls
        .map((u) => {
          const score = urlScore(u);
          const bg = scoreToBg(score);
          const badge = simBadge(score);
          return `<li class="defanged-url url-item" style="background:${bg}">${esc(u)}${badge}</li>`;
        })
        .join("")}</ul>`
    : `<span class="muted">No URLs in MISP attributes or email body for this message.</span>`;
  return `<div class="row"><span class="label">${esc(label)}</span><br/>${body}</div>`;
}

function attrRow(label, value, score) {
  const bg = scoreToBg(score);
  const badge = simBadge(score);
  return `<div class="row attr-row" style="background:${bg}"><span class="label">${esc(
    label,
  )}${badge}</span><br/>${esc(value)}</div>`;
}

function renderEmailDetail(eid) {
  const e = getEmailRow(eid);
  const el = document.getElementById("emailDetail");

  if (!e) {
    el.innerHTML = `<p class="muted">No email in catalog for <code>${esc(
      eid,
    )}</code>. Regenerate <code>visualization/data.json</code> from the same MISP file used for this run.</p>`;
    return;
  }
  const toList = Array.isArray(e.receivers)
    ? e.receivers.join(", ")
    : e.receivers || "";
  const fromList = Array.isArray(e.senders)
    ? e.senders.join(", ")
    : e.senders || "";
  const sim = similarityFor(eid);
  const s = (k) => attrScore(sim, k);
  const bodyBg = scoreToBg(s("body"));
  const emailUrls = emailUrlsForRow(e);

  const gtLabel = e.ground_truth_label;
  const gtBanner =
    gtLabel != null
      ? `<div class="row gt-banner"><span class="gt-badge">Ground truth</span> campaign <strong>${esc(
          gtLabel,
        )}</strong></div>`
      : hasGroundTruth(eid, e)
        ? `<div class="row gt-banner"><span class="gt-badge">Ground truth</span></div>`
        : "";

  el.innerHTML = `
    ${gtBanner}
    <div class="row"><span class="label">External ID</span><br/>${esc(e.external_id || eid)}</div>
    ${attrRow("From", fromList, s("senders"))}
    ${attrRow("To", toList, s("receivers"))}
    ${attrRow("Date", e.date, s("date"))}
    ${attrRow("Subject", e.subject, s("subject"))}
    ${urlsAttrRow("URLs", emailUrls)}
    <div class="row"><span class="label">Body</span>${simBadge(s("body"))}</div>
    <pre class="body" style="background:${bodyBg}">${esc(e.body)}</pre>
  `;
}

loadData().catch((err) => {
  document.body.innerHTML = `<p style="padding:2rem;color:#c00">Failed to load: ${esc(
    err.message,
  )}</p>`;
});
