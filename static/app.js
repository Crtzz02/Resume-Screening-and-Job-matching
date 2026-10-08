const form = document.getElementById("scan-form");
const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const fileLabel = document.getElementById("file-label");
const pasteToggle = document.getElementById("paste-toggle");
const pasteBlock = document.getElementById("paste-block");
const resumeText = document.getElementById("resume-text");
const modeSelect = document.getElementById("mode");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const resultsMeta = document.getElementById("results-meta");
const bestMatchEl = document.getElementById("best-match");
const scoreComparisonEl = document.getElementById("score-comparison");
const jobListEl = document.getElementById("job-list");
const rankListEl = document.getElementById("rank-list");
const entityBlockEl = document.getElementById("entity-block");
const catalogGrid = document.getElementById("catalog-grid");

let selectedFile = null;
let pasteMode = false;

function setStatus(message, isError = false) {
  statusEl.textContent = message || "";
  statusEl.style.color = isError ? "#9b1c1c" : "var(--highlight)";
}

function pct(score) {
  return `${Math.round((score || 0) * 100)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

pasteToggle.addEventListener("click", () => {
  pasteMode = !pasteMode;
  pasteBlock.classList.toggle("hidden", !pasteMode);
  pasteToggle.textContent = pasteMode ? "Use file upload" : "Paste text instead";
  if (pasteMode) {
    selectedFile = null;
    fileInput.value = "";
    fileLabel.textContent = "Using pasted text";
  } else {
    fileLabel.textContent = "Drop resume here";
  }
});

function assignFile(file) {
  if (!file) return;
  selectedFile = file;
  pasteMode = false;
  pasteBlock.classList.add("hidden");
  pasteToggle.textContent = "Paste text instead";
  fileLabel.textContent = file.name;
  setStatus(`Ready: ${file.name}`);
}

fileInput.addEventListener("change", () => {
  assignFile(fileInput.files?.[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  assignFile(file);
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const mode = modeSelect.value;

  if (pasteMode) {
    const text = resumeText.value.trim();
    if (text.length < 40) {
      setStatus("Paste a fuller resume (at least a few sentences).", true);
      return;
    }
    await runTextMatch(text, mode);
    return;
  }

  if (!selectedFile) {
    setStatus("Choose a resume file first.", true);
    return;
  }
  await runUploadMatch(selectedFile, mode);
});

async function runUploadMatch(file, mode) {
  const body = new FormData();
  body.append("file", file);
  body.append("mode", mode);
  setBusy(true, "Scanning resume against fields and jobs…");
  try {
    const response = await fetch("/api/v1/match/fields/upload", {
      method: "POST",
      body,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Upload matching failed");
    }
    renderResults(payload);
    setStatus("Done.");
  } catch (error) {
    setStatus(error.message || "Something went wrong.", true);
  } finally {
    setBusy(false);
  }
}

async function runTextMatch(text, mode) {
  setBusy(true, "Scanning resume against fields and jobs…");
  try {
    const response = await fetch("/api/v1/match/fields", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Text matching failed");
    }
    renderResults(payload);
    setStatus("Done.");
  } catch (error) {
    setStatus(error.message || "Something went wrong.", true);
  } finally {
    setBusy(false);
  }
}

function setBusy(isBusy, message = "") {
  submitBtn.disabled = isBusy;
  setStatus(message);
}

function renderScoreComparison(comparison, best) {
  const c = comparison || {
    tfidf: best?.tfidf_score,
    semantic: best?.semantic_score,
    hybrid: best?.hybrid_score ?? best?.score,
    note: "",
  };
  const scores = [
    { key: "tfidf", label: "TF-IDF keywords", value: c.tfidf },
    { key: "semantic", label: "Semantic (SBERT)", value: c.semantic },
    { key: "hybrid", label: "Hybrid", value: c.hybrid },
  ];
  const maxVal = Math.max(...scores.map((s) => s.value || 0), 0);
  scoreComparisonEl.innerHTML = `
    <div class="compare-grid">
      ${scores
        .map((s) => {
          const win = (s.value || 0) >= maxVal - 0.0001 && maxVal > 0;
          return `
          <div class="compare-card ${win ? "winner" : ""}">
            <p class="label">${s.label}</p>
            <div class="value">${pct(s.value)}</div>
            <div class="bar"><i style="width:${Math.max(4, Math.round((s.value || 0) * 100))}%"></i></div>
          </div>`;
        })
        .join("")}
    </div>
    <p class="compare-note">${escapeHtml(c.note || "")}</p>
  `;
}

function renderJobs(jobs) {
  if (!jobs || !jobs.length) {
    jobListEl.innerHTML =
      "<p class='section-lede'>No job index found. Run <code>python scripts/build_fields_from_jds.py</code> to generate recommended jobs.</p>";
    return;
  }
  jobListEl.innerHTML = jobs
    .map((job, index) => {
      return `
      <div class="job-row" style="animation-delay:${index * 40}ms">
        <div>
          <strong>${escapeHtml(job.title)}</strong>
          <div class="meta">${escapeHtml(job.field_title || "")} · ${escapeHtml(job.category || "")}</div>
          <p class="snippet">${escapeHtml(job.snippet || "")}</p>
          <div class="meta" style="margin-top:.4rem">
            TF-IDF ${pct(job.tfidf_score)} · Semantic ${pct(job.semantic_score)}
          </div>
        </div>
        <div><strong>${pct(job.score)}</strong></div>
      </div>`;
    })
    .join("");
}

function renderResults(data) {
  resultsEl.classList.remove("hidden");
  const best = data.best_match;
  const source = data.filename ? `File: ${data.filename}` : "Pasted text";
  resultsMeta.textContent = `${source} · Ranking mode: ${data.mode}`;

  if (!best) {
    bestMatchEl.innerHTML = `<div class="panel"><p>No fields available to rank.</p></div>`;
    scoreComparisonEl.innerHTML = "";
    jobListEl.innerHTML = "";
    rankListEl.innerHTML = "";
    entityBlockEl.innerHTML = "";
    return;
  }

  const matched = (best.matched_skills || [])
    .map((skill) => `<span class="pill">${escapeHtml(skill)}</span>`)
    .join("");
  const missing = (best.missing_skills || [])
    .slice(0, 8)
    .map((skill) => `<span class="pill miss">${escapeHtml(skill)}</span>`)
    .join("");

  bestMatchEl.innerHTML = `
    <div class="panel">
      <p class="cat" style="margin:0 0 .35rem;color:var(--highlight);font-size:.8rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;">${escapeHtml(best.category)}</p>
      <h3>${escapeHtml(best.title)}</h3>
      <p>${escapeHtml(best.summary)}</p>
      <div class="pill-row">${matched || "<span class='pill'>No exact skill hits</span>"}</div>
      ${missing ? `<p style="margin:1rem 0 .35rem;font-weight:600;font-size:.9rem;">Skills to grow</p><div class="pill-row">${missing}</div>` : ""}
    </div>
    <div class="panel">
      <div class="score">${pct(best.score)}<span>Ranked by ${escapeHtml(data.mode)} mode</span></div>
      <p style="margin:1rem 0 0;color:var(--ink-soft);font-size:.92rem;">
        Overlap terms: ${(best.matched_keywords || []).slice(0, 10).map(escapeHtml).join(", ") || "n/a"}
      </p>
    </div>
  `;

  renderScoreComparison(data.score_comparison, best);
  renderJobs(data.recommended_jobs);

  rankListEl.innerHTML = (data.rankings || [])
    .map((row, index) => {
      const width = Math.max(4, Math.round((row.score || 0) * 100));
      return `
        <div class="rank-row" style="animation-delay:${index * 40}ms">
          <div class="idx">${index + 1}</div>
          <div>
            <strong>${escapeHtml(row.title)}</strong>
            <div style="color:var(--ink-soft);font-size:.85rem;">
              ${escapeHtml(row.category)} · TF-IDF ${pct(row.tfidf_score)} · Sem ${pct(row.semantic_score)} · Hyb ${pct(row.hybrid_score ?? row.score)}
            </div>
            <div class="bar"><i style="width:${width}%"></i></div>
          </div>
          <div><strong>${pct(row.score)}</strong></div>
        </div>
      `;
    })
    .join("");

  const entities = data.entities || {};
  entityBlockEl.innerHTML = [
    ["Organizations", entities.organizations],
    ["Locations", entities.locations],
    ["Tech skills (NER)", entities.tech_skills],
  ]
    .map(
      ([label, values]) => `
      <div class="panel">
        <strong>${label}</strong>
        <div class="pill-row" style="margin-top:.65rem;">
          ${(values || []).length
            ? values
                .slice(0, 8)
                .map((v) => `<span class="pill">${escapeHtml(v)}</span>`)
                .join("")
            : `<span class="pill">None detected</span>`}
        </div>
      </div>`
    )
    .join("");

  resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadCatalog() {
  try {
    const response = await fetch("/api/v1/fields");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Could not load fields");
    const fields = payload.fields || [];
    catalogGrid.innerHTML = fields
      .map(
        (field) => `
        <article class="catalog-card">
          <div class="cat">${escapeHtml(field.category)}</div>
          <h3>${escapeHtml(field.title)}</h3>
          <p>${escapeHtml(field.summary)}</p>
        </article>`
      )
      .join("");

    const batchField = document.getElementById("batch-field");
    if (batchField) {
      batchField.innerHTML = fields
        .map(
          (field) =>
            `<option value="${escapeHtml(field.id)}">${escapeHtml(field.title)}</option>`
        )
        .join("");
    }
  } catch (error) {
    catalogGrid.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
  }
}

loadCatalog();

// --- Recruiter batch screening ---
const batchForm = document.getElementById("batch-form");
const batchFilesInput = document.getElementById("batch-files");
const batchDropzone = document.getElementById("batch-dropzone");
const batchFileLabel = document.getElementById("batch-file-label");
const batchTargetType = document.getElementById("batch-target-type");
const batchFieldWrap = document.getElementById("batch-field-wrap");
const batchJdWrap = document.getElementById("batch-jd-wrap");
const batchJd = document.getElementById("batch-jd");
const batchMode = document.getElementById("batch-mode");
const batchSubmit = document.getElementById("batch-submit");
const batchStatus = document.getElementById("batch-status");
const batchResults = document.getElementById("batch-results");
const batchResultsMeta = document.getElementById("batch-results-meta");
const batchRankList = document.getElementById("batch-rank-list");

let batchFiles = [];

function setBatchStatus(message, isError = false) {
  batchStatus.textContent = message || "";
  batchStatus.style.color = isError ? "#9b1c1c" : "var(--highlight)";
}

batchTargetType.addEventListener("change", () => {
  const useJd = batchTargetType.value === "jd";
  batchFieldWrap.classList.toggle("hidden", useJd);
  batchJdWrap.classList.toggle("hidden", !useJd);
});

function assignBatchFiles(fileList) {
  batchFiles = Array.from(fileList || []).slice(0, 40);
  if (!batchFiles.length) {
    batchFileLabel.textContent = "Drop multiple resumes here";
    return;
  }
  batchFileLabel.textContent =
    batchFiles.length === 1
      ? batchFiles[0].name
      : `${batchFiles.length} resumes selected`;
  setBatchStatus(`Ready: ${batchFiles.length} file(s)`);
}

batchFilesInput.addEventListener("change", () => {
  assignBatchFiles(batchFilesInput.files);
});

["dragenter", "dragover"].forEach((eventName) => {
  batchDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    batchDropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  batchDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    batchDropzone.classList.remove("dragover");
  });
});

batchDropzone.addEventListener("drop", (event) => {
  assignBatchFiles(event.dataTransfer?.files);
});

batchDropzone.addEventListener("click", () => batchFilesInput.click());
batchDropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    batchFilesInput.click();
  }
});

function statusClass(label) {
  if (label === "Strong Match") return "strong";
  if (label === "Potential Match") return "potential";
  return "low";
}

batchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!batchFiles.length) {
    setBatchStatus("Select at least one resume.", true);
    return;
  }

  const body = new FormData();
  batchFiles.forEach((file) => body.append("files", file));
  body.append("mode", batchMode.value);

  if (batchTargetType.value === "jd") {
    const jd = batchJd.value.trim();
    if (jd.length < 40) {
      setBatchStatus("Paste a fuller job description.", true);
      return;
    }
    body.append("job_description", jd);
  } else {
    body.append("field_id", document.getElementById("batch-field").value);
  }

  batchSubmit.disabled = true;
  setBatchStatus(`Ranking ${batchFiles.length} resume(s)…`);

  try {
    const response = await fetch("/api/v1/match/batch", {
      method: "POST",
      body,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Batch matching failed");
    }
    renderBatchResults(payload);
    setBatchStatus("Done.");
  } catch (error) {
    setBatchStatus(error.message || "Something went wrong.", true);
  } finally {
    batchSubmit.disabled = false;
  }
});

function renderBatchResults(data) {
  batchResults.classList.remove("hidden");
  batchResultsMeta.textContent = `Target: ${data.target_label} · ${data.resume_count} ranked · Engine: ${data.mode}${
    data.error_count ? ` · ${data.error_count} skipped` : ""
  }`;

  batchRankList.innerHTML = (data.rankings || [])
    .map((row, index) => {
      const width = Math.max(4, Math.round((row.score || 0) * 100));
      const skills = (row.matched_skills || [])
        .slice(0, 6)
        .map((s) => `<span class="pill">${escapeHtml(s)}</span>`)
        .join("");
      return `
        <div class="rank-row" style="animation-delay:${index * 40}ms">
          <div class="idx">${row.rank || index + 1}</div>
          <div>
            <strong>${escapeHtml(row.filename)}</strong>
            <div style="color:var(--ink-soft);font-size:.85rem;">
              TF-IDF ${pct(row.tfidf_score)} · Sem ${pct(row.semantic_score)} · Hyb ${pct(row.hybrid_score)}
            </div>
            <span class="status-pill ${statusClass(row.status)}">${escapeHtml(row.status || "")}</span>
            ${skills ? `<div class="pill-row">${skills}</div>` : ""}
            <div class="bar"><i style="width:${width}%"></i></div>
          </div>
          <div><strong>${pct(row.score)}</strong></div>
        </div>`;
    })
    .join("");

  batchResults.scrollIntoView({ behavior: "smooth", block: "start" });
}