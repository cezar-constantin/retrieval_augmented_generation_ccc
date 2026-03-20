import { documentationCopy, demoDocuments, suggestedQuestions } from "./content.js";

const DOC_PALETTE = ["#1a35a8", "#2f9db7", "#c51049", "#7ea51f", "#ff5a0a", "#5c6788"];
const PREVIEW_EMBED_DIMENSIONS = 24;
const FALLBACK_EMBED_DIMENSIONS = 192;

const state = {
  pendingFiles: [],
  documents: [],
  paragraphs: [],
  processing: false,
  pipelineDirty: false,
  topK: 3,
  minParagraphLength: 50,
  retrieval: null,
  embeddingEngine: {
    mode: "idle",
    label: "MiniLM on demand",
    pipeline: null,
    progress: "",
  },
  projectionAxes: null,
};

const elements = {
  heroEyebrow: document.getElementById("hero-eyebrow"),
  heroTitle: document.getElementById("hero-title"),
  heroText: document.getElementById("hero-text"),
  encoderStatus: document.getElementById("encoder-status"),
  documentCount: document.getElementById("document-count"),
  paragraphCount: document.getElementById("paragraph-count"),
  pipelineStatus: document.getElementById("pipeline-status"),
  simulatorCopy: document.getElementById("simulator-copy"),
  fileInput: document.getElementById("file-input"),
  loadDemoButton: document.getElementById("load-demo-button"),
  runPipelineButton: document.getElementById("run-pipeline-button"),
  resetButton: document.getElementById("reset-button"),
  topKRange: document.getElementById("top-k-range"),
  topKValue: document.getElementById("top-k-value"),
  minLengthRange: document.getElementById("min-length-range"),
  minLengthValue: document.getElementById("min-length-value"),
  queryInput: document.getElementById("query-input"),
  askButton: document.getElementById("ask-button"),
  suggestionRow: document.getElementById("suggestion-row"),
  answerTitle: document.getElementById("answer-title"),
  answerCopy: document.getElementById("answer-copy"),
  answerSources: document.getElementById("answer-sources"),
  stepCards: Array.from(document.querySelectorAll(".step-card")),
  stepCopyExtracting: document.getElementById("step-copy-extracting"),
  stepCopySplitting: document.getElementById("step-copy-splitting"),
  stepCopyEmbeddings: document.getElementById("step-copy-embeddings"),
  stepCopyRetrieval: document.getElementById("step-copy-retrieval"),
  stepCopyGeneration: document.getElementById("step-copy-generation"),
  semanticMap: document.getElementById("semantic-map"),
  rankingList: document.getElementById("ranking-list"),
  extractionLiveGrid: document.getElementById("extraction-live-grid"),
  paragraphBoard: document.getElementById("paragraph-board"),
  splittingSummaryPill: document.getElementById("splitting-summary-pill"),
  embeddingDetailsGrid: document.getElementById("embedding-details-grid"),
  embeddingGrid: document.getElementById("embedding-grid"),
  retrievalQueryTitle: document.getElementById("retrieval-query-title"),
  retrievalQueryCopy: document.getElementById("retrieval-query-copy"),
  retrievalResults: document.getElementById("retrieval-results"),
  tabButtons: Array.from(document.querySelectorAll(".tab-button")),
  tabPanels: Array.from(document.querySelectorAll(".tab-panel")),
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function createId(prefix, suffix) {
  return `${prefix}-${suffix}-${Math.random().toString(36).slice(2, 7)}`;
}

function summarizeText(text, maxLength = 220) {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength).trimEnd()}...`;
}

function normalizeExtractedText(text) {
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/\u0000/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function inferSourceLabel(name) {
  const lower = name.toLowerCase();
  if (lower.endsWith(".pdf")) return "PDF";
  if (lower.endsWith(".docx")) return "DOCX";
  if (lower.endsWith(".md")) return "Markdown";
  if (lower.endsWith(".txt")) return "Text";
  return "Document";
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/gi, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2);
}

function splitIntoSentences(text) {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length >= 30);
}

function cosineSimilarity(left, right) {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }
  if (leftNorm === 0 || rightNorm === 0) {
    return 0;
  }
  return dot / Math.sqrt(leftNorm * rightNorm);
}

function normalizeVector(vector) {
  let norm = 0;
  for (let index = 0; index < vector.length; index += 1) {
    norm += vector[index] * vector[index];
  }
  norm = Math.sqrt(norm);
  if (norm <= 1e-9) {
    return Array.from(vector, () => 0);
  }
  return Array.from(vector, (value) => value / norm);
}

function fnv1a(token, seed = 2166136261) {
  let hash = seed >>> 0;
  for (let index = 0; index < token.length; index += 1) {
    hash ^= token.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function buildFallbackEmbedding(text) {
  const vector = new Float32Array(FALLBACK_EMBED_DIMENSIONS);
  const tokens = tokenize(text);
  if (!tokens.length) {
    return Array.from(vector);
  }

  const counts = new Map();
  for (const token of tokens) {
    counts.set(token, (counts.get(token) || 0) + 1);
  }

  for (const [token, count] of counts.entries()) {
    const weight = 1 + Math.log(count);
    const hashA = fnv1a(token);
    const hashB = fnv1a(token, 1315423911);
    const hashC = fnv1a(token, 2654435761);

    const indexA = hashA % FALLBACK_EMBED_DIMENSIONS;
    const indexB = hashB % FALLBACK_EMBED_DIMENSIONS;
    const indexC = hashC % FALLBACK_EMBED_DIMENSIONS;

    vector[indexA] += (hashA & 1 ? 1 : -1) * weight;
    vector[indexB] += (hashB & 1 ? 0.7 : -0.7) * weight;
    vector[indexC] += (hashC & 1 ? 0.4 : -0.4) * weight;
  }

  return normalizeVector(vector);
}

function buildProjectionAxes(dimensions) {
  const axisX = new Float32Array(dimensions);
  const axisY = new Float32Array(dimensions);

  for (let index = 0; index < dimensions; index += 1) {
    axisX[index] = Math.sin(index * 1.71 + 0.37);
    axisY[index] = Math.cos(index * 2.19 + 1.11);
  }

  return {
    axisX: normalizeVector(axisX),
    axisY: normalizeVector(axisY),
  };
}

function projectVector(vector) {
  if (!state.projectionAxes || state.projectionAxes.axisX.length !== vector.length) {
    state.projectionAxes = buildProjectionAxes(vector.length);
  }

  const { axisX, axisY } = state.projectionAxes;
  let x = 0;
  let y = 0;
  for (let index = 0; index < vector.length; index += 1) {
    x += vector[index] * axisX[index];
    y += vector[index] * axisY[index];
  }
  return { x, y };
}

function getDocColor(index) {
  return DOC_PALETTE[index % DOC_PALETTE.length];
}

function setStepState(stepName, status) {
  const card = elements.stepCards.find((candidate) => candidate.dataset.step === stepName);
  if (!card) {
    return;
  }

  card.classList.toggle("is-active", status === "active");
  card.classList.toggle("is-done", status === "done");
}

function renderBulletList(container, bullets) {
  container.innerHTML = bullets
    .map((bullet) => `<div class="bullet-item">${escapeHtml(bullet)}</div>`)
    .join("");
}

function renderDocumentationContent() {
  const { hero, simulatorNote, tabs } = documentationCopy;
  elements.heroEyebrow.textContent = hero.eyebrow;
  elements.heroTitle.textContent = hero.title;
  elements.heroText.textContent = hero.text;
  elements.simulatorCopy.textContent = tabs.simulator.copy;
  const simulatorNoteNode = document.getElementById("simulator-note");
  if (simulatorNoteNode) {
    simulatorNoteNode.textContent = simulatorNote;
  }

  const extraction = tabs.extracting;
  const extractingKicker = document.getElementById("extracting-kicker");
  extractingKicker.textContent = extraction.kicker;
  extractingKicker.hidden = !extraction.kicker;
  document.getElementById("extracting-title").textContent = extraction.title;
  document.getElementById("extracting-copy").textContent = extraction.copy;
  renderBulletList(document.getElementById("extracting-bullets"), extraction.bullets);
  document.getElementById("extracting-code").textContent = extraction.code;

  const splitting = tabs.splitting;
  document.getElementById("splitting-kicker").textContent = splitting.kicker;
  document.getElementById("splitting-title").textContent = splitting.title;
  document.getElementById("splitting-copy").textContent = splitting.copy;
  renderBulletList(document.getElementById("splitting-bullets"), splitting.bullets);
  document.getElementById("splitting-code").textContent = splitting.code;

  const embeddings = tabs.embeddings;
  document.getElementById("embeddings-kicker").textContent = embeddings.kicker;
  document.getElementById("embeddings-title").textContent = embeddings.title;
  document.getElementById("embeddings-copy").textContent = embeddings.copy;
  renderBulletList(document.getElementById("embeddings-bullets"), embeddings.bullets);
  document.getElementById("embeddings-code").textContent = embeddings.code;

  elements.embeddingDetailsGrid.innerHTML = embeddings.details
    .map(
      (detail) => `
        <article class="detail-card">
          <p class="card-kicker">Additional details</p>
          <h3>${escapeHtml(detail.title)}</h3>
          <p>${escapeHtml(detail.text)}</p>
        </article>
      `,
    )
    .join("");

  const retrieval = tabs.retrieval;
  document.getElementById("retrieval-kicker").textContent = retrieval.kicker;
  document.getElementById("retrieval-title").textContent = retrieval.title;
  document.getElementById("retrieval-copy").textContent = retrieval.copy;
  renderBulletList(document.getElementById("retrieval-bullets"), retrieval.bullets);
  document.getElementById("retrieval-formula-title").textContent = retrieval.formulaTitle;
  document.getElementById("retrieval-formula").textContent = retrieval.formula;
  document.getElementById("retrieval-code").textContent = retrieval.code;
}

function setupTabs() {
  elements.tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.tabTarget;
      elements.tabButtons.forEach((candidate) => {
        const isActive = candidate === button;
        candidate.classList.toggle("is-active", isActive);
        candidate.setAttribute("aria-selected", String(isActive));
      });

      elements.tabPanels.forEach((panel) => {
        const isActive = panel.dataset.tabPanel === target;
        panel.hidden = !isActive;
        panel.classList.toggle("is-active", isActive);
      });
    });
  });
}

function setupSuggestions() {
  elements.suggestionRow.innerHTML = "";
  for (const suggestion of suggestedQuestions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "suggestion-chip";
    button.textContent = suggestion;
    button.addEventListener("click", () => {
      elements.queryInput.value = suggestion;
      elements.queryInput.focus();
    });
    elements.suggestionRow.appendChild(button);
  }
}

function updateEncoderStatus() {
  let label = state.embeddingEngine.label;
  if (state.embeddingEngine.progress) {
    label = `${label} - ${state.embeddingEngine.progress}`;
  }
  elements.encoderStatus.textContent = label;
}

function updateHeroMetrics() {
  if (state.documents.length) {
    elements.documentCount.textContent = `${state.documents.length} loaded`;
  } else if (state.pendingFiles.length) {
    elements.documentCount.textContent = `${state.pendingFiles.length} selected`;
  } else {
    elements.documentCount.textContent = "0 loaded";
  }
  elements.paragraphCount.textContent = `${state.paragraphs.length} indexed`;
  elements.topKValue.textContent = String(state.topK);
  elements.minLengthValue.textContent = `${state.minParagraphLength} chars`;
  elements.splittingSummaryPill.textContent = `Threshold: ${state.minParagraphLength} chars`;
  updateEncoderStatus();
}

function updateProcessingControls() {
  const disabled = state.processing;
  elements.runPipelineButton.disabled = disabled;
  elements.askButton.disabled = disabled;
  elements.loadDemoButton.disabled = disabled;
  elements.resetButton.disabled = disabled;
}

function setPipelineStatus(text) {
  elements.pipelineStatus.textContent = text;
}

function markPipelineDirty(reason = "Documents changed, run the pipeline again") {
  state.pipelineDirty = true;
  state.retrieval = null;
  setPipelineStatus(reason);
  setStepState("extracting", "active");
  setStepState("splitting", "");
  setStepState("embeddings", "");
  setStepState("retrieval", "");
  setStepState("generation", "");
  renderRetrievalOutputs();
}

function renderAnswerState(title, copy, sources = []) {
  elements.answerTitle.textContent = title;
  elements.answerCopy.textContent = copy;
  elements.answerSources.innerHTML = sources
    .map((source) => `<span class="source-tag">${escapeHtml(source)}</span>`)
    .join("");
}

function renderExtractionGrid() {
  if (!state.documents.length && !state.pendingFiles.length) {
    elements.extractionLiveGrid.innerHTML = `<div class="empty-message">Load documents to inspect the extracted text.</div>`;
    return;
  }

  const items = state.documents.length
    ? state.documents.map((document, index) => ({
        title: document.name,
        status: document.sourceLabel,
        preview: summarizeText(document.text || "No text extracted yet.", 320),
        stats: [
          `${document.text.length.toLocaleString()} chars`,
          `${document.rawParagraphCandidates.length} raw splits`,
          `${document.paragraphs.length} kept chunks`,
        ],
        color: getDocColor(index),
      }))
    : state.pendingFiles.map((entry, index) => ({
        title: entry.name,
        status: entry.sourceLabel,
        preview: entry.kind === "demo" ? summarizeText(entry.text, 320) : "Waiting for extraction...",
        stats: [entry.kind === "demo" ? "Demo content already loaded" : "Waiting for pipeline run"],
        color: getDocColor(index),
      }));

  elements.extractionLiveGrid.innerHTML = items
    .map(
      (item) => `
        <article class="document-card">
          <div class="document-card-header">
            <div>
              <p class="card-kicker">Document</p>
              <h3>${escapeHtml(item.title)}</h3>
            </div>
            <span class="doc-stat" style="color:${item.color}; border-color:${item.color}33;">${escapeHtml(item.status)}</span>
          </div>
          <div class="document-preview">${escapeHtml(item.preview)}</div>
          <div class="document-stats">
            ${item.stats.map((stat) => `<span class="doc-stat">${escapeHtml(stat)}</span>`).join("")}
          </div>
        </article>
      `,
    )
    .join("");
}

function renderParagraphBoard() {
  if (!state.documents.length) {
    elements.paragraphBoard.innerHTML = `<div class="empty-message">Run the pipeline to inspect the paragraph chunks.</div>`;
    return;
  }

  const cards = [];
  for (const document of state.documents) {
    const previewParagraphs = document.paragraphs.slice(0, 3);
    previewParagraphs.forEach((paragraph) => {
      cards.push(`
        <article class="paragraph-card ${paragraph.isRetrieved ? "is-highlighted" : ""}">
          <div class="paragraph-card-header">
            <div>
              <p class="card-kicker">Chunk ${paragraph.indexInDoc + 1}</p>
              <h3>${escapeHtml(document.name)}</h3>
            </div>
            <span class="doc-stat">${paragraph.charCount} chars</span>
          </div>
          <div class="paragraph-preview">${escapeHtml(paragraph.text)}</div>
          <div class="paragraph-tags">
            <span class="doc-stat">${document.rawParagraphCandidates.length} raw candidates</span>
            <span class="doc-stat">${document.paragraphs.length} kept in index</span>
          </div>
        </article>
      `);
    });
  }

  elements.paragraphBoard.innerHTML = cards.join("");
}

function buildEmbeddingPreview(vector) {
  if (!Array.isArray(vector) || !vector.length) {
    return new Array(PREVIEW_EMBED_DIMENSIONS).fill(0);
  }

  const sliceSize = Math.max(1, Math.floor(vector.length / PREVIEW_EMBED_DIMENSIONS));
  const preview = [];

  for (let index = 0; index < PREVIEW_EMBED_DIMENSIONS; index += 1) {
    let total = 0;
    let count = 0;
    const start = index * sliceSize;
    const end = Math.min(vector.length, start + sliceSize);
    for (let cursor = start; cursor < end; cursor += 1) {
      total += Math.abs(vector[cursor]);
      count += 1;
    }
    preview.push(count ? total / count : 0);
  }

  const maxValue = Math.max(...preview, 1e-6);
  return preview.map((value) => Math.round((value / maxValue) * 100));
}

function hasEmbeddingVector(value) {
  return Array.isArray(value) && value.length > 0;
}

function renderEmbeddingGrid() {
  if (!state.paragraphs.length) {
    elements.embeddingGrid.innerHTML = `<div class="empty-message">Run the pipeline to preview paragraph embeddings.</div>`;
    return;
  }

  const cards = [];
  const selectedParagraphs = state.retrieval ? state.retrieval.topResults : state.paragraphs.slice(0, 6);
  const uniqueParagraphs = [];
  const seen = new Set();
  for (const paragraph of selectedParagraphs.concat(state.paragraphs.slice(0, 6))) {
    if (!paragraph || seen.has(paragraph.id) || !hasEmbeddingVector(paragraph.embedding)) {
      continue;
    }
    seen.add(paragraph.id);
    uniqueParagraphs.push(paragraph);
    if (uniqueParagraphs.length >= 6) {
      break;
    }
  }

  if (!uniqueParagraphs.length && !state.retrieval?.queryEmbedding) {
    elements.embeddingGrid.innerHTML =
      `<div class="empty-message">Embeddings are being computed. Preview cards will appear as vectors become available.</div>`;
    return;
  }

  uniqueParagraphs.forEach((paragraph) => {
    const preview = buildEmbeddingPreview(paragraph.embedding);
    cards.push(`
      <article class="embedding-card ${paragraph.isRetrieved ? "is-highlighted" : ""}">
        <div class="embedding-card-header">
          <div>
            <p class="card-kicker">Paragraph embedding</p>
            <h3>${escapeHtml(paragraph.docName)}</h3>
          </div>
          <span class="doc-stat">${paragraph.charCount} chars</span>
        </div>
        <p>${escapeHtml(summarizeText(paragraph.text, 150))}</p>
        <div class="embedding-preview">
          ${preview
            .map((value) => `<span class="embedding-bar" style="--level:${value}%;"></span>`)
            .join("")}
        </div>
      </article>
    `);
  });

  if (state.retrieval?.queryEmbedding) {
    const queryPreview = buildEmbeddingPreview(state.retrieval.queryEmbedding);
    cards.unshift(`
      <article class="embedding-card is-highlighted">
        <div class="embedding-card-header">
          <div>
            <p class="card-kicker">Query embedding</p>
            <h3>${escapeHtml(state.retrieval.query)}</h3>
          </div>
          <span class="score-badge">Query</span>
        </div>
        <p>The question is encoded with the same model so its vector can be compared directly against every stored paragraph.</p>
        <div class="embedding-preview">
          ${queryPreview.map((value) => `<span class="embedding-bar" style="--level:${value}%;"></span>`).join("")}
        </div>
      </article>
    `);
  }

  elements.embeddingGrid.innerHTML = cards.join("");
}

function renderSemanticMap() {
  if (!state.paragraphs.length) {
    elements.semanticMap.innerHTML = `<div class="semantic-map-empty">Paragraph points will appear here after the pipeline runs.</div>`;
    return;
  }

  const points = state.paragraphs
    .filter((paragraph) => hasEmbeddingVector(paragraph.embedding))
    .map((paragraph) => ({
      ...paragraph,
      projection: projectVector(paragraph.embedding),
    }));

  if (!points.length) {
    elements.semanticMap.innerHTML =
      `<div class="semantic-map-empty">Embeddings are being computed. The semantic map will appear once vectors are ready.</div>`;
    return;
  }

  if (state.retrieval?.queryEmbedding) {
    points.push({
      id: "query-point",
      docIndex: -1,
      projection: projectVector(state.retrieval.queryEmbedding),
      isQuery: true,
      label: "Query",
      isRetrieved: true,
    });
  }

  const xs = points.map((point) => point.projection.x);
  const ys = points.map((point) => point.projection.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  elements.semanticMap.innerHTML = points
    .map((point) => {
      const left = 8 + ((point.projection.x - minX) / Math.max(maxX - minX, 1e-6)) * 84;
      const top = 8 + ((point.projection.y - minY) / Math.max(maxY - minY, 1e-6)) * 84;
      const isRetrieved = point.isRetrieved;
      const color = point.isQuery ? undefined : getDocColor(point.docIndex);
      const glow = isRetrieved ? "0 0 0 8px rgba(79, 185, 209, 0.18)" : "0 0 0 0 rgba(23, 37, 90, 0.08)";
      return `
        <div
          class="map-point ${point.isQuery ? "is-query" : ""}"
          style="
            left:${left}%;
            top:${top}%;
            background:${point.isQuery ? "" : color};
            box-shadow:${glow};
            opacity:${isRetrieved || point.isQuery ? 1 : 0.78};
          "
          title="${escapeHtml(point.isQuery ? state.retrieval.query : point.docName)}"
        >
          ${point.isQuery ? `<span class="map-point-label">Query</span>` : ""}
        </div>
      `;
    })
    .join("");
}

function renderRankingList() {
  if (!state.retrieval?.topResults.length) {
    elements.rankingList.innerHTML = `<div class="empty-message">Run a query to populate the ranking list.</div>`;
    return;
  }

  elements.rankingList.innerHTML = state.retrieval.topResults
    .map((result) => {
      const percentage = Math.max(0, Math.round(result.score * 100));
      return `
        <article class="ranking-row">
          <div>
            <h4>${escapeHtml(result.docName)}</h4>
            <p>${escapeHtml(summarizeText(result.text, 130))}</p>
            <div class="bar-track">
              <div class="bar-fill" style="width:${percentage}%;"></div>
            </div>
          </div>
          <strong>${percentage}%</strong>
        </article>
      `;
    })
    .join("");
}

function renderRetrievalOutputs() {
  renderRankingList();
  renderSemanticMap();
  renderEmbeddingGrid();

  if (!state.retrieval?.topResults.length) {
    elements.retrievalQueryTitle.textContent = "No query yet";
    elements.retrievalQueryCopy.textContent =
      "Submit a question to see how the query embedding is compared against every paragraph.";
    elements.retrievalResults.innerHTML = `<div class="empty-message">Retrieved passages will appear here after a query.</div>`;
    return;
  }

  elements.retrievalQueryTitle.textContent = state.retrieval.query;
  elements.retrievalQueryCopy.textContent =
    "The query was encoded with the same embedding model and ranked against all paragraph embeddings using cosine similarity.";
  elements.retrievalResults.innerHTML = state.retrieval.topResults
    .map((result, index) => {
      const percentage = Math.max(0, Math.round(result.score * 100));
      return `
        <article class="retrieval-card is-highlighted">
          <div class="retrieval-card-header">
            <div>
              <p class="card-kicker">Rank ${index + 1}</p>
              <h3>${escapeHtml(result.docName)}</h3>
            </div>
            <span class="score-badge">${percentage}%</span>
          </div>
          <div class="retrieval-preview">${escapeHtml(result.text)}</div>
          <div class="paragraph-tags">
            <span class="doc-stat">Chunk ${result.indexInDoc + 1}</span>
            <span class="doc-stat">${result.charCount} chars</span>
          </div>
        </article>
      `;
    })
    .join("");
}

function refreshRetrievedFlags() {
  const retrievedIds = new Set(state.retrieval?.topResults.map((result) => result.id) || []);
  state.paragraphs.forEach((paragraph) => {
    paragraph.isRetrieved = retrievedIds.has(paragraph.id);
  });
  state.documents.forEach((document) => {
    document.paragraphs.forEach((paragraph) => {
      paragraph.isRetrieved = retrievedIds.has(paragraph.id);
    });
  });
}

function buildGroundedAnswer(query, topResults) {
  const queryTokens = new Set(tokenize(query));
  const candidateSentences = [];

  for (const result of topResults) {
    const sentences = splitIntoSentences(result.text);
    for (const sentence of sentences) {
      const tokens = tokenize(sentence);
      let overlap = 0;
      for (const token of tokens) {
        if (queryTokens.has(token)) {
          overlap += 1;
        }
      }
      const score = overlap * 0.12 + result.score;
      candidateSentences.push({
        text: sentence,
        source: result.docName,
        score,
      });
    }
  }

  candidateSentences.sort((left, right) => right.score - left.score);
  const picked = [];
  const seen = new Set();
  for (const sentence of candidateSentences) {
    const signature = sentence.text.toLowerCase();
    if (seen.has(signature)) {
      continue;
    }
    seen.add(signature);
    picked.push(sentence);
    if (picked.length >= 3) {
      break;
    }
  }

  if (!picked.length) {
    return {
      title: "Evidence retrieved, but synthesis stayed conservative",
      copy:
        "The system found relevant passages, but there was not enough clear sentence-level evidence to compose a short grounded answer.",
      sources: topResults.map((result) => result.docName),
    };
  }

  const joined = picked.map((item) => item.text).join(" ");
  const sources = Array.from(new Set(topResults.map((result) => result.docName)));
  return {
    title: "Grounded synthesis from retrieved passages",
    copy: joined,
    sources,
  };
}

function clearPipelineResults() {
  state.retrieval = null;
  refreshRetrievedFlags();
  renderAnswerState(
    "No answer yet",
    "Upload or load documents, run the pipeline, and submit a question to generate a grounded answer.",
    [],
  );
  renderRetrievalOutputs();
  renderParagraphBoard();
}

async function loadRemoteScript(url) {
  if (document.querySelector(`script[data-external-script="${url}"]`)) {
    return;
  }

  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.async = true;
    script.dataset.externalScript = url;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Could not load script: ${url}`));
    document.head.appendChild(script);
  });
}

async function extractTextFromPdf(file) {
  const pdfjsLib = await import("https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.min.mjs");
  pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.worker.min.mjs";

  const bytes = new Uint8Array(await file.arrayBuffer());
  const pdf = await pdfjsLib.getDocument({ data: bytes }).promise;
  const pages = [];
  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber);
    const textContent = await page.getTextContent();
    const pageText = textContent.items.map((item) => item.str).join(" ");
    pages.push(pageText);
  }
  return normalizeExtractedText(pages.join("\n"));
}

async function extractTextFromDocx(file) {
  await loadRemoteScript("https://cdn.jsdelivr.net/npm/mammoth@1.8.0/mammoth.browser.min.js");
  const { value } = await window.mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() });
  return normalizeExtractedText(value);
}

async function extractTextFromPendingEntry(entry) {
  if (entry.kind === "demo") {
    return normalizeExtractedText(entry.text);
  }

  const { file } = entry;
  const lower = file.name.toLowerCase();
  if (lower.endsWith(".txt") || lower.endsWith(".md")) {
    return normalizeExtractedText(await file.text());
  }
  if (lower.endsWith(".pdf")) {
    return extractTextFromPdf(file);
  }
  if (lower.endsWith(".docx")) {
    return extractTextFromDocx(file);
  }
  return normalizeExtractedText(await file.text());
}

function splitDocumentIntoParagraphs(text, minLength) {
  const rawCandidates = text
    .split(/\n+/)
    .map((paragraph) => paragraph.replace(/\s+/g, " ").trim())
    .filter(Boolean);
  const kept = rawCandidates.filter((paragraph) => paragraph.length >= minLength);
  return { rawCandidates, kept };
}

function tensorToVector(output) {
  if (output?.data) {
    return normalizeVector(output.data);
  }
  if (Array.isArray(output)) {
    return normalizeVector(output.flat());
  }
  if (typeof output?.tolist === "function") {
    return normalizeVector(output.tolist().flat());
  }
  return normalizeVector([]);
}

async function ensureEmbeddingEngine() {
  if (state.embeddingEngine.pipeline || state.embeddingEngine.mode === "fallback") {
    return;
  }

  state.embeddingEngine.mode = "loading";
  state.embeddingEngine.label = "Loading browser embedding model";
  state.embeddingEngine.progress = "";
  updateHeroMetrics();

  try {
    const { pipeline, env } = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.0");
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    if (env.backends?.onnx?.wasm) {
      env.backends.onnx.wasm.numThreads = 1;
    }

    state.embeddingEngine.pipeline = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
      quantized: true,
      progress_callback: (event) => {
        if (event.status && Number.isFinite(event.progress)) {
          state.embeddingEngine.progress = `${event.status} ${Math.round(event.progress)}%`;
          updateHeroMetrics();
        }
      },
    });

    state.embeddingEngine.mode = "transformer";
    state.embeddingEngine.label = "MiniLM loaded in browser";
    state.embeddingEngine.progress = "";
  } catch (error) {
    console.warn("Falling back to hashed embeddings.", error);
    state.embeddingEngine.mode = "fallback";
    state.embeddingEngine.label = "Deterministic fallback vectors";
    state.embeddingEngine.progress = "";
  }

  updateHeroMetrics();
}

async function embedText(text) {
  await ensureEmbeddingEngine();

  if (state.embeddingEngine.pipeline) {
    const output = await state.embeddingEngine.pipeline(text, {
      pooling: "mean",
      normalize: true,
    });
    return tensorToVector(output);
  }

  return buildFallbackEmbedding(text);
}

async function nextFrame() {
  await new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function buildDocumentsFromPendingEntries(processedEntries) {
  return processedEntries.map((entry, index) => {
    const { rawCandidates, kept } = splitDocumentIntoParagraphs(entry.text, state.minParagraphLength);
    return {
      id: createId("doc", index),
      name: entry.name,
      sourceLabel: entry.sourceLabel,
      text: entry.text,
      docIndex: index,
      rawParagraphCandidates: rawCandidates,
      paragraphs: kept.map((paragraphText, paragraphIndex) => ({
        id: createId("paragraph", `${index}-${paragraphIndex}`),
        docId: index,
        docIndex: index,
        docName: entry.name,
        sourceLabel: entry.sourceLabel,
        text: paragraphText,
        charCount: paragraphText.length,
        indexInDoc: paragraphIndex,
        embedding: null,
        isRetrieved: false,
      })),
    };
  });
}

async function runPipeline() {
  if (!state.pendingFiles.length) {
    setPipelineStatus("Add documents or load the demo set first");
    renderAnswerState("No documents selected", "Choose a set of files before running the RAG pipeline.", []);
    return;
  }

  state.processing = true;
  updateProcessingControls();
  clearPipelineResults();

  try {
    setPipelineStatus("Processing selected documents");
    setStepState("extracting", "active");
    setStepState("splitting", "");
    setStepState("embeddings", "");
    setStepState("retrieval", "");
    setStepState("generation", "");

    const extractedEntries = [];
    for (let index = 0; index < state.pendingFiles.length; index += 1) {
      const entry = state.pendingFiles[index];
      setStepState("extracting", "active");
      const text = await extractTextFromPendingEntry(entry);
      extractedEntries.push({
        name: entry.name,
        sourceLabel: entry.sourceLabel,
        text,
      });
      await nextFrame();
    }

    setStepState("extracting", "done");

    setPipelineStatus("Preparing document chunks");
    setStepState("splitting", "active");

    state.documents = buildDocumentsFromPendingEntries(extractedEntries);
    state.paragraphs = state.documents.flatMap((document) => document.paragraphs);
    renderExtractionGrid();
    renderParagraphBoard();
    updateHeroMetrics();
    await nextFrame();

    setStepState("splitting", "done");

    setPipelineStatus("Preparing document representations");
    setStepState("embeddings", "active");

    for (let index = 0; index < state.paragraphs.length; index += 1) {
      const paragraph = state.paragraphs[index];
      paragraph.embedding = await embedText(paragraph.text);
      setStepState("embeddings", "active");
      if ((index + 1) % 3 === 0 || index === state.paragraphs.length - 1) {
        renderEmbeddingGrid();
        await nextFrame();
      }
    }

    setStepState("embeddings", "done");
    setStepState("retrieval", "");
    setStepState("generation", "");

    state.pipelineDirty = false;
    setPipelineStatus("Pipeline ready for querying");
    updateHeroMetrics();
    renderSemanticMap();
    renderEmbeddingGrid();
  } catch (error) {
    console.error(error);
    setPipelineStatus("Pipeline failed");
    renderAnswerState(
      "Processing error",
      "One of the document-processing steps failed. If the issue comes from a PDF or DOCX parser, try a TXT or Markdown file as a fallback.",
      [],
    );
  } finally {
    state.processing = false;
    updateProcessingControls();
    renderExtractionGrid();
    renderParagraphBoard();
  }
}

function getTopResults(sortedResults) {
  return sortedResults.slice(0, state.topK);
}

function refreshRetrievalViews() {
  if (!state.retrieval) {
    renderRetrievalOutputs();
    return;
  }

  state.retrieval.topResults = getTopResults(state.retrieval.sortedResults);
  refreshRetrievedFlags();
  const answer = buildGroundedAnswer(state.retrieval.query, state.retrieval.topResults);
  renderAnswerState(answer.title, answer.copy, answer.sources);
  setStepState("retrieval", "done");
  setStepState("generation", "done");
  renderParagraphBoard();
  renderRetrievalOutputs();
}

async function runRetrieval() {
  const query = elements.queryInput.value.trim();
  if (!query) {
    renderAnswerState("No question entered", "Type a question before asking the documents.", []);
    return;
  }

  if (!state.paragraphs.length || state.pipelineDirty) {
    await runPipeline();
    if (!state.paragraphs.length) {
      return;
    }
  }

  state.processing = true;
  updateProcessingControls();
    setPipelineStatus("Searching the selected documents");
  setStepState("retrieval", "active");
  setStepState("generation", "active");

  try {
    const queryEmbedding = await embedText(query);
    const sortedResults = state.paragraphs
      .map((paragraph) => ({
        ...paragraph,
        score: clamp(cosineSimilarity(queryEmbedding, paragraph.embedding), 0, 1),
      }))
      .sort((left, right) => right.score - left.score);

    state.retrieval = {
      query,
      queryEmbedding,
      sortedResults,
      topResults: [],
    };
    refreshRetrievalViews();
    setPipelineStatus("Answer generated from retrieved evidence");
  } catch (error) {
    console.error(error);
    renderAnswerState(
      "Retrieval error",
      "The query could not be encoded. Try processing the documents again or use shorter inputs.",
      [],
    );
    setPipelineStatus("Retrieval failed");
  } finally {
    state.processing = false;
    updateProcessingControls();
  }
}

function resetAll() {
  state.pendingFiles = [];
  state.documents = [];
  state.paragraphs = [];
  state.pipelineDirty = false;
  state.retrieval = null;
  elements.fileInput.value = "";
  elements.queryInput.value = "";
  state.projectionAxes = null;

  setPipelineStatus("Waiting for documents");
  setStepState("extracting", "");
  setStepState("splitting", "");
  setStepState("embeddings", "");
  setStepState("retrieval", "");
  setStepState("generation", "");

  renderAnswerState(
    "No answer yet",
    "Upload or load documents, run the pipeline, and submit a question to generate a grounded answer.",
    [],
  );
  renderExtractionGrid();
  renderParagraphBoard();
  renderRetrievalOutputs();
  updateHeroMetrics();
}

function addPendingEntries(entries, autoRun = false) {
  state.pendingFiles = entries;
  state.documents = [];
  state.paragraphs = [];
  state.retrieval = null;
  state.projectionAxes = null;
  markPipelineDirty("Documents loaded. Run the pipeline to rebuild the RAG index.");
  renderExtractionGrid();
  renderParagraphBoard();
  renderRetrievalOutputs();
  updateHeroMetrics();
  if (autoRun) {
    void runPipeline();
  }
}

function handleFileSelection() {
  const files = Array.from(elements.fileInput.files || []);
  if (!files.length) {
    return;
  }

  const entries = files.map((file) => ({
    kind: "file",
    file,
    name: file.name,
    sourceLabel: inferSourceLabel(file.name),
  }));
  addPendingEntries(entries);
}

function loadDemoSet() {
  const entries = demoDocuments.map((document) => ({
    kind: "demo",
    name: document.name,
    sourceLabel: document.sourceLabel,
    text: document.text,
  }));
  addPendingEntries(entries, true);
}

function attachEvents() {
  elements.fileInput.addEventListener("change", handleFileSelection);
  elements.loadDemoButton.addEventListener("click", loadDemoSet);
  elements.runPipelineButton.addEventListener("click", () => {
    void runPipeline();
  });
  elements.askButton.addEventListener("click", () => {
    void runRetrieval();
  });
  elements.resetButton.addEventListener("click", resetAll);
  elements.topKRange.addEventListener("input", () => {
    state.topK = Number(elements.topKRange.value);
    updateHeroMetrics();
    if (state.retrieval) {
      refreshRetrievalViews();
    }
  });
  elements.minLengthRange.addEventListener("input", () => {
    state.minParagraphLength = Number(elements.minLengthRange.value);
    updateHeroMetrics();
    if (state.pendingFiles.length) {
      markPipelineDirty(`Paragraph threshold changed to ${state.minParagraphLength} characters.`);
    }
  });
}

function initialize() {
  renderDocumentationContent();
  setupTabs();
  setupSuggestions();
  attachEvents();
  resetAll();
}

initialize();
