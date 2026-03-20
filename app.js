import { documentationCopy, demoDocuments, suggestedQuestions } from "./content.js";

const DOC_PALETTE = ["#1a35a8", "#2f9db7", "#c51049", "#7ea51f", "#ff5a0a", "#5c6788"];
const PREVIEW_EMBED_DIMENSIONS = 24;
const FALLBACK_EMBED_DIMENSIONS = 192;
const EXTRACTION_VIEWER_TARGET_CHARS = 900;
const EXTRACTION_VIEWER_MIN_CHARS = 420;
const TEACHING_TOKEN_LIMIT = 28;
const MOBILE_CHART_BREAKPOINT = 768;
const MOBILE_BACKGROUND_POINT_LIMIT = 20;
const MOBILE_TOKEN_POINT_LIMIT = 14;
const CONTENT_STOPWORDS = new Set([
  "the",
  "and",
  "for",
  "that",
  "with",
  "from",
  "this",
  "into",
  "when",
  "where",
  "which",
  "while",
  "their",
  "there",
  "they",
  "them",
  "have",
  "has",
  "been",
  "were",
  "your",
  "will",
  "would",
  "than",
  "then",
  "after",
  "before",
  "also",
  "each",
  "only",
  "over",
  "under",
  "between",
  "about",
  "through",
  "using",
  "used",
]);
const teachingProjectionAxesCache = new Map();
const teachingParagraphCache = new Map();

const state = {
  pendingFiles: [],
  documents: [],
  paragraphs: [],
  extractionReference: {
    type: "code",
    documentName: null,
  },
  splittingReference: {
    type: "code",
    documentName: null,
  },
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
  embeddingInspector: {
    stageKey: "tokenization",
    paragraphId: null,
  },
  ui: {
    mobileChartMode: false,
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
  extractingReferenceActions: document.getElementById("extracting-reference-actions"),
  extractingReferenceOutput: document.getElementById("extracting-reference-output"),
  splittingReferenceActions: document.getElementById("splitting-reference-actions"),
  splittingReferenceTitle: document.getElementById("splitting-reference-title"),
  splittingReferenceOutput: document.getElementById("splitting-reference-output"),
  paragraphBoard: document.getElementById("paragraph-board"),
  splittingSummaryPill: document.getElementById("splitting-summary-pill"),
  embeddingDetailsGrid: document.getElementById("embedding-details-grid"),
  embeddingStageDisplay: document.getElementById("embedding-stage-display"),
  embeddingGrid: document.getElementById("embedding-grid"),
  retrievalQueryTitle: document.getElementById("retrieval-query-title"),
  retrievalQueryCopy: document.getElementById("retrieval-query-copy"),
  retrievalResults: document.getElementById("retrieval-results"),
  retrievalChartPanel: document.getElementById("retrieval-chart-panel"),
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

function detectMobileChartMode() {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }

  return (
    window.matchMedia(`(max-width: ${MOBILE_CHART_BREAKPOINT}px)`).matches ||
    window.matchMedia("(pointer: coarse)").matches
  );
}

function sampleEvenly(items, limit) {
  if (items.length <= limit) {
    return items;
  }

  const sampled = [];
  const seenIndexes = new Set();
  for (let index = 0; index < limit; index += 1) {
    const candidateIndex = Math.round((index * (items.length - 1)) / Math.max(limit - 1, 1));
    if (seenIndexes.has(candidateIndex)) {
      continue;
    }
    seenIndexes.add(candidateIndex);
    sampled.push(items[candidateIndex]);
  }

  return sampled;
}

function summarizeText(text, maxLength = 220) {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength).trimEnd()}...`;
}

function splitTextIntoDisplayChunks(text, targetChars = EXTRACTION_VIEWER_TARGET_CHARS, minChars = EXTRACTION_VIEWER_MIN_CHARS) {
  const normalized = normalizeExtractedText(text);
  if (!normalized) {
    return [];
  }

  const chunks = [];
  let cursor = 0;

  while (cursor < normalized.length) {
    const remaining = normalized.length - cursor;
    if (remaining <= targetChars) {
      chunks.push(normalized.slice(cursor).trim());
      break;
    }

    const preferredEnd = Math.min(cursor + targetChars, normalized.length);
    let splitPoint = normalized.lastIndexOf("\n\n", preferredEnd);

    if (splitPoint <= cursor + minChars) {
      splitPoint = normalized.lastIndexOf("\n", preferredEnd);
    }
    if (splitPoint <= cursor + minChars) {
      splitPoint = normalized.lastIndexOf(". ", preferredEnd);
      if (splitPoint > cursor + minChars) {
        splitPoint += 1;
      }
    }
    if (splitPoint <= cursor + minChars) {
      splitPoint = normalized.lastIndexOf(" ", preferredEnd);
    }
    if (splitPoint <= cursor) {
      splitPoint = preferredEnd;
    }

    chunks.push(normalized.slice(cursor, splitPoint).trim());
    cursor = splitPoint;

    while (cursor < normalized.length && /\s/.test(normalized[cursor])) {
      cursor += 1;
    }
  }

  return chunks.filter(Boolean);
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

function normalizeParagraphCandidate(text) {
  return String(text)
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .split(/\n+/)
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .join(" ")
    .replace(/\s+([,.;:!?%)\]}])/g, "$1")
    .replace(/([(\[{])\s+/g, "$1")
    .trim();
}

function splitIntoParagraphCandidates(text) {
  const normalized = normalizeExtractedText(text);
  if (!normalized) {
    return [];
  }

  let candidates = normalized.split(/\n\s*\n+/);
  if (candidates.length <= 1) {
    candidates = normalized.split(/\n+/);
  }

  return candidates.map((candidate) => normalizeParagraphCandidate(candidate)).filter(Boolean);
}

function isLikelyPdfPageNumber(text) {
  const normalized = normalizeParagraphCandidate(text);
  return /^\d{1,4}$/.test(normalized) || /^[ivxlcdm]{1,8}$/i.test(normalized);
}

function isLikelyPdfSectionHeading(text) {
  const normalized = normalizeParagraphCandidate(text);
  if (!normalized || isLikelyPdfPageNumber(normalized) || normalized.length > 120) {
    return false;
  }

  if (/^\d+(?:\.\d+)*\s+\S/.test(normalized)) {
    return true;
  }

  return /^(abstract|introduction|conclusion|summary|overview|appendix|references)$/i.test(normalized);
}

function startsLikePdfContinuation(text) {
  const normalized = normalizeParagraphCandidate(text);
  if (!normalized) {
    return false;
  }

  return (
    /^[a-z(]/.test(normalized) ||
    /^(and|or|but|because|since|which|where|when|while|with|within|without|for|to|of|in|on|by|from|the|this|these|those|such|process|thereby|thus|therefore)\b/i.test(
      normalized
    )
  );
}

function endsLikePdfContinuation(text) {
  const normalized = normalizeParagraphCandidate(text);
  if (!normalized) {
    return false;
  }

  return !/[.!?]"?$/.test(normalized) || /[:;,]$/.test(normalized);
}

function shouldMergePdfParagraphAcrossPage(previousText, nextText) {
  const previous = normalizeParagraphCandidate(previousText);
  const next = normalizeParagraphCandidate(nextText);
  if (!previous || !next) {
    return false;
  }

  if (
    isLikelyPdfPageNumber(previous) ||
    isLikelyPdfPageNumber(next) ||
    isLikelyPdfSectionHeading(previous) ||
    isLikelyPdfSectionHeading(next)
  ) {
    return false;
  }

  return previous.length >= 80 && next.length >= 40 && endsLikePdfContinuation(previous) && startsLikePdfContinuation(next);
}

function mergePdfParagraphs(previousText, nextText) {
  const previous = normalizeParagraphCandidate(previousText);
  const next = normalizeParagraphCandidate(nextText);
  if (!previous) {
    return next;
  }
  if (!next) {
    return previous;
  }

  const separator = endsLikePdfContinuation(previous) ? " " : "\n";
  return normalizeParagraphCandidate(`${previous}${separator}${next}`);
}

function postProcessPdfParagraphs(pageParagraphGroups) {
  const flattened = [];

  for (const pageParagraphs of pageParagraphGroups) {
    for (const paragraph of pageParagraphs) {
      const normalized = normalizeParagraphCandidate(paragraph);
      if (!normalized || isLikelyPdfPageNumber(normalized)) {
        continue;
      }

      if (
        flattened.length &&
        shouldMergePdfParagraphAcrossPage(flattened[flattened.length - 1], normalized)
      ) {
        flattened[flattened.length - 1] = mergePdfParagraphs(flattened[flattened.length - 1], normalized);
        continue;
      }

      flattened.push(normalized);
    }
  }

  const mergedHeadings = [];
  for (let index = 0; index < flattened.length; index += 1) {
    const current = flattened[index];
    const next = flattened[index + 1];

    if (
      isLikelyPdfSectionHeading(current) &&
      next &&
      !isLikelyPdfSectionHeading(next) &&
      !isLikelyPdfPageNumber(next)
    ) {
      mergedHeadings.push(normalizeParagraphCandidate(`${current}\n${next}`));
      index += 1;
      continue;
    }

    mergedHeadings.push(current);
  }

  return mergedHeadings;
}

function joinPdfTokens(tokens) {
  let text = "";
  for (const rawToken of tokens) {
    const token = String(rawToken ?? "").replace(/\s+/g, " ").trim();
    if (!token) {
      continue;
    }
    if (!text) {
      text = token;
      continue;
    }

    const attachToPrevious = /^[,.;:!?%)\]}]/.test(token);
    const attachToNext = /[(\[{/"'-]$/.test(text);
    text += attachToPrevious || attachToNext ? token : ` ${token}`;
  }

  return text
    .replace(/\s+([,.;:!?%)\]}])/g, "$1")
    .replace(/([(\[{])\s+/g, "$1")
    .trim();
}

function buildPdfLines(textItems) {
  const lines = [];
  let currentTokens = [];
  let currentY = null;
  let currentXStart = 0;
  let currentXEnd = 0;
  let currentHeight = 0;

  const flushLine = () => {
    const text = joinPdfTokens(currentTokens);
    if (text) {
      lines.push({
        text,
        y: currentY ?? 0,
        xStart: currentXStart,
        xEnd: currentXEnd,
        height: currentHeight || 12,
      });
    }
    currentTokens = [];
    currentY = null;
    currentXStart = 0;
    currentXEnd = 0;
    currentHeight = 0;
  };

  for (const item of textItems) {
    const token = String(item?.str ?? "");
    const hasVisibleText = token.replace(/\s+/g, "").length > 0;
    const y = Number(item?.transform?.[5] ?? 0);
    const x = Number(item?.transform?.[4] ?? 0);
    const width = Math.abs(Number(item?.width ?? 0));
    const height = Math.abs(Number(item?.height ?? item?.transform?.[3] ?? 12)) || 12;

    const startsNewLine =
      currentTokens.length &&
      Math.abs(y - currentY) > Math.max(2, Math.min(currentHeight || height, height) * 0.45);

    if (startsNewLine) {
      flushLine();
    }

    if (hasVisibleText) {
      if (!currentTokens.length) {
        currentY = y;
        currentXStart = x;
        currentXEnd = x + width;
        currentHeight = height;
      } else {
        currentXStart = Math.min(currentXStart, x);
        currentXEnd = Math.max(currentXEnd, x + width);
        currentHeight = Math.max(currentHeight, height);
      }
      currentTokens.push(token);
    }

    if (item?.hasEOL && currentTokens.length) {
      flushLine();
    }
  }

  flushLine();
  return lines;
}

function buildPdfParagraphs(lines) {
  if (!lines.length) {
    return [];
  }

  const minXStart = Math.min(...lines.map((line) => line.xStart));
  const averageWidth =
    lines.reduce((sum, line) => sum + Math.max(line.xEnd - line.xStart, 1), 0) / lines.length;

  const paragraphs = [];
  let currentParagraph = lines[0].text;
  let previousLine = lines[0];

  for (let index = 1; index < lines.length; index += 1) {
    const line = lines[index];
    const verticalGap = Math.abs(previousLine.y - line.y);
    const lineHeight = Math.max(previousLine.height, line.height, 1);
    const previousWidth = Math.max(previousLine.xEnd - previousLine.xStart, 1);
    const lineIndented = line.xStart - minXStart > Math.max(12, line.height * 0.9);
    const bulletLike = /^[\u2022\-*]/.test(line.text) || /^\d+[.)]\s/.test(line.text);
    const previousShort = previousWidth < averageWidth * 0.72;

    const startsNewParagraph =
      verticalGap > lineHeight * 1.3 ||
      bulletLike ||
      previousShort ||
      (lineIndented && /[.!?:]"?$/.test(previousLine.text));

    if (startsNewParagraph) {
      paragraphs.push(currentParagraph.trim());
      currentParagraph = line.text;
    } else if (currentParagraph.endsWith("-") && !currentParagraph.endsWith(" -")) {
      currentParagraph = `${currentParagraph.slice(0, -1)}${line.text}`;
    } else {
      currentParagraph = `${currentParagraph} ${line.text}`;
    }

    previousLine = line;
  }

  paragraphs.push(currentParagraph.trim());
  return paragraphs.map((paragraph) => normalizeParagraphCandidate(paragraph)).filter(Boolean);
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

function getTeachingProjectionAxes(dimensions) {
  if (!teachingProjectionAxesCache.has(dimensions)) {
    teachingProjectionAxesCache.set(dimensions, buildProjectionAxes(dimensions));
  }
  return teachingProjectionAxesCache.get(dimensions);
}

function projectTeachingVector(vector) {
  const { axisX, axisY } = getTeachingProjectionAxes(vector.length);
  let x = 0;
  let y = 0;
  for (let index = 0; index < vector.length; index += 1) {
    x += vector[index] * axisX[index];
    y += vector[index] * axisY[index];
  }
  return { x, y };
}

function tokenizeForEmbeddingDisplay(text) {
  return (String(text).match(/[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*/g) || []).slice(0, TEACHING_TOKEN_LIMIT);
}

function buildStageSentences(text) {
  const sentences = splitIntoSentences(text);
  if (sentences.length >= 2) {
    return sentences.slice(0, 4);
  }

  const normalized = normalizeExtractedText(text);
  if (!normalized) {
    return [];
  }

  if (normalized.length <= 160) {
    return [normalized];
  }

  const midpoint = Math.floor(normalized.length / 2);
  let splitPoint = normalized.lastIndexOf(", ", midpoint);
  if (splitPoint <= 0) {
    splitPoint = normalized.lastIndexOf(" ", midpoint);
  }
  if (splitPoint <= 0) {
    splitPoint = midpoint;
  }

  return [normalized.slice(0, splitPoint).trim(), normalized.slice(splitPoint).trim()].filter(Boolean);
}

function combineWeightedVectors(vectors, weights = null) {
  if (!vectors.length) {
    return [];
  }

  const combined = new Float32Array(vectors[0].length);
  let totalWeight = 0;

  vectors.forEach((vector, index) => {
    const weight = weights ? weights[index] : 1;
    totalWeight += weight;
    for (let cursor = 0; cursor < vector.length; cursor += 1) {
      combined[cursor] += vector[cursor] * weight;
    }
  });

  if (!totalWeight) {
    return normalizeVector(combined);
  }

  for (let index = 0; index < combined.length; index += 1) {
    combined[index] /= totalWeight;
  }

  return normalizeVector(combined);
}

function mixVectors(left, right, rightWeight = 0.3) {
  if (!left.length) {
    return right;
  }
  if (!right.length) {
    return left;
  }

  const mixed = new Float32Array(left.length);
  const leftWeight = 1 - rightWeight;
  for (let index = 0; index < left.length; index += 1) {
    mixed[index] = left[index] * leftWeight + right[index] * rightWeight;
  }
  return normalizeVector(mixed);
}

function cosineToPercentage(value) {
  return Math.round(clamp((value + 1) * 50, 0, 100));
}

function buildTokenTeachingVector(token, index, tokenCount) {
  const base = buildFallbackEmbedding(`${token.toLowerCase()} ${index}`);
  const adjusted = new Float32Array(base.length);
  const positionalBias = (index - (tokenCount - 1) / 2) / Math.max(tokenCount, 1);

  for (let cursor = 0; cursor < base.length; cursor += 1) {
    adjusted[cursor] =
      base[cursor] +
      Math.sin((cursor + 1) * (index + 1) * 0.017) * 0.08 +
      Math.cos((cursor + 3) * (positionalBias + 1.2)) * 0.03;
  }

  return normalizeVector(adjusted);
}

function buildTokenImportanceScores(tokens) {
  return tokens.map((token, index) => {
    const normalized = token.toLowerCase();
    const stopwordPenalty = CONTENT_STOPWORDS.has(normalized) ? 0.35 : 1;
    const lengthBoost = clamp(token.length / 8, 0.45, 1.4);
    const positionBoost = 1 - Math.abs(index - (tokens.length - 1) / 2) / Math.max(tokens.length, 1) * 0.25;
    return stopwordPenalty * lengthBoost * positionBoost;
  });
}

function buildEmbeddingPreviewParagraphs() {
  const sourceParagraphs = state.retrieval
    ? state.retrieval.topResults.concat(state.paragraphs.slice(0, 6))
    : state.paragraphs.slice(0, 6);
  const uniqueParagraphs = [];
  const seen = new Set();

  for (const paragraph of sourceParagraphs) {
    if (!paragraph || seen.has(paragraph.id) || !hasEmbeddingVector(paragraph.embedding)) {
      continue;
    }
    seen.add(paragraph.id);
    uniqueParagraphs.push(paragraph);
    if (uniqueParagraphs.length >= 6) {
      break;
    }
  }

  return uniqueParagraphs;
}

function ensureSelectedEmbeddingParagraph() {
  const previewParagraphs = buildEmbeddingPreviewParagraphs();
  if (!previewParagraphs.length) {
    state.embeddingInspector.paragraphId = null;
    return null;
  }

  const selected = previewParagraphs.find((paragraph) => paragraph.id === state.embeddingInspector.paragraphId);
  if (selected) {
    return selected;
  }

  state.embeddingInspector.paragraphId = previewParagraphs[0].id;
  return previewParagraphs[0];
}

function setEmbeddingInspectorStage(stageKey) {
  state.embeddingInspector.stageKey = stageKey;
  renderEmbeddingExplorer();
}

function setEmbeddingInspectorParagraph(paragraphId) {
  state.embeddingInspector.paragraphId = paragraphId;
  renderEmbeddingGrid();
  renderEmbeddingExplorer();
}

function buildTeachingParagraphData(paragraph) {
  const cacheKey = `${paragraph.id}:${paragraph.text.length}:${paragraph.charCount}`;
  if (teachingParagraphCache.has(cacheKey)) {
    return teachingParagraphCache.get(cacheKey);
  }

  const tokens = tokenizeForEmbeddingDisplay(paragraph.text);
  const displayTokens = tokens.length ? tokens : ["paragraph"];
  const tokenVectors = displayTokens.map((token, index) =>
    buildTokenTeachingVector(token, index, displayTokens.length),
  );
  const tokenPoints = tokenVectors.map((vector, index) => ({
    index,
    token: displayTokens[index],
    vector,
    projection: projectTeachingVector(vector),
  }));

  const attentionNodes = tokenPoints.map((point, index) => {
    const rankedInfluences = tokenPoints
      .map((candidate, candidateIndex) => {
        if (index === candidateIndex) {
          return null;
        }

        const semanticScore = cosineSimilarity(point.vector, candidate.vector);
        const distanceScore = 1 - Math.abs(index - candidateIndex) / Math.max(tokenPoints.length - 1, 1);
        return {
          index: candidateIndex,
          score: semanticScore * 0.72 + distanceScore * 0.28,
        };
      })
      .filter(Boolean)
      .sort((left, right) => right.score - left.score)
      .slice(0, 3);

    const influenceWeights = rankedInfluences.map((influence) => Math.max(0.08, (influence.score + 1) / 2));
    const influenceVectors = rankedInfluences.map((influence) => tokenPoints[influence.index].vector);
    const influenceBlend = combineWeightedVectors(influenceVectors, influenceWeights);
    const transformedVector = influenceBlend.length ? mixVectors(point.vector, influenceBlend, 0.42) : point.vector;

    return {
      ...point,
      influences: rankedInfluences.map((influence, influenceIndex) => ({
        ...influence,
        weight: influenceWeights[influenceIndex],
        token: tokenPoints[influence.index].token,
      })),
      transformedVector,
      transformedProjection: projectTeachingVector(transformedVector),
    };
  });

  const transformedVectors = attentionNodes.map((node) => node.transformedVector);
  const pooledVector = combineWeightedVectors(transformedVectors);
  const pooledProjection = projectTeachingVector(pooledVector);

  const tokenImportance = buildTokenImportanceScores(displayTokens);
  const contextualEvidenceVector = combineWeightedVectors(transformedVectors, tokenImportance);
  const contextualVector = mixVectors(pooledVector, contextualEvidenceVector, 0.34);

  const sentences = buildStageSentences(paragraph.text);
  const sentenceVectors = sentences.map((sentence) => buildFallbackEmbedding(sentence));
  const sentenceScores = sentenceVectors.map((vector, index) => ({
    text: sentences[index],
    score: clamp((cosineSimilarity(vector, contextualVector) + 1) / 2, 0, 1),
    vector,
  }));
  const sentenceWeights = sentenceScores.map((sentence) => 0.25 + sentence.score);
  const sentenceBlend = sentenceVectors.length
    ? combineWeightedVectors(
        sentenceScores.map((sentence) => sentence.vector),
        sentenceWeights,
      )
    : contextualVector;
  const contrastiveVector = mixVectors(contextualVector, sentenceBlend, 0.28);

  const finalVector = hasEmbeddingVector(paragraph.embedding) ? paragraph.embedding : contrastiveVector;

  const semanticNeighbors = state.paragraphs
    .filter((candidate) => candidate.id !== paragraph.id && hasEmbeddingVector(candidate.embedding))
    .map((candidate) => ({
      ...candidate,
      score: cosineSimilarity(finalVector, candidate.embedding),
      projection: projectVector(candidate.embedding),
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 4);

  const result = {
    tokens: displayTokens,
    tokenPoints,
    attentionNodes,
    pooledVector,
    pooledProjection,
    contextualVector,
    contrastiveVector,
    finalVector,
    tokenImportance,
    sentenceScores,
    semanticNeighbors,
  };

  teachingParagraphCache.set(cacheKey, result);
  return result;
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

  const splitting = tabs.splitting;
  const splittingKicker = document.getElementById("splitting-kicker");
  splittingKicker.textContent = splitting.kicker;
  splittingKicker.hidden = !splitting.kicker;
  document.getElementById("splitting-title").textContent = splitting.title;
  document.getElementById("splitting-copy").textContent = splitting.copy;

  const embeddings = tabs.embeddings;
  const embeddingsKicker = document.getElementById("embeddings-kicker");
  embeddingsKicker.textContent = embeddings.kicker;
  embeddingsKicker.hidden = !embeddings.kicker;
  document.getElementById("embeddings-title").textContent = embeddings.title;
  document.getElementById("embeddings-copy").textContent = embeddings.copy;
  const embeddingsBullets = document.getElementById("embeddings-bullets");
  renderBulletList(embeddingsBullets, embeddings.bullets);
  embeddingsBullets.hidden = !embeddings.bullets.length;
  document.getElementById("embeddings-code").textContent = embeddings.code;

  const retrieval = tabs.retrieval;
  const retrievalKicker = document.getElementById("retrieval-kicker");
  retrievalKicker.textContent = retrieval.kicker;
  retrievalKicker.hidden = !retrieval.kicker;
  document.getElementById("retrieval-title").textContent = retrieval.title;
  document.getElementById("retrieval-copy").textContent = retrieval.copy;
  const retrievalBullets = document.getElementById("retrieval-bullets");
  renderBulletList(retrievalBullets, retrieval.bullets);
  retrievalBullets.hidden = !retrieval.bullets.length;
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
  if (elements.splittingSummaryPill) {
    elements.splittingSummaryPill.textContent = `Threshold: ${state.minParagraphLength} chars`;
  }
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

function setExtractionReference(type, documentName = null) {
  state.extractionReference = { type, documentName };
  renderExtractionReferencePanel();
}

function getDocumentReferenceButtons() {
  return state.documents.length
    ? state.documents.map((document) => ({
        name: document.name,
        sourceLabel: document.sourceLabel,
      }))
    : state.pendingFiles.map((entry) => ({
        name: entry.name,
        sourceLabel: entry.sourceLabel,
      }));
}

function buildExtractedChunkViewer(documentName, chunks, docColor, totalChars) {
  if (!chunks.length) {
    return `<div class="empty-message">No extracted chunks were produced for this document.</div>`;
  }

  return `
    <div class="reference-meta">
      <span class="doc-stat">${escapeHtml(documentName)}</span>
      <span class="doc-stat">${chunks.length} chunk${chunks.length === 1 ? "" : "s"}</span>
      <span class="doc-stat">${totalChars.toLocaleString()} chars</span>
    </div>
    <div class="raw-split-stack">
      ${chunks
        .map(
          (chunk, index) => `
            <section class="raw-split-block" style="--split-accent:${docColor};">
              <div class="raw-split-header">
                <span class="raw-split-index">Chunk ${index + 1}</span>
                <span class="doc-stat">${chunk.length} chars</span>
              </div>
              <p>${escapeHtml(chunk)}</p>
            </section>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderExtractionReferencePanel() {
  if (!elements.extractingReferenceActions || !elements.extractingReferenceOutput) {
    return;
  }

  const buttons = getDocumentReferenceButtons();
  const availableNames = new Set(buttons.map((button) => button.name));
  if (
    state.extractionReference.type === "document" &&
    !availableNames.has(state.extractionReference.documentName)
  ) {
    state.extractionReference = { type: "code", documentName: null };
  }

  elements.extractingReferenceActions.innerHTML = `
    <button
      type="button"
      class="secondary-button reference-button ${state.extractionReference.type === "code" ? "is-active" : ""}"
      data-extract-reference="code"
    >
      Show python code
    </button>
    ${buttons
      .map(
        (button) => `
          <button
            type="button"
            class="secondary-button reference-button ${
              state.extractionReference.type === "document" &&
              state.extractionReference.documentName === button.name
                ? "is-active"
                : ""
            }"
            data-extract-document="${escapeHtml(button.name)}"
          >
            ${escapeHtml(button.name)}
          </button>
        `,
      )
      .join("")}
  `;

  elements.extractingReferenceActions
    .querySelector('[data-extract-reference="code"]')
    ?.addEventListener("click", () => {
      setExtractionReference("code");
    });

  elements.extractingReferenceActions.querySelectorAll("[data-extract-document]").forEach((button) => {
    button.addEventListener("click", () => {
      setExtractionReference("document", button.getAttribute("data-extract-document"));
    });
  });

  if (state.extractionReference.type === "code") {
    elements.extractingReferenceOutput.innerHTML = `
      <pre class="code-block reference-code-block"><code>${escapeHtml(documentationCopy.tabs.extracting.code)}</code></pre>
    `;
    return;
  }

  const selectedDocument = state.documents.find(
    (document) => document.name === state.extractionReference.documentName,
  );

  if (selectedDocument) {
    const normalizedText = normalizeExtractedText(selectedDocument.text);
    elements.extractingReferenceOutput.innerHTML = buildExtractedChunkViewer(
      selectedDocument.name,
      splitTextIntoDisplayChunks(normalizedText),
      getDocColor(selectedDocument.docIndex),
      normalizedText.length,
    );
    return;
  }

  const pendingEntry = state.pendingFiles.find(
    (entry) => entry.name === state.extractionReference.documentName,
  );

  if (pendingEntry?.text) {
    const normalizedText = normalizeExtractedText(pendingEntry.text);
    elements.extractingReferenceOutput.innerHTML = buildExtractedChunkViewer(
      pendingEntry.name,
      splitTextIntoDisplayChunks(normalizedText),
      getDocColor(state.pendingFiles.findIndex((entry) => entry.name === pendingEntry.name)),
      normalizedText.length,
    );
    return;
  }

  elements.extractingReferenceOutput.innerHTML = `
    <div class="empty-message">Run the pipeline to extract this document and inspect the extracted chunks.</div>
  `;
}

function setSplittingReference(type, documentName = null) {
  state.splittingReference = { type, documentName };
  renderSplittingReferencePanel();
}

function buildParagraphSplitViewer(documentName, rawCandidates, docColor) {
  if (!rawCandidates.length) {
    return `<div class="empty-message">No paragraph splits were produced for this document.</div>`;
  }

  const keptCount = rawCandidates.filter((candidate) => candidate.length >= state.minParagraphLength).length;
  return `
    <div class="reference-meta">
      <span class="doc-stat">${escapeHtml(documentName)}</span>
      <span class="doc-stat">${rawCandidates.length} paragraph split${rawCandidates.length === 1 ? "" : "s"}</span>
      <span class="doc-stat">${keptCount} kept at ${state.minParagraphLength}+ chars</span>
    </div>
    <div class="raw-split-stack">
      ${rawCandidates
        .map((split, index) => {
          const isKept = split.length >= state.minParagraphLength;
          return `
            <section class="raw-split-block ${isKept ? "is-kept" : "is-filtered"}" style="--split-accent:${docColor};">
              <div class="raw-split-header">
                <span class="raw-split-index">Paragraph split ${index + 1}</span>
                <span class="doc-stat">${isKept ? "Kept" : "Filtered"}</span>
              </div>
              <p>${escapeHtml(split)}</p>
            </section>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderSplittingReferencePanel() {
  if (
    !elements.splittingReferenceActions ||
    !elements.splittingReferenceTitle ||
    !elements.splittingReferenceOutput
  ) {
    return;
  }

  const buttons = getDocumentReferenceButtons();
  const availableNames = new Set(buttons.map((button) => button.name));
  if (
    state.splittingReference.type === "document" &&
    !availableNames.has(state.splittingReference.documentName)
  ) {
    state.splittingReference = { type: "code", documentName: null };
  }

  elements.splittingReferenceActions.innerHTML = `
    <button
      type="button"
      class="secondary-button reference-button ${state.splittingReference.type === "code" ? "is-active" : ""}"
      data-splitting-reference="code"
    >
      Show python code
    </button>
    ${buttons
      .map(
        (button) => `
          <button
            type="button"
            class="secondary-button reference-button ${
              state.splittingReference.type === "document" &&
              state.splittingReference.documentName === button.name
                ? "is-active"
                : ""
            }"
            data-splitting-document="${escapeHtml(button.name)}"
          >
            ${escapeHtml(button.name)}
          </button>
        `,
      )
      .join("")}
  `;

  elements.splittingReferenceActions
    .querySelector('[data-splitting-reference="code"]')
    ?.addEventListener("click", () => {
      setSplittingReference("code");
    });

  elements.splittingReferenceActions.querySelectorAll("[data-splitting-document]").forEach((button) => {
    button.addEventListener("click", () => {
      setSplittingReference("document", button.getAttribute("data-splitting-document"));
    });
  });

  if (state.splittingReference.type === "code") {
    elements.splittingReferenceTitle.textContent = "Python reference";
    elements.splittingReferenceOutput.innerHTML = `
      <pre class="code-block reference-code-block"><code>${escapeHtml(documentationCopy.tabs.splitting.code)}</code></pre>
    `;
    return;
  }

  const selectedDocument = state.documents.find(
    (document) => document.name === state.splittingReference.documentName,
  );

  if (selectedDocument) {
    const { rawCandidates } = splitDocumentIntoParagraphs(selectedDocument.text, state.minParagraphLength);
    elements.splittingReferenceTitle.textContent = "Python reference";
    elements.splittingReferenceOutput.innerHTML = buildParagraphSplitViewer(
      selectedDocument.name,
      rawCandidates,
      getDocColor(selectedDocument.docIndex),
    );
    return;
  }

  const pendingEntry = state.pendingFiles.find(
    (entry) => entry.name === state.splittingReference.documentName,
  );
  elements.splittingReferenceTitle.textContent = "Python reference";

  if (pendingEntry?.text) {
    const normalizedText = normalizeExtractedText(pendingEntry.text);
    const { rawCandidates } = splitDocumentIntoParagraphs(normalizedText, state.minParagraphLength);
    elements.splittingReferenceOutput.innerHTML = buildParagraphSplitViewer(
      pendingEntry.name,
      rawCandidates,
      getDocColor(state.pendingFiles.findIndex((entry) => entry.name === pendingEntry.name)),
    );
    return;
  }

  elements.splittingReferenceOutput.innerHTML = `
    <div class="empty-message">Run the pipeline to inspect the paragraph splits for this document.</div>
  `;
}

function renderExtractionGrid() {
  renderExtractionReferencePanel();

  if (!elements.extractionLiveGrid) {
    return;
  }

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
  renderSplittingReferencePanel();

  if (!elements.paragraphBoard) {
    return;
  }

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

function buildEmbeddingBarsMarkup(vector, className = "embedding-preview") {
  const preview = buildEmbeddingPreview(vector);
  return `
    <div class="${className}">
      ${preview.map((value) => `<span class="embedding-bar" style="--level:${value}%;"></span>`).join("")}
    </div>
  `;
}

function layoutScatterPoints(points, width = 760, height = 460) {
  if (!points.length) {
    return [];
  }

  const xs = points.map((point) => point.projection.x);
  const ys = points.map((point) => point.projection.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return points.map((point) => ({
    ...point,
    plotX: 76 + ((point.projection.x - minX) / Math.max(maxX - minX, 1e-6)) * (width - 152),
    plotY: 64 + ((point.projection.y - minY) / Math.max(maxY - minY, 1e-6)) * (height - 128),
  }));
}

function buildScatterGuides(width = 760, height = 460) {
  return `
    <rect x="16" y="16" width="${width - 32}" height="${height - 32}" rx="28" class="embedding-plot-frame"></rect>
    <line x1="24" y1="${height / 2}" x2="${width - 24}" y2="${height / 2}" class="embedding-axis-line"></line>
    <line x1="${width / 2}" y1="24" x2="${width / 2}" y2="${height - 24}" class="embedding-axis-line"></line>
  `;
}

function buildTokenScatterSvg(points, options = {}) {
  const mobileMode = options.mobileMode ?? state.ui.mobileChartMode;
  const width = mobileMode ? 680 : 760;
  const height = mobileMode ? 360 : 460;
  const projectionKey = options.projectionKey || "projection";
  let basePoints = points.map((point) => ({
    ...point,
    projection: point[projectionKey],
  }));

  if (mobileMode) {
    basePoints = sampleEvenly(basePoints, options.mobileLimit || MOBILE_TOKEN_POINT_LIMIT);
  }

  const allPoints = options.centerPoint
    ? basePoints.concat([
        {
          ...options.centerPoint,
          projection: options.centerPoint.projection,
          isCenter: true,
        },
      ])
    : basePoints;
  const laidOutPoints = layoutScatterPoints(allPoints, width, height);
  const pointLookup = new Map(
    laidOutPoints.filter((point) => !point.isCenter).map((point) => [point.index, point]),
  );
  const centerPoint = laidOutPoints.find((point) => point.isCenter);

  return `
    <svg class="embedding-stage-plot" viewBox="0 0 ${width} ${height}" aria-label="${escapeHtml(options.ariaLabel || "Embedding stage plot")}">
      ${buildScatterGuides(width, height)}
      ${
        options.showPoolingLinks && centerPoint
          ? laidOutPoints
              .filter((point) => !point.isCenter)
              .map(
                (point) => `
                  <line
                    x1="${point.plotX}"
                    y1="${point.plotY}"
                    x2="${centerPoint.plotX}"
                    y2="${centerPoint.plotY}"
                    class="embedding-merge-line"
                  ></line>
                `,
              )
              .join("")
          : ""
      }
      ${laidOutPoints
        .filter((point) => !point.isCenter)
        .map((point) => {
          const radius = 10 + clamp((point.importance || 0.7) * 5, 0, 6);
          const hoverTitle = options.titleBuilder ? options.titleBuilder(point) : point.token;
          const attentionLinks =
            !mobileMode && options.showAttentionLinks && point.influences
              ? point.influences
                  .map((influence) => {
                    const target = pointLookup.get(influence.index);
                    if (!target) {
                      return "";
                    }
                    return `
                      <line
                        x1="${point.plotX}"
                        y1="${point.plotY}"
                        x2="${target.plotX}"
                        y2="${target.plotY}"
                        class="embedding-attention-link"
                        style="--link-opacity:${clamp(influence.weight, 0.18, 0.95)};"
                      ></line>
                    `;
                  })
                  .join("")
              : "";

          return `
            <g class="embedding-token-node">
              ${attentionLinks ? `<g class="embedding-attention-links">${attentionLinks}</g>` : ""}
              <circle
                cx="${point.plotX}"
                cy="${point.plotY}"
                r="${radius}"
                class="embedding-token-point"
                style="--token-color:${options.pointColor || "var(--blue)"};"
              >
                <title>${escapeHtml(hoverTitle)}</title>
              </circle>
              <text x="${point.plotX}" y="${point.plotY + 4}" class="embedding-token-label">${point.index + 1}</text>
            </g>
          `;
        })
        .join("")}
      ${
        centerPoint
          ? `
            <g class="embedding-center-node">
              <circle
                cx="${centerPoint.plotX}"
                cy="${centerPoint.plotY}"
                r="20"
                class="embedding-center-point"
              ></circle>
              <text x="${centerPoint.plotX}" y="${centerPoint.plotY + 5}" class="embedding-center-label">P</text>
              <text x="${centerPoint.plotX}" y="${centerPoint.plotY + 34}" class="embedding-center-caption">${escapeHtml(
                options.centerPoint.label || "Paragraph vector",
              )}</text>
            </g>
          `
          : ""
      }
    </svg>
  `;
}

function buildSentenceInfluenceMarkup(sentenceScores) {
  if (!sentenceScores.length) {
    return `<div class="empty-message">No sentence-level evidence was available for this paragraph.</div>`;
  }

  return `
    <div class="sentence-influence-list">
      ${sentenceScores
        .map(
          (sentence, index) => `
            <article class="sentence-influence-card">
              <div class="sentence-influence-header">
                <span class="raw-split-index">Sentence ${index + 1}</span>
                <span class="doc-stat">${Math.round(sentence.score * 100)}%</span>
              </div>
              <div class="bar-track">
                <div class="bar-fill" style="width:${Math.round(sentence.score * 100)}%;"></div>
              </div>
              <p>${escapeHtml(sentence.text)}</p>
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function buildSemanticSpaceSvg(selectedParagraph, teachingData) {
  const neighborIds = new Set(teachingData.semanticNeighbors.map((neighbor) => neighbor.id));
  const mobileMode = state.ui.mobileChartMode;
  const points = state.paragraphs
    .filter((paragraph) => hasEmbeddingVector(paragraph.embedding))
    .map((paragraph) => ({
      ...paragraph,
      projection: projectVector(paragraph.embedding),
      isSelected: paragraph.id === selectedParagraph.id,
      isNeighbor: neighborIds.has(paragraph.id),
    }));

  if (!points.length) {
    return `<div class="empty-message">Semantic space appears once paragraph embeddings are ready.</div>`;
  }

  const selectedPoints = points.filter((point) => point.isSelected || point.isNeighbor);
  const backgroundPoints = points.filter((point) => !point.isSelected && !point.isNeighbor);
  const displayPoints = mobileMode ? selectedPoints.concat(sampleEvenly(backgroundPoints, MOBILE_BACKGROUND_POINT_LIMIT)) : points;

  const width = mobileMode ? 680 : 760;
  const height = mobileMode ? 360 : 460;
  const laidOutPoints = layoutScatterPoints(displayPoints, width, height);

  return `
    <svg class="embedding-stage-plot" viewBox="0 0 ${width} ${height}" aria-label="Semantic space">
      ${buildScatterGuides(width, height)}
      ${laidOutPoints
        .map((point) => {
          const pointColor = point.isSelected
            ? "var(--orange)"
            : point.isNeighbor
              ? "var(--cyan)"
              : `${getDocColor(point.docIndex)}cc`;
          const radius = point.isSelected ? 17 : point.isNeighbor ? 12 : 9;
          return `
            <g>
              <circle
                cx="${point.plotX}"
                cy="${point.plotY}"
                r="${radius}"
                class="semantic-space-point ${point.isSelected ? "is-selected" : point.isNeighbor ? "is-neighbor" : ""}"
                style="--space-point:${pointColor};"
              >
                <title>${escapeHtml(`${point.docName} · Chunk ${point.indexInDoc + 1}`)}</title>
              </circle>
            </g>
          `;
        })
        .join("")}
    </svg>
  `;
}

function renderEmbeddingStageDisplay(selectedParagraph, detail, teachingData) {
  if (!elements.embeddingStageDisplay) {
    return;
  }

  if (!selectedParagraph) {
    elements.embeddingStageDisplay.innerHTML = `
      <div class="empty-message">Run the pipeline and select a paragraph from Embedding previews to inspect its embedding journey.</div>
    `;
    return;
  }

  const keyTokenChips = teachingData.tokens
    .map((token, index) => ({
      token,
      score: teachingData.tokenImportance[index],
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 6)
    .map(
      (item) => `<span class="token-chip">${escapeHtml(item.token)}</span>`,
    )
    .join("");

  const attentionCentrality = teachingData.attentionNodes
    .map((node) => ({
      token: node.token,
      score: teachingData.attentionNodes.reduce((sum, candidate) => {
        const influence = candidate.influences.find((item) => item.index === node.index);
        return sum + (influence ? influence.weight : 0);
      }, 0),
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 5)
    .map(
      (item) => `<span class="token-chip is-strong">${escapeHtml(item.token)}</span>`,
    )
    .join("");

  let visualHtml = "";
  let asideHtml = "";
  const mobileChartNotice = state.ui.mobileChartMode
    ? `<div class="embedding-stage-note mobile-chart-note"><p>Mobile view uses a lighter static chart so the simulator stays stable on phones.</p></div>`
    : "";

  switch (detail.key) {
    case "tokenization":
      visualHtml = buildTokenScatterSvg(
        teachingData.tokenPoints.map((point) => ({
          ...point,
          importance: teachingData.tokenImportance[point.index],
        })),
        {
          ariaLabel: "Tokenization projection",
          pointColor: "var(--blue)",
          titleBuilder: (point) => `Token ${point.index + 1}: ${point.token}`,
        },
      );
      asideHtml = `
        <div class="embedding-stage-note">
          <p class="card-kicker">What you see</p>
          <p>${escapeHtml(detail.text)}</p>
        </div>
        <div class="embedding-stage-note">
          <p class="card-kicker">Tokens in this paragraph</p>
          <div class="token-chip-row">${teachingData.tokens.map((token) => `<span class="token-chip">${escapeHtml(token)}</span>`).join("")}</div>
        </div>
        ${mobileChartNotice}
      `;
      break;
    case "transformer":
      visualHtml = buildTokenScatterSvg(
        teachingData.attentionNodes.map((node) => ({
          ...node,
          importance: teachingData.tokenImportance[node.index],
        })),
        {
          projectionKey: "transformedProjection",
          ariaLabel: "Self-attention projection",
          pointColor: "var(--cyan)",
          showAttentionLinks: true,
          titleBuilder: (node) =>
            `Token ${node.index + 1}: ${node.token}\nStrongest attention: ${node.influences
              .map((influence) => `${influence.token} (${cosineToPercentage(influence.score)}%)`)
              .join(", ")}`,
        },
      );
      asideHtml = `
        <div class="embedding-stage-note">
          <p class="card-kicker">Stage focus</p>
          <p>${escapeHtml(detail.text)}</p>
        </div>
        <div class="embedding-stage-note">
          <p class="card-kicker">Hover behavior</p>
          <p>Hover a token vector to reveal the strongest self-attention links that shaped it.</p>
        </div>
        <div class="embedding-stage-note">
          <p class="card-kicker">Most connected tokens</p>
          <div class="token-chip-row">${attentionCentrality}</div>
        </div>
        ${mobileChartNotice}
      `;
      break;
    case "pooling":
      visualHtml = buildTokenScatterSvg(
        teachingData.attentionNodes.map((node) => ({
          ...node,
          importance: teachingData.tokenImportance[node.index],
        })),
        {
          projectionKey: "transformedProjection",
          ariaLabel: "Pooling view",
          pointColor: "var(--blue)",
          centerPoint: {
            projection: teachingData.pooledProjection,
            label: "Pooled vector",
          },
          showPoolingLinks: true,
          titleBuilder: (node) => `Token ${node.index + 1}: ${node.token}`,
        },
      );
      asideHtml = `
        <div class="embedding-stage-note">
          <p class="card-kicker">Pooled paragraph vector</p>
          ${buildEmbeddingBarsMarkup(teachingData.pooledVector, "embedding-preview is-large")}
        </div>
        <div class="embedding-stage-note">
          <p class="card-kicker">Interpretation</p>
          <p>${escapeHtml(detail.text)}</p>
        </div>
        ${mobileChartNotice}
      `;
      break;
    case "contextual":
      visualHtml = `
        <div class="embedding-comparison-grid">
          <article class="embedding-comparison-card">
            <p class="card-kicker">Before context</p>
            <h4>Pooled vector</h4>
            ${buildEmbeddingBarsMarkup(teachingData.pooledVector, "embedding-preview is-large")}
          </article>
          <article class="embedding-comparison-card is-highlighted">
            <p class="card-kicker">After context</p>
            <h4>Context-shaped vector</h4>
            ${buildEmbeddingBarsMarkup(teachingData.contextualVector, "embedding-preview is-large")}
          </article>
        </div>
      `;
      asideHtml = `
        <div class="embedding-stage-note">
          <p class="card-kicker">Context-heavy tokens</p>
          <div class="token-chip-row">${keyTokenChips}</div>
        </div>
        <div class="embedding-stage-note">
          <p class="card-kicker">What changed</p>
          <p>${escapeHtml(detail.text)}</p>
        </div>
      `;
      break;
    case "contrastive":
      visualHtml = `
        <div class="embedding-comparison-grid">
          <article class="embedding-comparison-card">
            <p class="card-kicker">Before contrastive learning</p>
            <h4>Context-shaped vector</h4>
            ${buildEmbeddingBarsMarkup(teachingData.contextualVector, "embedding-preview is-large")}
          </article>
          <article class="embedding-comparison-card is-highlighted">
            <p class="card-kicker">After contrastive learning</p>
            <h4>Sentence-aligned vector</h4>
            ${buildEmbeddingBarsMarkup(teachingData.contrastiveVector, "embedding-preview is-large")}
          </article>
        </div>
      `;
      asideHtml = `
        <div class="embedding-stage-note">
          <p class="card-kicker">What changed</p>
          <p>${escapeHtml(detail.text)}</p>
        </div>
        <div class="embedding-stage-note">
          <p class="card-kicker">Sentence influence</p>
          ${buildSentenceInfluenceMarkup(teachingData.sentenceScores)}
        </div>
      `;
      break;
    case "semantic-space":
      visualHtml = buildSemanticSpaceSvg(selectedParagraph, teachingData);
      asideHtml = `
        <div class="embedding-stage-note">
          <p class="card-kicker">Stage focus</p>
          <p>${escapeHtml(detail.text)}</p>
        </div>
        <div class="embedding-stage-note">
          <p class="card-kicker">Final paragraph vector</p>
          ${buildEmbeddingBarsMarkup(teachingData.finalVector, "embedding-preview is-large")}
        </div>
        <div class="embedding-stage-note">
          <p class="card-kicker">Nearest neighbors</p>
          <div class="neighbor-list">
            ${teachingData.semanticNeighbors
              .map(
                (neighbor) => `
                  <div class="neighbor-row">
                    <span>${escapeHtml(`${neighbor.docName} · Chunk ${neighbor.indexInDoc + 1}`)}</span>
                    <strong>${Math.round(clamp((neighbor.score + 1) * 50, 0, 100))}%</strong>
                  </div>
                `,
              )
              .join("")}
          </div>
        </div>
        ${mobileChartNotice}
      `;
      break;
    default:
      visualHtml = `<div class="empty-message">Select a stage to inspect its transformation.</div>`;
      asideHtml = "";
  }

  elements.embeddingStageDisplay.innerHTML = `
    <div class="embedding-stage-shell">
      <div class="embedding-stage-header">
        <div>
          <p class="card-kicker">Selected paragraph</p>
          <h3>${escapeHtml(`${selectedParagraph.docName} · Chunk ${selectedParagraph.indexInDoc + 1}`)}</h3>
        </div>
        <span class="doc-stat">${selectedParagraph.charCount} chars</span>
      </div>
      <p class="topic-copy">${escapeHtml(summarizeText(selectedParagraph.text, 240))}</p>
      <div class="embedding-stage-workspace">
        <div class="embedding-stage-visual">${visualHtml}</div>
        <aside class="embedding-stage-aside">${asideHtml}</aside>
      </div>
    </div>
  `;
}

function renderEmbeddingExplorer() {
  if (!elements.embeddingDetailsGrid || !elements.embeddingStageDisplay) {
    return;
  }

  const embeddingDetails = documentationCopy.tabs.embeddings.details;
  elements.embeddingDetailsGrid.innerHTML = embeddingDetails
    .map(
      (detail, index) => `
        <button
          type="button"
          class="detail-card embedding-detail-card ${state.embeddingInspector.stageKey === detail.key ? "is-active" : ""}"
          data-embedding-stage="${detail.key}"
        >
          <p class="card-kicker">Stage ${index + 1}</p>
          <h3>${escapeHtml(detail.title)}</h3>
          <p>${escapeHtml(detail.text)}</p>
        </button>
      `,
    )
    .join("");

  elements.embeddingDetailsGrid.querySelectorAll("[data-embedding-stage]").forEach((button) => {
    button.addEventListener("click", () => {
      setEmbeddingInspectorStage(button.getAttribute("data-embedding-stage"));
    });
  });

  const selectedParagraph = ensureSelectedEmbeddingParagraph();
  const activeDetail =
    embeddingDetails.find((detail) => detail.key === state.embeddingInspector.stageKey) || embeddingDetails[0];

  if (!selectedParagraph) {
    renderEmbeddingStageDisplay(null, activeDetail, null);
    return;
  }

  renderEmbeddingStageDisplay(selectedParagraph, activeDetail, buildTeachingParagraphData(selectedParagraph));
}

function renderEmbeddingGrid() {
  if (!state.paragraphs.length) {
    elements.embeddingGrid.innerHTML = `<div class="empty-message">Run the pipeline to preview paragraph embeddings.</div>`;
    renderEmbeddingExplorer();
    return;
  }

  const previewParagraphs = buildEmbeddingPreviewParagraphs();
  if (!previewParagraphs.length) {
    elements.embeddingGrid.innerHTML =
      `<div class="empty-message">Embeddings are being computed. Preview cards will appear as vectors become available.</div>`;
    renderEmbeddingExplorer();
    return;
  }

  const selectedParagraph = ensureSelectedEmbeddingParagraph();
  elements.embeddingGrid.innerHTML = previewParagraphs
    .map(
      (paragraph) => `
        <article
          class="embedding-card ${paragraph.isRetrieved ? "is-highlighted" : ""} ${selectedParagraph?.id === paragraph.id ? "is-selected" : ""}"
          data-embedding-paragraph="${paragraph.id}"
          tabindex="0"
          role="button"
          aria-pressed="${selectedParagraph?.id === paragraph.id}"
        >
          <div class="embedding-card-header">
            <div>
              <p class="card-kicker">Paragraph embedding</p>
              <h3>${escapeHtml(paragraph.docName)}</h3>
            </div>
            <span class="doc-stat">${paragraph.charCount} chars</span>
          </div>
          <p>${escapeHtml(summarizeText(paragraph.text, 150))}</p>
          ${buildEmbeddingBarsMarkup(paragraph.embedding)}
        </article>
      `,
    )
    .join("");

  elements.embeddingGrid.querySelectorAll("[data-embedding-paragraph]").forEach((card) => {
    const paragraphId = card.getAttribute("data-embedding-paragraph");
    card.addEventListener("click", () => {
      setEmbeddingInspectorParagraph(paragraphId);
    });
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setEmbeddingInspectorParagraph(paragraphId);
      }
    });
  });

  renderEmbeddingExplorer();
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

  const mobileMode = state.ui.mobileChartMode;
  const prioritizedPoints = points.filter((point) => point.isQuery || point.isRetrieved);
  const backgroundPoints = points.filter((point) => !point.isQuery && !point.isRetrieved);
  const displayPoints = mobileMode ? prioritizedPoints.concat(sampleEvenly(backgroundPoints, MOBILE_BACKGROUND_POINT_LIMIT)) : points;

  const xs = displayPoints.map((point) => point.projection.x);
  const ys = displayPoints.map((point) => point.projection.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const pointMarkup = displayPoints
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

  elements.semanticMap.innerHTML = `${pointMarkup}${
    mobileMode
      ? '<div class="semantic-map-caption">Mobile view shows a lighter subset of points for stability.</div>'
      : ""
  }`;
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

function buildRetrievalChartMarkup() {
  if (
    !state.retrieval?.topResults.length ||
    !hasEmbeddingVector(state.retrieval.queryEmbedding) ||
    !elements.retrievalChartPanel
  ) {
    return `<div class="empty-message">Run a query to draw the retrieval map.</div>`;
  }

  const mobileMode = state.ui.mobileChartMode;
  const retrievedIds = new Set(state.retrieval.topResults.map((result) => result.id));
  const points = state.paragraphs
    .filter((paragraph) => hasEmbeddingVector(paragraph.embedding))
    .map((paragraph) => ({
      ...paragraph,
      projection: projectVector(paragraph.embedding),
      isRetrieved: retrievedIds.has(paragraph.id),
      isQuery: false,
      isOrigin: false,
    }));

  points.push({
    id: "retrieval-query-point",
    projection: projectVector(state.retrieval.queryEmbedding),
    isQuery: true,
    isOrigin: false,
  });
  points.push({
    id: "retrieval-origin-point",
    projection: { x: 0, y: 0 },
    isQuery: false,
    isOrigin: true,
  });

  const width = 860;
  const height = 520;
  const laidOutPoints = layoutScatterPoints(points, width, height);
  const queryPoint = laidOutPoints.find((point) => point.isQuery);
  const originPoint = laidOutPoints.find((point) => point.isOrigin);
  if (!queryPoint || !originPoint) {
    return `<div class="empty-message">Run a query to draw the retrieval map.</div>`;
  }
  const paragraphPoints = laidOutPoints.filter((point) => !point.isQuery && !point.isOrigin);
  const retrievedPoints = paragraphPoints.filter((point) => point.isRetrieved);
  const backgroundPoints = mobileMode
    ? sampleEvenly(paragraphPoints.filter((point) => !point.isRetrieved), MOBILE_BACKGROUND_POINT_LIMIT)
    : paragraphPoints.filter((point) => !point.isRetrieved);

  return `
    <div class="retrieval-chart-shell">
      <svg class="retrieval-chart-plot ${mobileMode ? "is-mobile-static" : ""}" viewBox="0 0 ${width} ${height}" aria-label="Retrieval similarity map">
        ${buildScatterGuides(width, height)}
        ${backgroundPoints
          .map(
            (point) => `
              <circle
                cx="${point.plotX}"
                cy="${point.plotY}"
                r="8"
                class="retrieval-chart-point"
                style="--retrieval-point:${getDocColor(point.docIndex)};"
              >
                <title>${escapeHtml(`${point.docName} - Chunk ${point.indexInDoc + 1}`)}</title>
              </circle>
            `,
          )
          .join("")}
        ${retrievedPoints
          .map((point) => {
            const result = state.retrieval.topResults.find((candidate) => candidate.id === point.id);
            const rank = state.retrieval.topResults.findIndex((candidate) => candidate.id === point.id) + 1;
            const cosineLabel = `${Math.round(result.score * 100)}%`;
            const cosineX = (point.plotX + queryPoint.plotX) / 2;
            const cosineY = Math.min(point.plotY, queryPoint.plotY) - 18;
            const mobileLines = mobileMode
              ? `
                <line
                  x1="${point.plotX}"
                  y1="${point.plotY}"
                  x2="${originPoint.plotX}"
                  y2="${originPoint.plotY}"
                  class="retrieval-chart-link is-chunk"
                ></line>
                <line
                  x1="${queryPoint.plotX}"
                  y1="${queryPoint.plotY}"
                  x2="${originPoint.plotX}"
                  y2="${originPoint.plotY}"
                  class="retrieval-chart-link is-query"
                ></line>
              `
              : "";
            return `
              <g class="retrieval-chart-node">
                ${mobileMode ? mobileLines : `<g class="retrieval-chart-hover">
                  <line
                    x1="${point.plotX}"
                    y1="${point.plotY}"
                    x2="${originPoint.plotX}"
                    y2="${originPoint.plotY}"
                    class="retrieval-chart-link is-chunk"
                  ></line>
                  <line
                    x1="${queryPoint.plotX}"
                    y1="${queryPoint.plotY}"
                    x2="${originPoint.plotX}"
                    y2="${originPoint.plotY}"
                    class="retrieval-chart-link is-query"
                  ></line>
                  <text x="${cosineX}" y="${cosineY}" class="retrieval-chart-cosine">cosine ${cosineLabel}</text>
                </g>`}
                <circle
                  cx="${point.plotX}"
                  cy="${point.plotY}"
                  r="15"
                  class="retrieval-chart-point is-top"
                  style="--retrieval-point:${getDocColor(point.docIndex)};"
                >
                  <title>${escapeHtml(`${result.docName} - Chunk ${result.indexInDoc + 1} - cosine ${cosineLabel}`)}</title>
                </circle>
                <text x="${point.plotX}" y="${point.plotY + 4}" class="retrieval-chart-rank">${rank}</text>
              </g>
            `;
          })
          .join("")}
        <g class="retrieval-chart-origin">
          <circle cx="${originPoint.plotX}" cy="${originPoint.plotY}" r="10" class="retrieval-origin-point"></circle>
          <text x="${originPoint.plotX}" y="${originPoint.plotY + 24}" class="retrieval-origin-label">Origin</text>
        </g>
        <g class="retrieval-chart-query">
          <circle cx="${queryPoint.plotX}" cy="${queryPoint.plotY}" r="17" class="retrieval-query-point"></circle>
          <text x="${queryPoint.plotX}" y="${queryPoint.plotY + 4}" class="retrieval-chart-rank">Q</text>
          <text x="${queryPoint.plotX}" y="${queryPoint.plotY + 34}" class="retrieval-query-label">Query</text>
        </g>
      </svg>
      <div class="retrieval-chart-legend">
        <span class="doc-stat">${
          mobileMode
            ? "Mobile view uses a lighter static map. Read the rank cards for the full details."
            : "Hover a highlighted chunk to reveal the cosine value and its relation to the query."
        }</span>
      </div>
    </div>
  `;
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
    if (elements.retrievalChartPanel) {
      elements.retrievalChartPanel.innerHTML = `<div class="empty-message">Run a query to draw the retrieval map.</div>`;
    }
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

  if (elements.retrievalChartPanel) {
    elements.retrievalChartPanel.innerHTML = buildRetrievalChartMarkup();
  }
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
  const pageParagraphGroups = [];
  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber);
    const textContent = await page.getTextContent();
    const lines = buildPdfLines(textContent.items);
    pageParagraphGroups.push(buildPdfParagraphs(lines));
  }
  return normalizeExtractedText(postProcessPdfParagraphs(pageParagraphGroups).join("\n\n"));
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
  const rawCandidates = splitIntoParagraphCandidates(text);
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
  state.embeddingInspector.paragraphId = null;
  state.extractionReference = {
    type: "code",
    documentName: null,
  };
  state.splittingReference = {
    type: "code",
    documentName: null,
  };
  state.pipelineDirty = false;
  state.retrieval = null;
  elements.fileInput.value = "";
  elements.queryInput.value = "";
  state.projectionAxes = null;
  teachingParagraphCache.clear();

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
  state.embeddingInspector.paragraphId = null;
  state.extractionReference = {
    type: "code",
    documentName: null,
  };
  state.splittingReference = {
    type: "code",
    documentName: null,
  };
  state.retrieval = null;
  state.projectionAxes = null;
  teachingParagraphCache.clear();
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

let responsiveRefreshTimer = null;

function refreshViewportMode() {
  const nextMode = detectMobileChartMode();
  if (state.ui.mobileChartMode === nextMode) {
    return;
  }

  state.ui.mobileChartMode = nextMode;
  renderRetrievalOutputs();
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
  window.addEventListener("resize", () => {
    window.clearTimeout(responsiveRefreshTimer);
    responsiveRefreshTimer = window.setTimeout(refreshViewportMode, 120);
  });
}

function initialize() {
  state.ui.mobileChartMode = detectMobileChartMode();
  renderDocumentationContent();
  setupTabs();
  setupSuggestions();
  attachEvents();
  resetAll();
}

initialize();
