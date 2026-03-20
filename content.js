export const documentationCopy = {
  hero: {
    eyebrow: "Interactive teaching simulator",
    title: "Retrieval Augmented Generation",
    text:
      "Upload a set of documents, run the RAG pipeline step by step, and watch how the system extracts text, splits it into paragraphs, pre-computes embeddings, retrieves the closest passages, and then builds a grounded answer for teaching purposes.",
  },
  simulatorNote:
    "The provided Python script is retrieval-first: it returns the most relevant paragraphs. This simulator keeps that logic, then adds a small grounded synthesis layer so students can also see how retrieved context can feed a final answer in a full RAG story.",
  tabs: {
    simulator: {
      kicker: "Live pipeline",
      title: "Simulator",
      copy:
        "This view connects every stage in one place. Students can upload documents, trigger the pipeline, inspect the retrieved evidence, and compare the final answer with the underlying paragraphs.",
    },
    extracting: {
      kicker: "",
      title: "Extracting text from Documents",
      copy:
        "This part of the pipeline checks the file extension, then uses a dedicated extraction path for each supported format. Whether DOCX, PDF, TXT, or Markdown, the code turns each uploaded file into plain text that the retrieval system can search.",
      bullets: [
        "Purpose: turn each uploaded file into plain text that the retrieval system can search.",
        "DOCX path: read document paragraphs and keep the non-empty ones.",
        "PDF path: extract text page by page, then join it into one searchable text body.",
        "Return value: the full text content of each document.",
      ],
      code: `def extract_text(file_path):
    """
    Extracts text from DOCX and PDF files.
    """
    text = ""
    if file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    elif file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        text = "\\n".join([page.get_text("text") for page in doc])
    return text`,
    },
    splitting: {
      kicker: "6. Loading and Splitting Document Texts into Paragraphs",
      title: "Loading and Splitting Document Texts into Paragraphs",
      copy:
        "After extraction, the Python script loops through every document, splits the text by newline characters, strips whitespace, and keeps only paragraphs with at least 50 characters. The simulator mirrors that teaching logic so students can see how chunks are formed before retrieval.",
      bullets: [
        "A dictionary comprehension loads the full text for each document.",
        "Each document is split into candidate paragraphs using newline boundaries.",
        "Paragraphs shorter than 50 characters are removed in the original script.",
        "Metadata is stored alongside every paragraph so the source document stays visible during retrieval.",
      ],
      code: `document_texts = {name: extract_text(path) for name, path in file_paths.items()}

paragraphs = []
meta_data = []

for doc_name, text in document_texts.items():
    for paragraph in text.split("\\n"):
        paragraph = paragraph.strip()
        if len(paragraph) >= 50:
            paragraphs.append(paragraph)
            meta_data.append(doc_name)`,
    },
    embeddings: {
      kicker: "7. Pre-computing Embeddings for All Paragraphs",
      title: "Pre-computing embeddings for all paragraphs",
      copy:
        "The original code loads a SentenceTransformer model called all-MiniLM-L6-v2 and encodes each paragraph into a dense numerical vector. Those vectors capture semantic meaning, so similar ideas end up close together even when the wording changes.",
      bullets: [
        "The same encoder is used for both document paragraphs and the user query.",
        "Pre-computing paragraph embeddings makes later searches much faster.",
        "Cosine similarity works because all texts live in the same vector space.",
        "In this browser version, the simulator first tries to load MiniLM in the client, then falls back to a deterministic teaching-safe vectorizer if the model cannot be fetched.",
      ],
      details: [
        {
          title: "Tokenization",
          text:
            "Text is broken into tokens so the model can process smaller language units instead of raw characters.",
        },
        {
          title: "Transformer Architecture",
          text:
            "The tokens pass through a Transformer that uses self-attention to understand relationships between words, even when the important words are far apart.",
        },
        {
          title: "Pooling",
          text:
            "The model aggregates token-level information into one fixed-length vector that summarizes the paragraph.",
        },
        {
          title: "Contextual Meaning",
          text:
            "Embeddings preserve context, so similar ideas written with different vocabulary can still land close to each other.",
        },
        {
          title: "Contrastive Learning",
          text:
            "Sentence-transformer models are fine-tuned so semantically similar sentences move closer together while unrelated ones are pushed apart.",
        },
        {
          title: "Semantic Space",
          text:
            "The final vectors live in a high-dimensional semantic space where cosine similarity becomes a meaningful retrieval signal.",
        },
      ],
      code: `model = SentenceTransformer('all-MiniLM-L6-v2')
paragraph_embeddings = model.encode(paragraphs, convert_to_tensor=True)`,
    },
    retrieval: {
      kicker: "8. Defining the Retrieval Function",
      title: "Defining the Retrieval Function",
      copy:
        "Retrieval begins by encoding the user query with the same model, computing cosine similarity against every stored paragraph embedding, sorting the scores, and then returning the top results with source metadata.",
      bullets: [
        "Query encoding: the question is mapped into the same vector space as the paragraphs.",
        "Cosine similarity: higher values mean the query and the paragraph point in a more similar semantic direction.",
        "Ranking: the scores are sorted in descending order to find the best matches.",
        "Result construction: each returned item includes the source document, paragraph text, and similarity score.",
      ],
      formulaTitle: "Similarity formula",
      formula: "score = cosine(query_embedding, paragraph_embedding)",
      code: `def retrieve_answer(query, top_k=1):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, paragraph_embeddings)[0]
    top_results = scores.argsort(descending=True)[:top_k]

    results = []
    for idx in top_results:
        results.append({
            "source": meta_data[idx],
            "paragraph": paragraphs[idx],
            "score": scores[idx].item()
        })
    return results`,
    },
  },
};

export const demoDocuments = [
  {
    name: "Methodology for Rating Model Development",
    sourceLabel: "Demo DOCX",
    text: `A rating model development process should begin with a clear problem definition, the identification of the target population, and the business objective that the model is expected to support.

The development phase typically includes data collection, feature engineering, sample construction, and segmentation logic. Analysts should document why each variable is included and how missing values are handled.

Model performance is usually evaluated through discriminatory power, stability, and calibration diagnostics. Development teams should keep a transparent audit trail so later validation teams can reproduce the process.

Even when a model performs well on historical data, documentation remains essential because governance stakeholders must understand assumptions, exclusions, and intended use cases before approval.`,
  },
  {
    name: "Monitoring Methodology for Rating Models",
    sourceLabel: "Demo DOCX",
    text: `Monitoring checks whether an approved model continues to behave as expected after deployment. This includes tracking population drift, changes in score distributions, and deviations between expected and observed default rates.

Backtesting is a key monitoring activity because it compares realized outcomes with the model's original expectations. When performance deteriorates, analysts investigate whether the issue comes from the environment, the data pipeline, or the model design itself.

Effective monitoring uses thresholds, dashboards, and escalation rules. If a threshold is breached, the institution should document the event, investigate root causes, and decide whether recalibration or redevelopment is necessary.

Monitoring is not only a technical activity. It also supports governance, because management needs a concise explanation of whether the model still fits the portfolio and business context.`,
  },
  {
    name: "Methodology for Validation of Rating Models",
    sourceLabel: "Demo DOCX",
    text: `Validation provides an independent challenge of model design, data quality, assumptions, and implementation. A validation team should assess conceptual soundness, empirical performance, and ongoing suitability.

Independent validators often review sampling choices, variable treatment, overfitting risk, and whether performance metrics are interpreted consistently with policy requirements. They also verify that documentation is complete enough to support replication.

Validation reports usually combine quantitative testing with qualitative judgement. A model may pass statistical checks and still receive remediation actions if implementation controls, documentation, or governance arrangements are weak.

A strong validation framework strengthens trust in the model because it separates development incentives from independent review and helps confirm that the rating system is appropriate for decision-making.`,
  },
  {
    name: "Methodology for Calibrating a Rating Model",
    sourceLabel: "Demo DOCX",
    text: `Calibration aligns model outputs with observed risk levels so that predicted probabilities remain meaningful in practice. It is especially important when portfolio conditions change or when long-run averages differ from recent performance.

Calibration methods may include scaling score bands, adjusting probability mappings, or re-estimating parameters against updated data. The chosen method should remain consistent with policy constraints and explainability expectations.

A recalibration does not automatically replace full redevelopment. Instead, it is often used when ranking power remains acceptable but the level of predicted risk no longer reflects observed outcomes.

Documentation should explain when recalibration is allowed, who approves it, and how post-calibration performance will be monitored over time.`,
  },
];

export const suggestedQuestions = [
  "How is backtesting used in monitoring?",
  "Why are short paragraphs filtered out before retrieval?",
  "What does the embedding model do in this pipeline?",
  "When would recalibration be preferred to redevelopment?",
];
