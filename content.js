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
    description: {
      kicker: "Overview",
      title: "What this simulator is doing",
      copy:
        "This simulator demonstrates the full Retrieval Augmented Generation flow in a way that is easy to inspect. It starts with documents, turns them into searchable paragraph representations, compares a question against those representations, and then shows how the retrieved evidence can support a grounded answer.",
      bullets: [
        "It keeps the intermediate pipeline steps visible instead of hiding them behind one final output.",
        "It uses the same idea for document and query embeddings so similarity can be compared consistently.",
        "It is designed for teaching, so each tab focuses on one stage of the RAG pipeline.",
      ],
    },
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
      bullets: [],
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
      kicker: "",
      title: "Loading and Splitting Document Texts into Paragraphs",
      copy:
        "After extraction, the Python reference script loops through every document, splits the text by newline characters, strips whitespace, and keeps only paragraphs with at least 50 characters. In the browser simulator, PDFs are first reconstructed into lines, grouped into paragraph candidates, merged across page breaks when a sentence clearly continues, and then cleaned so the split view reflects paragraphs rather than whole pages.",
      bullets: [],
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
      kicker: "",
      title: "Pre-computing embeddings for all paragraphs",
      copy:
        "The model converts each paragraph into a compact numerical representation called an embedding. Paragraphs with similar meaning end up closer together, which helps the retrieval step find relevant content even when the wording is different.",
      bullets: [],
      details: [
        {
          key: "tokenization",
          title: "Tokenization",
          text:
            "Inspect how the selected paragraph is split into tokens and where each token lands in a teaching-focused 2D projection.",
        },
        {
          key: "transformer",
          title: "Transformer Architecture",
          text:
            "See the token vectors after self-attention. Hover a token to reveal which other tokens influenced it the most.",
        },
        {
          key: "pooling",
          title: "Pooling",
          text:
            "Watch the token-level information collapse into one paragraph-level vector through vector aggregation.",
        },
        {
          key: "contextual",
          title: "Contextual Meaning",
          text:
            "Compare the paragraph vector before and after contextual signals reshape it around the most informative tokens.",
        },
        {
          key: "contrastive",
          title: "Contrastive Learning",
          text:
            "See how sentence-level evidence nudges a teaching-stage paragraph vector during contrastive learning and which sentences matter most.",
        },
        {
          key: "semantic-space",
          title: "Semantic Space",
          text:
            "Explore the final paragraph vector used for retrieval and see where it sits relative to the other paragraph embeddings.",
        },
      ],
      code: `model = SentenceTransformer('all-MiniLM-L6-v2')
paragraph_embeddings = model.encode(paragraphs, convert_to_tensor=True)`,
    },
    retrieval: {
      kicker: "",
      title: "Defining the Retrieval Function",
      copy:
        "Retrieval begins by encoding the user query with the same model, computing cosine similarity against every stored paragraph embedding, sorting the scores, and then returning the top results with source metadata.",
      bullets: [],
      formulaTitle: "",
      formula: "",
      code: `def retrieve_answer(query, top_k=1):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, paragraph_embeddings)[0]
    top_results = scores.argsort(descending=True)[:top_k]
    return top_results`,
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
  "How is backtesting used in validation of a rating model?",
  "How is the accuracy ratio of the model defined?",
  "How are the ratings defined?",
];
