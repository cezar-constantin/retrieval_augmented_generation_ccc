# CCC Retrieval Augmented Generation

Interactive teaching simulator for explaining Retrieval Augmented Generation in the browser, rebuilt with the darker portfolio-style interface used in `neural_network_ccc`.

## What It Does

- lets students upload PDF, DOCX, TXT, or Markdown documents;
- extracts text from the uploaded files directly in the browser;
- splits the text into paragraph chunks;
- computes embeddings for each paragraph;
- retrieves the top-k passages for a question;
- builds a grounded answer from retrieved evidence;
- explains the main RAG stages in dedicated teaching tabs.

## Design Direction

- portfolio-style interface aligned with `CCC Neural Network`;
- dark background based on `#0B0F1A`;
- blue and cyan accent system for interactive states and data views;
- `Space Grotesk` headings with `Inter` body typography.

## Technical Notes

- static deployment ready for GitHub Pages;
- tries to load `all-MiniLM-L6-v2` in the browser via Transformers.js;
- falls back to deterministic local vectors if the remote model cannot be loaded;
- uses PDF.js for PDF extraction and Mammoth.js for DOCX extraction.

## Local Run

The project is static, so it does not need a build step.

1. Start a local server from the project root:

```powershell
python -m http.server 4173
```

2. Open:

```text
http://localhost:4173
```

## GitHub Pages Deployment

The repository includes `.github/workflows/deploy-pages.yml`.

1. push the code to the `main` branch of `cezar-constantin/retrieval_augmented_generation_ccc`;
2. in GitHub, enable `Pages` with `GitHub Actions` as the source;
3. after the first successful workflow run, the app will be available at:

```text
https://cezar-constantin.github.io/retrieval_augmented_generation_ccc/
```

## Important Files

- `index.html` - portfolio-style page structure and simulator layout;
- `styles.css` - dark theme and interactive visual styling;
- `app.js` - document processing, embeddings, retrieval, and UI rendering;
- `content.js` - teaching copy shown across the explanatory tabs.
