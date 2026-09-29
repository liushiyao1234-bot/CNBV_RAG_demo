# CNBV RAG Demo

A small Retrieval-Augmented Generation (RAG) system built from scratch to understand the internal mechanics of a modern RAG pipeline.

The project uses a Spanish CNBV/Ceneval study guide as the source document and implements the full pipeline manually, including:

**PDF parsing → text cleaning → semantic chunking → multilingual embedding retrieval → cross-encoder reranking → context filtering → LLM generation → source citation**

The project intentionally avoids high-level RAG frameworks such as LangChain for the core pipeline. The goal is not to build the shortest possible implementation, but to understand what each component actually does.

---

## Why This Project

Most RAG tutorials can be implemented in a few lines using an existing framework:

```python
loader → splitter → vectorstore → retriever → llm
```

That is convenient, but it also hides many important design decisions:

- How should a PDF be cleaned before indexing?
- What exactly is a chunk?
- How should chunk boundaries be chosen?
- What is the difference between embedding retrieval and reranking?
- Why can a retrieved chunk be relevant but still insufficient to answer a question?
- How large should the retrieval candidate pool be?
- How many chunks should actually be sent to the LLM?
- How can the final answer remain traceable to the source document?
- Which component is responsible when the system becomes slow?

This repository implements these steps separately so that each layer can be inspected, tested, and replaced independently.

---

## Architecture

```text
                    ┌─────────────────────┐
                    │      PDF Source     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     PDF Parsing     │
                    │      PyMuPDF        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Text Cleaning    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Chunking   │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │ Multilingual Embedding Model   │
              │ intfloat/multilingual-e5-small │
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Retriever Top 10    │
                    │ NumPy cosine search │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │ Cross-Encoder Reranker         │
              │ mmarco-mMiniLMv2-L12-H384-v1  │
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Reranker Top 5    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Context Filtering  │
                    │ score threshold +   │
                    │ max context count   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      GLM-5.3        │
                    │ Huawei Cloud MaaS   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Answer + Citations  │
                    └─────────────────────┘
```

---

## Current Models

### Embedding Model

```text
intfloat/multilingual-e5-small
```

The model is used as a bi-encoder.

Queries are encoded using:

```text
query: <user query>
```

Document chunks are encoded using:

```text
passage: <document chunk>
```

The embeddings are normalized, so dot product can be used directly as cosine similarity.

The multilingual embedding model also makes cross-language retrieval possible. For example, a user can ask a question in Chinese while the indexed document is written in Spanish.

---

### Reranker

```text
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

The reranker jointly processes:

```text
[query, candidate passage]
```

and produces a relevance score.

Unlike the embedding retriever, the cross-encoder does not precompute independent passage representations. It reads the query and passage together, which generally provides better ranking quality but requires more computation.

The reranker operates only on the candidate set returned by the retriever.

Current configuration:

```text
Retriever Top K: 10
Reranker Top K: 5
```

---

### Generator

The final answer is generated using:

```text
GLM-5.3
```

through Huawei Cloud ModelArts MaaS.

The LLM receives only the chunks selected by the retrieval and filtering pipeline.

The prompt explicitly instructs the model to:

- answer using only the supplied sources;
- say when the supplied context is insufficient;
- answer in the same language as the user;
- preserve important source terminology;
- cite the source chunk and page number.

---

## Project Structure

```text
cnbv-rag-demo/
│
├─ data/
│  ├─ raw/
│  │  └─ <source PDF>
│  │
│  └─ processed/
│     ├─ pages.json
│     ├─ pages_cleaned.json
│     ├─ chunks_fixed.json
│     ├─ chunks_recursive.json
│     └─ chunks_semantic.json
│
├─ src/
│  ├─ __init__.py
│  ├─ pdf_parser.py
│  ├─ text_cleaner.py
│  ├─ document_builder.py
│  │
│  ├─ chunkers/
│  │  ├─ __init__.py
│  │  ├─ fixed_chunker.py
│  │  ├─ recursive_chunker.py
│  │  └─ semantic_chunker.py
│  │
│  ├─ embedding.py
│  ├─ similarity.py
│  ├─ retriever.py
│  ├─ reranker.py
│  ├─ generator.py
│  └─ rag_pipeline.py
│
├─ experiments/
│  ├─ __init__.py
│  ├─ compare_chunkers.py
│  ├─ test_questions.json
│  ├─ test_reranker.py
│  ├─ diagnose_reranker.py
│  └─ compare_reranker_all.py
│
├─ outputs/
│  └─ <experiment outputs>
│
├─ .env
├─ .env.example
├─ .gitignore
├─ requirements.txt
└─ README.md
```

Generated data files and the original source document may be excluded from the public repository depending on source redistribution requirements.

---

## Chunking Experiments

Three chunking strategies were implemented.

### Fixed-Length Chunking

A deliberately simple baseline.

```text
Target size: 1000 characters
Overlap: 150 characters
```

The splitter ignores semantic boundaries and slices text primarily by character position.

Its purpose is to provide a baseline rather than an optimized solution.

---

### Recursive Chunking

Recursive chunking attempts to find more natural boundaries using separators such as:

```text
paragraph break
sentence boundary
newline
space
```

It also preserves a small overlap between adjacent chunks.

Unlike the fixed baseline, recursive chunks may span multiple PDF pages.

---

### Semantic Chunking

Semantic chunking first divides the document into smaller semantic units and generates embeddings for them.

Cosine similarity is calculated between adjacent units:

```text
unit 1 ↔ unit 2
unit 2 ↔ unit 3
unit 3 ↔ unit 4
...
```

Large drops in adjacent similarity are treated as candidate semantic boundaries.

The implementation also enforces minimum and maximum token limits so that semantic coherence does not produce excessively large chunks.

Current configuration includes:

```text
Breakpoint percentile: 15
Minimum chunk size: 120 tokens
Maximum chunk size: 450 tokens
```

Semantic chunking is used by the final RAG pipeline.

One important observation from the experiment was that a good RAG chunk does not need to contain an entire topic or chapter.

The goal is instead to create chunks that are:

```text
focused enough to retrieve
+
large enough to preserve useful context
```

Multiple chunks can be retrieved together when the answer spans more than one semantic region.

---

## Retrieval

The retriever intentionally uses a simple NumPy implementation instead of a vector database.

For the current document, there are only a small number of chunks, so introducing FAISS, Milvus, Elasticsearch, or another vector database would add infrastructure without improving the educational value of the demo.

The retrieval process is:

```text
User Query
    ↓
Query Embedding
    ↓
Dot Product Against All Passage Embeddings
    ↓
Sort by Similarity
    ↓
Top 10 Candidates
```

This stage is extremely fast for the current dataset.

Example measured latency:

```text
Retriever: ~0.02 s
```

A vector database would become useful once the corpus grows substantially.

---

## Why Reranking Is Necessary

Embedding retrieval is optimized for candidate recall, not perfect ranking.

During testing, several answer-bearing chunks appeared relatively deep in the initial retrieval results.

The architecture therefore separates:

```text
Candidate Retrieval
```

from:

```text
Candidate Ranking
```

The retriever first returns 10 candidates.

The cross-encoder then jointly evaluates the query and each candidate and reranks them.

This distinction is important because:

```text
Retrieval similarity ≠ answer completeness
```

A passage may be semantically related to the question while still not containing enough information to answer it.

---

## Reranker Performance Optimization

The first implementation used:

```text
BAAI/bge-reranker-v2-m3
```

The ranking quality was good, but CPU inference was too slow for interactive use.

Measured reranking latency was approximately:

```text
36–40 seconds per query
```

Profiling showed that other components were not responsible:

```text
Retriever:       ~0.02 s
Context Filter:  ~0.00 s
Reranker:        ~37–40 s
LLM Generation:  ~5–10 s
```

The reranker was therefore replaced with:

```text
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

On the same CPU environment, measured reranking latency dropped to approximately:

```text
1.6 seconds
```

while maintaining strong ranking behavior on the local evaluation set.

This experiment illustrates an important practical RAG design lesson:

```text
Model quality alone is not enough.

A useful production component must balance:

retrieval quality
×
latency
×
hardware requirements
×
cost
```

A larger model is not automatically the better engineering choice.

---

## Reranker Scores

Reranker scores should not be interpreted as probabilities.

For example:

```text
0.80
```

does **not** mean:

```text
80% probability that this chunk contains the correct answer
```

The score is primarily useful for relative ranking within the same model.

Different reranker models may produce very different score distributions even when they rank the same passages similarly.

Therefore, thresholds should be calibrated for the model and dataset rather than copied between models.

---

## Context Filtering

Sending every reranked candidate to the LLM is unnecessary and may introduce irrelevant information.

The current pipeline therefore applies another layer after reranking.

Current policy:

```text
Always keep Reranker Rank 1

Keep additional chunks when:
reranker_score >= 0.2

Maximum context chunks:
3
```

The final architecture is therefore:

```text
Retriever Top 10
        ↓
Reranker Top 5
        ↓
Context Filter
        ↓
1–3 chunks
        ↓
LLM
```

The score threshold is an empirical configuration for this demo and should not be treated as a universal value.

---

## Source Attribution

Each chunk retains source metadata including:

```text
chunk_id
start_page
end_page
```

When context is sent to the generator, the source metadata is included alongside the text.

For example:

```text
[Source: Chunk 8, Page 10]
<chunk text>
```

The model can therefore produce answers such as:

```text
Money laundering is generally divided into three stages:
placement, concealment, and integration.
[Chunk 8, Page 10]
```

The terminal also displays which sources were actually supplied to the LLM:

```text
Sources Sent to LLM:
[Chunk 7, Page 10] rerank_score=0.3985
[Chunk 8, Page 10] rerank_score=0.3533
```

This provides basic traceability between:

```text
source document
→ retrieved evidence
→ generated answer
```

---

## Multilingual Retrieval

The source document is written in Spanish, but the embedding and reranking models are multilingual.

The pipeline can therefore accept queries such as:

```text
¿Cuáles son las etapas del lavado de dinero?
```

```text
What are the stages of money laundering?
```

or:

```text
洗钱分为哪几个阶段？
```

while retrieving evidence from the same Spanish corpus.

The generator then answers in the language used by the user.

Cross-language retrieval has been manually tested in the demo, but the full evaluation dataset currently focuses primarily on Spanish queries.

---

## Evaluation

The repository includes a small manually designed evaluation set covering different types of questions, including:

```text
definitions
lists
test duration
test purpose
target population
concept comparisons
international organizations
cognitive dimensions
question types
exam procedures
identification requirements
pilot items
```

The evaluation scripts inspect both:

```text
Retriever Top K
```

and:

```text
Reranker Top K
```

This makes it possible to distinguish three different failure modes:

```text
1. Recall failure

The correct chunk never entered the candidate pool.


2. Ranking failure

The correct chunk was retrieved but ranked too low.


3. Answer sufficiency failure

A highly ranked chunk was relevant to the topic but did not
contain enough information to answer the question.
```

This distinction became one of the most useful lessons from the project.

---

## Installation

Create a Python environment and install the required packages.

Example:

```bash
pip install pymupdf sentence-transformers numpy torch requests python-dotenv
```

Create a `.env` file in the project root:

```text
MAAS_API_KEY=your_maas_api_key_here
```

An example file can be committed safely:

```text
.env.example
```

```text
MAAS_API_KEY=your_maas_api_key_here
```

Never commit the real `.env` file.

---

## Running the RAG Demo

After the processed chunks have been generated, start the interactive RAG pipeline with:

```bash
python -m src.rag_pipeline
```

The application initializes:

```text
Embedding model
↓
Semantic chunks
↓
Passage embeddings
↓
Reranker
```

and then enters interactive mode:

```text
Question:
```

Questions can be asked repeatedly without reloading the models.

Example:

```text
Question: 洗钱分为哪几个阶段？
```

Example output:

```text
Answer:
...

[Chunk 8, Page 10]

Sources Sent to LLM:
[Chunk 7, Page 10] rerank_score=...
[Chunk 8, Page 10] rerank_score=...

Timing:
Retriever:       0.02s
Reranker:        1.58s
Context Filter:  0.0000s
LLM Generation:  9.80s
Total:           11.41s
```

Type:

```text
exit
```

to stop the program.

---

## Security

API credentials are loaded through environment variables.

The repository should exclude:

```text
.env
__pycache__/
*.pyc
```

Recommended `.gitignore` entries:

```gitignore
.env

__pycache__/
*.pyc
*.pyo

.vscode/
.idea/
```

Before publishing the repository, verify that API keys or other credentials have never been committed.

If a secret has already been committed, adding it to `.gitignore` is not sufficient because the value remains in Git history.

---

## Limitations

This project is intentionally small and educational.

Current limitations include:

- Passage embeddings are recomputed when the application starts.
- Retrieval uses brute-force NumPy search instead of a vector database.
- The evaluation dataset is small.
- Cross-language retrieval has not yet been benchmarked systematically.
- Context filtering uses a manually selected reranker threshold.
- Citation labels are generated by the LLM based on source metadata provided in the prompt.
- The source PDF contains some text extraction artifacts caused by PDF formatting.
- Images and image-only PDF content are not included in the current text-only RAG pipeline.
- Generation latency depends on the external MaaS API.

These are acceptable trade-offs for the current goal of understanding and demonstrating the complete RAG workflow.

---

## Possible Future Improvements

Possible extensions include:

```text
1. Persist passage embeddings to disk
   instead of recomputing them at startup.

2. Replace brute-force retrieval with
   FAISS or a vector database for larger corpora.

3. Add BM25 + dense hybrid retrieval.

4. Add systematic Chinese/English/Spanish
   cross-language evaluation.

5. Replace free-form citation generation with
   programmatically mapped source IDs.

6. Add streaming generation to reduce
   perceived LLM latency.

7. Add retrieval and generation evaluation metrics.

8. Introduce automatic threshold calibration
   for context filtering.

9. Add OCR / multimodal processing for
   image-based PDF content.

10. Add a lightweight web interface or API.
```

---

## Key Takeaways

The most important lesson from this project is that RAG is not a single model or a single retrieval operation.

It is a pipeline of independent decisions:

```text
How should the source be parsed?

How should text be cleaned?

Where should chunk boundaries be placed?

How should chunks be represented?

How many candidates should be retrieved?

How should those candidates be reranked?

Which chunks are actually useful enough
to send to the LLM?

How should the answer remain traceable
to the original source?

Where is latency introduced?

Which model provides the best
quality–latency trade-off?
```

Building each layer manually made these boundaries much clearer than using an end-to-end RAG framework.

The final system is intentionally simple, but every major component can now be inspected, benchmarked, replaced, or improved independently.

---

## Tech Stack

```text
Python
PyMuPDF
Sentence Transformers
multilingual-e5-small
mMARCO MiniLM Cross-Encoder
NumPy
PyTorch
Huawei Cloud ModelArts MaaS
GLM-5.3
```

---

## License

This repository contains implementation code for educational and demonstration purposes.

The source document used for experimentation is not part of the project license. Refer to the original document publisher for its applicable copyright and redistribution terms.