# RAG Architecture

The RAG layer is designed to be understandable in an interview and defensible in a portfolio review.

## Flow

```text
CFPB processed Parquet
  -> public narrative extraction
  -> cached hybrid index
  -> metadata-filtered retrieval
  -> deterministic analytics context
  -> grounded answer layer
  -> Streamlit Evidence Search page
```

## Indexed Evidence

The index stores only CFPB complaints with usable public narratives. The current release indexed 5,343 narratives from the 250,000-record local CFPB extract.

Each retrieved evidence row keeps:

- complaint ID;
- date received;
- company;
- product;
- issue;
- state;
- cleaned public narrative;
- lexical, dense, and hybrid retrieval scores.

## Retrieval

The retriever combines:

- TF-IDF lexical search for exact complaint language;
- local LSA/SVD vectors as a lightweight dense retrieval layer;
- metadata filters for company, product, issue, state, and date range;
- weighted rank fusion.

This avoids a hosted embedding dependency while still giving better coverage than plain keyword search.

## Answer Layer

The answer layer has two modes:

- `deterministic_no_key_fallback`: builds a structured answer from retrieved citations and deterministic analytics.
- optional LLM mode: calls OpenAI or Anthropic only when `LLM_PROVIDER` and the matching API key are configured.

The system prompt requires citations, evidence grounding, abstention when evidence is weak, and clear caveats. The LLM is not allowed to calculate or invent metrics.

## Traceability

The Streamlit page exposes:

- selected filters;
- analytics calls;
- retrieval candidate count;
- retrieval method and score weights;
- cited complaint records;
- prompt context sent to the fallback or LLM.

This is intentionally visible so the project feels like an analyst product, not a black-box chatbot.
