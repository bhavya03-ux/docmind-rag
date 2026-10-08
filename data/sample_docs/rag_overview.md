# Retrieval-Augmented Generation (RAG) Overview

Retrieval-Augmented Generation, usually shortened to RAG, combines a search step with a text-generation step. Instead of asking a language model to answer from memory alone, the system first looks up relevant passages in a document collection and then gives those passages to the model as context.

## Why use RAG
Language models can produce fluent but wrong answers, a problem known as hallucination. Grounding the answer in retrieved passages reduces this risk because the model can quote or paraphrase text that actually exists. RAG also lets a team update knowledge by re-indexing documents, with no need to retrain the model.

## The main steps
1. Ingestion: documents are read and cleaned.
2. Chunking: each document is split into small overlapping pieces so that a search result is focused and fits in the prompt.
3. Embedding: each chunk is converted into a vector of numbers that captures its meaning.
4. Retrieval: the question is embedded too, and the closest chunks are returned.
5. Generation: the model writes an answer from the retrieved chunks and cites them.

## Hybrid search
Vector search finds passages with similar meaning even when the wording differs, but it can miss exact identifiers such as error codes or product names. Keyword search with BM25 is strong on exact terms but weak on paraphrases. Hybrid search runs both and merges the two ranked lists, often with Reciprocal Rank Fusion, which adds 1 / (k + rank) for every list that contains a chunk.

## Evaluating a RAG system
Retrieval quality can be measured with hit rate, which checks whether a relevant chunk appears in the top results, and mean reciprocal rank, which rewards placing the relevant chunk higher. Answer quality needs separate checks, such as verifying that required facts appear and that citations point to real sources.
