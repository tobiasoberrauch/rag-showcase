# RAG Showcase

Dieses Repository enthält eine Sammlung von Beispielen und Skripten zur Verwendung von Retrieval-Augmented Generation (RAG) mit verschiedenen Modellen und Vektordatenbanken. Das Ziel ist es, einen umfassenden Überblick über die Implementierung und Nutzung von RAG zu geben.

## Inhaltsverzeichnis

- [RAG Showcase](#rag-showcase)
  - [Inhaltsverzeichnis](#inhaltsverzeichnis)
  - [Überblick](#überblick)
  - [Ordnerstruktur](#ordnerstruktur)
  - [Installation](#installation)
  - [Weitere Links](#weitere-links)
  - [Nutzung](#nutzung)
  - [Beispiele](#beispiele)
  - [Ergebnisse](#ergebnisse)
  - [Beitragende](#beitragende)
  - [Lizenz](#lizenz)

## Überblick

RAG kombiniert Informationsabruf und Textgenerierung, um auf Eingaben des Benutzers zu antworten, indem es relevante Dokumente aus einer Datenbank abruft und diese Informationen verwendet, um eine fundierte Antwort zu generieren.

## Ordnerstruktur

```plaintext
rag-showcase/
│
├── data/
│
├── langchain/
│   ├── 00_split_text_semantic.py
│   ├── 00_split_text.py
│   ├── 01_load_documents.py
│   ├── 02_vector_db_fs.py
│   ├── 03_vector_db_qdrant.py
│   ├── 04_write_documents_into_qdrant.py
│   ├── 05_query_qdrant.py
│   ├── 06_llm_lmstudio.py
│   ├── 06_llm_ollama.py
│   ├── 07_rag.py
│   ├── 10_chunk_bechmark.py
│   └── benchmark_results.csv
│
├── llama-index/
│   ├── 01_load_documents.py
│   ├── 04_write_documents_into_qdrant.py
│   ├── 05_query_qdrant_comparison.py
│   ├── 06_llm_lmstudio.py
│   ├── 06_llm_ollama.py
│   └── 07_rag.py
│
├── storage/
│
├── .gitignore
├── docker-compose.yaml
├── README.md
└── sandbox.ipynb
```

## Installation

1. Klone das Repository:

   ```bash
   git clone https://github.com/tobiasoberrauch/rag-showcase.git
   cd rag-showcase
   ```

2. Installiere die Abhängigkeiten:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Starte Docker-Container:

   ```bash
   docker-compose up
   ```

## Weitere Links

- [https://openai.com/index/introducing-text-and-code-embeddings/](Embeddings)

## Nutzung

Die Skripte im Ordner `langchain` und `llama-index` zeigen verschiedene Schritte zur Implementierung von RAG. Hier ist eine kurze Anleitung zur Nutzung der Hauptskripte:

1. **Dokumente laden**:

   ```bash
   python langchain/01_load_documents.py
   python llama-index/01_load_documents.py
   ```

2. **Vektordatenbank erstellen und Dokumente hinzufügen**:

   ```bash
   python langchain/04_write_documents_into_qdrant.py
   python llama-index/04_write_documents_into_qdrant.py
   ```

3. **Anfragen stellen und Antworten generieren**:

   ```bash
   python langchain/05_query_qdrant.py
   python llama-index/05_query_qdrant_comparison.py
   ```

## Beispiele

- `00_split_text_semantic.py` und `00_split_text.py`: Skripte zum Teilen von Texten.
- `01_load_documents.py`: Dokumente in das System laden.
- `02_vector_db_fs.py` und `03_vector_db_qdrant.py`: Vektordatenbanken erstellen.
- `04_write_documents_into_qdrant.py`: Dokumente in die Vektordatenbank schreiben.
- `05_query_qdrant.py`: Anfragen an die Vektordatenbank stellen und Antworten generieren.
- `06_llm_lmstudio.py` und `06_llm_ollama.py`: Nutzung von verschiedenen LLMs.
- `07_rag.py`: Hauptskript zur Durchführung von RAG.
- `10_chunk_bechmark.py`: Benchmarking von Chunks.

## Ergebnisse

Die Benchmark-Ergebnisse sind in der Datei `benchmark_results.csv` zu finden.

## Beitragende

- Tobias Oberrauch

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Siehe die [LICENSE-Datei](LICENSE) für Details.
