# 🎬 CineBot — Hierarchical Multi-Agent Movie Assistant

Capstone Project Module 3 — Purwadhika AI Engineering

---

## 📌 Deskripsi

CineBot adalah chatbot rekomendasi film berbasis **LangGraph Hierarchical Multi-Agent** yang mampu menjawab berbagai pertanyaan seputar film menggunakan tiga sumber data:

- 🗄️ **SQLite** — Data IMDB Top 1000 (rating, revenue, sutradara, cast, era)
- 🔍 **Qdrant Vector DB** — Semantic search & rekomendasi film mirip
- 🌐 **Tavily Web Search** — Informasi film terkini dari internet

Seluruh aktivitas agent di-track menggunakan **Langfuse** untuk observability.

---

## 🏗️ Arsitektur

```
User
└── Main Supervisor
    ├── RAG Department
    │   ├── Semantic Search Agent  (Qdrant)
    │   └── Similar Movie Agent    (Qdrant)
    ├── SQL Department
    │   ├── Revenue Agent          (SQLite)
    │   ├── Rating Agent           (SQLite)
    │   ├── Director/Cast Agent    (SQLite)
    │   └── Era Agent              (SQLite)
    └── WEB Department
        └── Web Search Agent       (Tavily API)
```

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|---|---|
| Framework | LangGraph + LangChain |
| LLM | OpenAI GPT-4o-mini |
| Vector DB | Qdrant Cloud |
| SQL DB | SQLite (IMDB Top 1000) |
| Web Search | Tavily API |
| Observability | Langfuse |
| Frontend | Streamlit |

---

## ⚙️ Instalasi

### 1. Clone repository
```bash
git clone https://github.com/username/capstone-project-3.git
cd capstone-project-3
```

### 2. Buat virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment variables
```bash
cp .env.example .env
# Edit .env dan isi dengan API keys kamu
```

### 5. Ingest data ke Qdrant
```bash
python ingest_qdrant.py
```

### 6. Jalankan aplikasi
```bash
streamlit run app.py
```

---

## 🔑 Environment Variables

Buat file `.env` berdasarkan `.env.example`:

```
OPENAI_API_KEY=your_openai_api_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_HOST=https://us.cloud.langfuse.com
TAVILY_API_KEY=your_tavily_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

---

## 💬 Contoh Query

| Tipe | Contoh Query |
|---|---|
| Rekomendasi by vibe | "Film dengan vibe dark kayak Black Swan" |
| Film mirip | "Film mirip Interstellar" |
| Rating tertinggi | "Top 5 film rating tertinggi" |
| Revenue | "Film terlaris sepanjang masa" |
| Sutradara | "Semua film Christopher Nolan" |
| Era | "Film terbaik era 90an" |
| Web search | "Film Marvel terbaru 2025" |

---

## 📁 Struktur Project

```
capstone-project-3/
├── app.py                 # Streamlit frontend
├── agents.py              # LangGraph multi-agent logic
├── data_pipeline.py       # Data preprocessing
├── ingest_qdrant.py       # Qdrant ingestion script
├── tools/
│   ├── rag_tool.py        # Qdrant semantic search tools
│   ├── sql_tool.py        # SQLite query tools
│   └── web_tool.py        # Tavily web search tool
├── movies.db              # SQLite database
├── imdb_top_1000.csv      # Raw dataset
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔭 Observability

CineBot menggunakan **Langfuse** untuk tracking:
- Traces per query
- Token usage & cost
- LLM call latency
- Agent routing path

---

## ⚠️ Known Limitations

- Latency 15-30 detik per query karena multiple LLM calls (hierarchical architecture)
- Data film terbatas pada IMDB Top 1000 untuk query SQL/RAG
- Query web search bergantung pada koneksi internet dan Tavily API

---

## 🚀 Future Improvements

- Streaming response dari LLM
- Cache untuk query yang sering ditanyakan
- Keyword-based routing di sub-supervisor untuk mengurangi latency
- Persistent chat history

---

## 👤 Author

Fajri Algifari
