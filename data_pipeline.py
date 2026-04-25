"""
data_pipeline.py
================
Capstone Project Module 3 - IMDB Top 1000 Movies
Pipeline: CSV → Clean → SQLite + Qdrant-ready documents

Jalankan script ini SEKALI sebelum ingest ke Qdrant.
Output:
  - movies.db  (SQLite, untuk Revenue/Rating/Director/Era agent)
  - qdrant_docs.pkl (list of dicts, siap di-embed & di-insert ke Qdrant)
"""

import pandas as pd
import sqlite3
import pickle
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATH    = "imdb_top_1000.csv"
DB_PATH     = "movies.db"
QDRANT_DOCS = "qdrant_docs.pkl"


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[LOAD] {len(df)} rows loaded.")
    return df


# ─────────────────────────────────────────────
# 2. CLEAN & TRANSFORM
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # ── Released_Year: ada 1 anomali 'PG' (Apollo 13 → 1995)
    df["Released_Year"] = df["Released_Year"].replace("PG", "1995")
    df["Released_Year"] = pd.to_numeric(
        df["Released_Year"], errors="coerce"
    ).astype("Int64")

    # ── Runtime: '142 min' → 142
    df["Runtime"] = (
        df["Runtime"]
        .str.replace(" min", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )

    # ── Gross: '28,341,469' → 28341469 (Int, nullable karena 169 null)
    df["Gross"] = (
        df["Gross"]
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )

    # ── Meta_score: 157 null → 0 
    df["Meta_score"] = df["Meta_score"].fillna(0).astype(int)

    # ── Certificate: 101 null → 'Not Rated'
    df["Certificate"] = df["Certificate"].fillna("Not Rated")

    # ── Cast: gabungin Star1-4 jadi satu kolom string
    df["Cast"] = df[["Star1", "Star2", "Star3", "Star4"]].apply(
        lambda x: ", ".join(x.dropna()), axis=1
    )

    # ── Drop kolom yang tidak dipakai
    df = df.drop(columns=["Poster_Link"])

    print(f"[CLEAN] Done. Null remaining:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ─────────────────────────────────────────────
# 3A. EXPORT → SQLite
# ─────────────────────────────────────────────
def export_to_sqlite(df: pd.DataFrame, db_path: str):
    """
    Simpan ke SQLite untuk agent yang butuh:
    - Filter/sort by rating, gross, year
    - Exact match by director, cast, title
    - Agregasi (MAX, GROUP BY, dll)
    """
    # Kolom yang masuk SQLite (semua kecuali duplikat)
    sql_cols = [
        "Series_Title", "Released_Year", "Certificate", "Runtime",
        "Genre", "IMDB_Rating", "Meta_score", "Director",
        "Cast", "No_of_Votes", "Gross", "Overview"
    ]

    conn = sqlite3.connect(db_path)
    df[sql_cols].to_sql("movies", conn, if_exists="replace", index=False)

    # Index untuk kolom yang sering di-query
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rating ON movies(IMDB_Rating)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gross  ON movies(Gross)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_year   ON movies(Released_Year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dir    ON movies(Director)")
    conn.commit()
    conn.close()

    print(f"[SQLite] Saved to '{db_path}' — table: movies ({len(df)} rows)")


# ─────────────────────────────────────────────
# 3B. EXPORT → Qdrant-ready documents
# ─────────────────────────────────────────────
def export_to_qdrant_docs(df: pd.DataFrame, output_path: str):
    """
    Siapkan list of dicts untuk di-embed & di-insert ke Qdrant.

    Setiap item punya:
      - page_content : teks yang di-embed (semantic search)
      - metadata     : payload yang disimpan di Qdrant (untuk display & filter)
    """
    docs = []
    for _, row in df.iterrows():

        # ── page_content: gabungan info penting untuk semantic similarity
        page_content = (
            f"Title: {row['Series_Title']} ({row['Released_Year']})\n"
            f"Genre: {row['Genre']}\n"
            f"Director: {row['Director']}\n"
            f"Cast: {row['Cast']}\n"
            f"Overview: {row['Overview']}"
        )

        # ── metadata: disimpan sebagai Qdrant payload
        metadata = {
            "title"       : row["Series_Title"],
            "year"        : int(row["Released_Year"]) if pd.notna(row["Released_Year"]) else None,
            "genre"       : row["Genre"],
            "director"    : row["Director"],
            "cast"        : row["Cast"],
            "rating"      : float(row["IMDB_Rating"]),
            "meta_score"  : int(row["Meta_score"]),
            "runtime"     : int(row["Runtime"]) if pd.notna(row["Runtime"]) else None,
            "certificate" : row["Certificate"],
            "votes"       : int(row["No_of_Votes"]),
            "gross"       : int(row["Gross"]) if pd.notna(row["Gross"]) else None,
            "overview"    : row["Overview"],
        }

        docs.append({
            "page_content": page_content,
            "metadata"    : metadata
        })

    with open(output_path, "wb") as f:
        pickle.dump(docs, f)

    print(f"[Qdrant Docs] {len(docs)} documents saved to '{output_path}'")
    print(f"\nSample page_content:\n{'-'*40}\n{docs[0]['page_content']}\n{'-'*40}")
    print(f"\nSample metadata:\n{docs[0]['metadata']}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(CSV_PATH)
    df = clean_data(df)
    export_to_sqlite(df, DB_PATH)
    export_to_qdrant_docs(df, QDRANT_DOCS)
    print("\n✅ Pipeline complete!")
    print(f"   → SQLite  : {DB_PATH}")
    print(f"   → Qdrant  : {QDRANT_DOCS}")
