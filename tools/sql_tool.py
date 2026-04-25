"""
tools/sql_tool.py
=================
Empat tool yang pakai SQLite (movies.db):
1. revenue_tool      — film dengan gross tertinggi
2. rating_tool       — film dengan rating tertinggi
3. director_cast_tool — cari film by sutradara atau aktor
4. era_tool          — cari film by dekade / rentang tahun
"""

import os
import sqlite3
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "movies.db")


def _query(sql: str, params: tuple = ()) -> list[dict]:
    """Helper: jalankan SQL query dan return list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _format_movie(row: dict, rank: int) -> str:
    gross = f"${row['Gross']:,}" if row.get("Gross") else "N/A"
    return (
        f"{rank}. {row['Series_Title']} ({row['Released_Year']}) | "
        f"⭐ {row['IMDB_Rating']} | {row['Genre']}\n"
        f"   Director: {row['Director']}\n"
        f"   Gross: {gross} | Votes: {row.get('No_of_Votes', 'N/A'):,}\n"
        f"   {row.get('Overview', '')[:120]}..."
    )


@tool
def revenue_tool(top_n: int = 10) -> str:
    """
    Cari film dengan pendapatan (gross) tertinggi.
    Gunakan tool ini untuk pertanyaan seperti:
    - 'Film dengan penghasilan terbesar'
    - 'Film terlaris sepanjang masa'
    - 'Top 5 film box office tertinggi'
    Masukkan jumlah film yang ingin ditampilkan (default 10).
    """
    top_n = min(max(top_n, 1), 20)
    rows = _query(
        """
        SELECT Series_Title, Released_Year, IMDB_Rating, Genre,
               Director, Gross, No_of_Votes, Overview
        FROM movies
        WHERE Gross IS NOT NULL
        ORDER BY Gross DESC
        LIMIT ?
        """,
        (top_n,)
    )
    if not rows:
        return "Data revenue tidak tersedia."

    lines = [f"Top {top_n} film berdasarkan gross revenue:\n"]
    for i, row in enumerate(rows, 1):
        lines.append(_format_movie(row, i))
    return "\n\n".join(lines)


@tool
def rating_tool(top_n: int = 10, min_votes: int = 10000) -> str:
    """
    Cari film dengan rating IMDB tertinggi.
    Gunakan tool ini untuk pertanyaan seperti:
    - 'Film terbaik berdasarkan rating'
    - 'Film dengan nilai IMDB tertinggi'
    - 'Top 10 film rating tertinggi'
    Masukkan jumlah film (default 10) dan minimum votes (default 10000).
    """
    top_n = min(max(top_n, 1), 20)
    rows = _query(
        """
        SELECT Series_Title, Released_Year, IMDB_Rating, Genre,
               Director, Gross, No_of_Votes, Overview
        FROM movies
        WHERE No_of_Votes >= ?
        ORDER BY IMDB_Rating DESC, No_of_Votes DESC
        LIMIT ?
        """,
        (min_votes, top_n)
    )
    if not rows:
        return "Tidak ada film yang ditemukan."

    lines = [f"Top {top_n} film berdasarkan IMDB rating:\n"]
    for i, row in enumerate(rows, 1):
        lines.append(_format_movie(row, i))
    return "\n\n".join(lines)


@tool
def director_cast_tool(name: str) -> str:
    """
    Cari film berdasarkan nama sutradara atau aktor/aktris.
    Gunakan tool ini untuk pertanyaan seperti:
    - 'Film yang disutradarai Christopher Nolan'
    - 'Film yang dibintangi Leonardo DiCaprio'
    - 'Filmografi Quentin Tarantino'
    Masukkan nama sutradara atau aktor.
    """
    pattern = f"%{name}%"
    rows = _query(
        """
        SELECT Series_Title, Released_Year, IMDB_Rating, Genre,
               Director, Cast, Gross, No_of_Votes, Overview
        FROM movies
        WHERE Director LIKE ? OR Cast LIKE ?
        ORDER BY IMDB_Rating DESC
        LIMIT 10
        """,
        (pattern, pattern)
    )
    if not rows:
        return f"Tidak ada film dengan sutradara/aktor '{name}'."

    lines = [f"Film dengan sutradara/aktor '{name}':\n"]
    for i, row in enumerate(rows, 1):
        lines.append(_format_movie(row, i))
    return "\n\n".join(lines)


@tool
def era_tool(start_year: int, end_year: int, top_n: int = 10) -> str:
    """
    Cari film terbaik dari era / dekade / rentang tahun tertentu.
    Gunakan tool ini untuk pertanyaan seperti:
    - 'Film terbaik era 90an'
    - 'Film klasik tahun 1970-1980'
    - 'Rekomendasi film dari tahun 2000 sampai 2010'
    Masukkan tahun mulai, tahun akhir, dan jumlah film (default 10).
    """
    top_n = min(max(top_n, 1), 20)
    rows = _query(
        """
        SELECT Series_Title, Released_Year, IMDB_Rating, Genre,
               Director, Gross, No_of_Votes, Overview
        FROM movies
        WHERE Released_Year BETWEEN ? AND ?
        ORDER BY IMDB_Rating DESC, No_of_Votes DESC
        LIMIT ?
        """,
        (start_year, end_year, top_n)
    )
    if not rows:
        return f"Tidak ada film dari tahun {start_year}-{end_year}."

    lines = [f"Top {top_n} film era {start_year}-{end_year}:\n"]
    for i, row in enumerate(rows, 1):
        lines.append(_format_movie(row, i))
    return "\n\n".join(lines)
