from langchain_tavily import TavilySearch

def get_web_search_tool():
    """Cari info film terkini (rilis baru, berita, jadwal) di internet."""
    return TavilySearch(
        max_results=5,
        topic="general",
        time_range="year",
    )