from langchain_community.tools.tavily_search import TavilySearchResults

def get_web_search_tool():
    """
    Cari informasi terbaru seputar film, berita selebriti, 
    dan jadwal rilis film dari internet secara real-time.
    """
    return TavilySearchResults(k=3)