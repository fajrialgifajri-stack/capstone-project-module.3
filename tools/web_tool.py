from langchain_community.tools.tavily_search import TavilySearchResults
import os

# Set API Key (Bisa masukin ke file .env lo)
os.environ["TAVILY_API_KEY"] = "tvly-xxx-lo-punya"

def get_web_search_tool():
    """
    Cari informasi terbaru seputar film, berita selebriti, 
    dan jadwal rilis film dari internet secara real-time.
    """
    return TavilySearchResults(k=3)