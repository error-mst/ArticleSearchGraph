import requests
import re

# Endpoint - /works
BASE_URL = "https://api.openalex.org/works"

def get_json(arxiv_id: str):
    # base url
    url = f"{BASE_URL}?filter=doi:10.48550/arXiv.{arxiv_id}"

    response = requests.get(url, timeout=30)
    return response.json()

def get_referenced_works(article_json) -> tuple(str) | None:
    # Возвращаем ссылки на referenced works
    if not article_json.get("results"):
        raise KeyError(f"Отсутствует ключ: results")
    if not article_json["results"].get(0):
        raise KeyError(f"Не было найдено ни одно результата")
    if not article_json["results"][0].get("referenced_works"):
        raise KeyError(f"Не было найдено ни одной работы, на которую ссылаются")
    
    return article_json["results"][0]["referenced_works"]
    
def get_arxiv_id(url: str):
    pattern = r"arxiv\.(\d{4}\.\d{4,5})"
    match = re.search(pattern, url)

    if match:
        return match.group(0), match.group(1)
    else:
        return None
