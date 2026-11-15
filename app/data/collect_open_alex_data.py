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
    if not article_json["results"][0].get("referenced_works"):
        raise KeyError(f"Не было найдено ни одной работы, на которую ссылаются")
    
    return article_json["results"][0]["referenced_works"]
    
def get_arxiv_id(url: str):
    # Извлекаем OpenAlex id
    oa_pattern = r"W\d*"
    oa_id = re.search(oa_pattern, url).group(0)
    # Запрос на получение данных о статье
    params = {
        "select": "id,ids,title"
    }
    doi = requests.get(f"{BASE_URL}/{oa_id}", params=params, timeout=30)\
                    .json().get("ids", {})
    try:
        doi = doi.get("doi")
    except Exception as e:
        KeyError("Возможно статья не загржена на arXiv")
    # Извлекаем arxiv id
    pattern = r"arxiv\.(\d{4}\.\d{4,5})"
    match = re.search(pattern, doi)

    if match:
        return match.group(1)
    else:
        return None

def get_list_ref_works(arxiv_id: str) -> list[str] | None:
    json = get_json(arxiv_id)
    try:
        ref_works = list(map(get_arxiv_id, get_referenced_works(json)))
        return ref_works
    except Exception as e:
        print(f"Ошибка: {e}")
        return


