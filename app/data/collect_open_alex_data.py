import requests
import re
import asyncio
import aiohttp

from collections import defaultdict

# Endpoint - /works
BASE_URL = "https://api.openalex.org/works"
ARXIV_ARTICLES: dict[str, dict] = {}
REFERENCED_WORKS: dict[str, list[str]] = defaultdict(list)

N_QUERIES = 5
SEMAPHORE = asyncio.Semaphore(N_QUERIES)

DELAY = 0.15

async def get_json(arxiv_id: str, session: aiohttp.ClientSession):
    # base url
    url = f"{BASE_URL}?filter=doi:10.48550/arXiv.{arxiv_id}"

    async with SEMAPHORE:
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 429:
                    return
                if response.status != 200:
                    print(f"{arxiv_id}: STATUS: {response.status}")
                
                ARXIV_ARTICLES[arxiv_id] = await response.json()
        except Exception as e:
            print(f"{arxiv_id}: ReqError: {e}")
        finally:
            await asyncio.sleep(DELAY)

def get_referenced_works(article_json):
    # Возвращаем ссылки на referenced works
    if not article_json.get("results"):
        return []

    return article_json["results"][0].get("referenced_works", [])


async def get_arxiv_id(arxiv_id: str, url: str, session: aiohttp.ClientSession):
    # Извлекаем OpenAlex id
    oa_pattern = r"W\d*"
    oa_id = re.search(oa_pattern, url)
    
    if not oa_id:
        return
    oa_id = oa_id.group(0)
    # Запрос на получение данных о статье
    
    async with SEMAPHORE:
        try: 
            params = {
                "select": "id,ids,title"
            }
            async with session.get(f"{BASE_URL}/{oa_id}", params=params, timeout=50) as response:
                if response.status != 200:
                    return 
                data = await response.json()

            pattern = r"arxiv\.(\d{4}\.\d{4,5})"

            ids = data.get("ids", [])
            doi = data.get("doi")
            if not doi:
                return

            ref_arxiv_id = re.search(pattern, doi).group(1)
            if ref_arxiv_id not in ARXIV_ARTICLES:
                return

            REFERENCED_WORKS[arxiv_id].append(ref_arxiv_id)

        except Exception as e:
            KeyError(f"Ошибка при обработке: {oa_id}")

        finally:
            asyncio.sleep(SEMAPHORE)

#def get_list_ref_works(arxiv_id: str) -> list[str] | None:
#    json = get_json(arxiv_id)
#    try:
#        ref_works = list(map(get_arxiv_id, get_referenced_works(json)))
#        return ref_works
#    except Exception as e:
#        print(f"Ошибка: {e}")
#        return


async def get_all_ref_works(articles: list[str]):
    tasks = []

    async with aiohttp.ClientSession() as session:
        print(20 * "-")
        print("START GETTING DATA")
        tasks = [asyncio.create_task(get_json(article, session)) for article in articles]
        await asyncio.gather(*tasks)
        print("DATA ARE DONE")    
        print(20 * "-")
        print("START REF WORKS")

        tasks = []
        for arxiv_id, article_json in ARXIV_ARTICLES.items():
            for ref_work in get_referenced_works(article_json):
                task = asyncio.create_task(get_arxiv_id(arxiv_id, ref_work, session))
                tasks.append(task)
        await asyncio.gather(*tasks)
        print("DONE") 
        print(20 * "-")
def get_data(articles: list[str]):
    asyncio.run(get_all_ref_works(articles))
    return ARXIV_ARTICLES, REFERENCED_WORKS
