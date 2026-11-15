import arxiv
import requests
import pandas as pd


def get_articles(query: str, max_results: int = 5000):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    return client.results(search)

def get_structured_data(query: str, max_results: int = 5000):
    data = []
    articles_ = []

    articles = get_articles(query, max_results=max_results)
    for article in articles:
        articles_.append(article.get_short_id())
        
        data.append({
            "arxiv_id": article.get_short_id(),
            "title": article.title,
            "abstract": article.summary,
            "authors": ", ".join(author.name for author in article.authors),
            "published": article.published,
            "updated": article.updated,
            "categories": ", ".join(article.categories),
             "pdf_url": article.pdf_url,
        })

    return articles_, pd.DataFrame(data)
