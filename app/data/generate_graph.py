from .collect_arxiv_data import get_structured_data 
from .collect_open_alex_data import get_data

def generate_graph_data(query: str):
    
    # Загружаю статьи в соответствии с запросом
    # Создаю ребра для связанных статей
    # Сохраняю информацию об узлах и ребрах

    # Возвращаю объект GraphData
    
    # dict: arxiv_id -> article data
    articles = get_structured_data(query)
    article_referenced_works = {}    
    
    index2arxiv_id = {
        i: arxiv_id 
        for i, arxiv_id in enumerate(articles.keys())
    }
    arxiv_id2index = {
        arxiv_id: i
        for i, arxiv_id in index2arxiv_id.items()
    }    
    
    edges = [] 
    from_ = []
    to_ = []

    data = get_data(articles)
    edges.append(from_)
    edges.append(to_)
    print(data)
    print(edges)
