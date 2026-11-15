import arxiv
import pandas as pd


def create_edges(data_path: str, idx_columns: list[str]): # -> list[list[int]]:
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        ValueError("Возможно, по данному пути не существует файла")

    arxiv_idx = df[idx_column]
    arxId_to_index = { arxiv_id: i for i, arxiv_id in enumerate (arxiv_idx) }
    edge_index = []


    for arxiv_id in arxiv_idx:
        # Запрос ссылок на данную статью

        # Если есть в нашей мапе, то создаем ребро!
        # Добавлять 2 ребра или одно???
        pass

    pass

