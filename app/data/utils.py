import pandas as pd
from dataclasses import dataclass, field, asdict

@dataclass
class GraphData:
    articles: list[str]
    article_data: dict[str, dict[str, str]]
    graph_table: pd.DataFrame
    arxiv_id2index: dict[str, int]
    index2arxiv_id: dict[int, str]
    edges: list[list[int]]
     
