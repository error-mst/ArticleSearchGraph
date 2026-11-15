
import requests
from app.data.collect_open_alex_data import get_json, get_arxiv_id,  get_referenced_works, get_list_ref_works
from app.data.generate_graph import generate_graph_data

generate_graph_data('cat:cs.LG AND abs:"graph neural network"')

