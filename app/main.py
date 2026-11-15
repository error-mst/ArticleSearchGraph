import pandas as pd
import numpy as np

from .data.collect_arxiv_data import get_structured_data
print(get_structured_data('cat:cs.LG AND abs:"graph neural network"'))
