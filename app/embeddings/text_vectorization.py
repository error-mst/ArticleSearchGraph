import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer


def vectorize_data(data_path: str, model_name: str, columns: list[str]):
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Ошибка: {e}")

    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        ValueError("Проверьте название модели")

    # Форматирование текстовых данных и преобразование в эмбеддинги
    texts = ( df[columns].fillna("").astype(str).agg("\n\n".join, axis=1).str.strip().tolist() )
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    return embeddings
