import pandas as pd
from langchain.docstore.document import Document
from pathlib import Path

def load_dataset(dataset_path):
    dataset_path = Path(dataset_path)  # Преобразуем в Path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    df = pd.read_csv(dataset_path)
    documents = []
    for _, row in df.iterrows():
        document = Document(
            page_content=row['combined'],
            metadata={
                'title': row['title'],
                'genres': row['genres'],
                'overview': row['overview'],
                'weighted_rate': row['weighted_rate']
            }
        )
        documents.append(document)
    return documents