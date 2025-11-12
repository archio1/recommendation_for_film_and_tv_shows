import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class LightGCNConv(MessagePassing):
    """Light Graph Convolution слой для LightGCN"""

    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        # Нормализация по степени вершин
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Пропагация сообщений
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    """
    LightGCN модель для рекомендаций фильмов/сериалов

    Paper: "LightGCN: Simplifying and Powering Graph Convolution Network
            for Recommendation" (SIGIR 2020)
    """

    def __init__(
            self,
            num_users: int,
            num_items: int,
            embedding_dim: int = 64,
            num_layers: int = 3,
            dropout: float = 0.0
    ):
        """
        Args:
            num_users: Количество пользователей
            num_items: Количество фильмов/сериалов
            embedding_dim: Размерность эмбеддингов
            num_layers: Количество GCN слоев
            dropout: Вероятность dropout (обычно 0 для LightGCN)
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Эмбеддинги для пользователей и фильмов
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # LightGCN слои
        self.convs = nn.ModuleList([
            LightGCNConv() for _ in range(num_layers)
        ])

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Xavier инициализация"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index):
        """
        Forward pass

        Args:
            edge_index: Граф связей [2, num_edges]

        Returns:
            user_embeddings, item_embeddings
        """
        # Начальные эмбеддинги
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ])

        # Сохраняем эмбеддинги с каждого слоя
        embeddings = [x]

        # Пропускаем через GCN слои
        for conv in self.convs:
            x = conv(x, edge_index)
            if self.dropout is not None:
                x = self.dropout(x)
            embeddings.append(x)

        # Усредняем эмбеддинги со всех слоёв (ключевая идея LightGCN)
        final_embedding = torch.stack(embeddings, dim=0).mean(dim=0)

        # Разделяем на user и item эмбеддинги
        user_emb = final_embedding[:self.num_users]
        item_emb = final_embedding[self.num_users:]

        return user_emb, item_emb

    def predict(self, user_emb, item_emb, user_ids, item_ids):
        """
        Предсказание рейтингов

        Args:
            user_emb: User embeddings [num_users, dim]
            item_emb: Item embeddings [num_items, dim]
            user_ids: ID пользователей [batch_size]
            item_ids: ID фильмов [batch_size]

        Returns:
            Предсказанные рейтинги [batch_size]
        """
        users = user_emb[user_ids]
        items = item_emb[item_ids]

        # Скалярное произведение
        scores = (users * items).sum(dim=1)
        return scores

    def recommend(self, user_emb, item_emb, user_id, top_k=10, exclude_items=None):
        """
        Рекомендации для пользователя

        Args:
            user_emb: User embeddings
            item_emb: Item embeddings
            user_id: ID пользователя
            top_k: Количество рекомендаций
            exclude_items: Список ID фильмов для исключения

        Returns:
            top_k_items: ID рекомендованных фильмов
            top_k_scores: Скоры
        """
        user = user_emb[user_id].unsqueeze(0)  # [1, dim]

        # Вычисляем скоры для всех фильмов
        scores = torch.matmul(user, item_emb.t()).squeeze(0)  # [num_items]

        # Исключаем уже просмотренные
        if exclude_items is not None:
            scores[exclude_items] = -float('inf')

        # Топ-K
        top_k_scores, top_k_items = torch.topk(scores, top_k)

        return top_k_items.cpu().numpy(), top_k_scores.cpu().numpy()


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss для обучения"""

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: Скоры для положительных примеров
            neg_scores: Скоры для отрицательных примеров
        """
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss


# Пример использования
if __name__ == "__main__":
    # Параметры
    num_users = 1000
    num_items = 5000
    embedding_dim = 64

    # Создаем модель
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=3
    )

    # Пример графа (user-item interactions)
    # edge_index[0] - user IDs, edge_index[1] - item IDs (сдвинутые на num_users)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2],  # users
        [num_users, num_users + 1, num_users + 2, num_users + 3, num_users]  # items
    ])

    # Forward pass
    user_emb, item_emb = model(edge_index)
    print(f"User embeddings shape: {user_emb.shape}")
    print(f"Item embeddings shape: {item_emb.shape}")

    # Рекомендации для пользователя 0
    recommended_items, scores = model.recommend(user_emb, item_emb, user_id=0, top_k=5)
    print(f"\nТоп-5 рекомендаций для пользователя 0:")
    print(f"Items: {recommended_items}")
    print(f"Scores: {scores}")