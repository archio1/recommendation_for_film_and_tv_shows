import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree, dropout_edge


class LightGCNConv(MessagePassing):
    """Light Graph Convolution слой"""

    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    """
    Hybrid LightGCN v2 (Residual + Deep Normalization).
    Архитектура изменена для предотвращения коллапса эмбеддингов.
    """

    def __init__(
            self,
            num_users: int,
            num_items: int,
            num_genres: int = 0,
            embedding_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
            edge_dropout: float = 0.2,  # NEW: dropout на рёбрах графа
            use_layer_weights: bool = True  # NEW: обучаемые веса слоёв
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.num_genres = num_genres
        self.use_layer_weights = use_layer_weights

        # 1. ID Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)

        # 2. Content Projections (если есть жанры)
        if num_genres > 0:
            self.genre_encoder = nn.Linear(num_genres, embedding_dim)
            self.year_encoder = nn.Linear(1, embedding_dim)

        # 3. GCN слои
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

        # 4. NEW: Обучаемые веса для агрегации слоёв
        if use_layer_weights:
            # num_layers + 1 потому что включаем начальные эмбеддинги (layer 0)
            self.layer_weights = nn.Parameter(torch.ones(num_layers + 1) / (num_layers + 1))

        self._init_weights()

    def _init_weights(self):
        # Xavier init с большей дисперсией для лучшего разделения
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1.5)
        nn.init.xavier_uniform_(self.item_id_embedding.weight, gain=1.5)
        if self.num_genres > 0:
            nn.init.xavier_uniform_(self.genre_encoder.weight, gain=1.0)
            nn.init.xavier_uniform_(self.year_encoder.weight, gain=1.0)

    def get_item_embedding(self, item_features=None):
        """
        Создает вектор фильма БЕЗ нормализации на этом этапе.
        """
        id_emb = self.item_id_embedding.weight

        if item_features is not None and self.num_genres > 0:
            genre_vecs, year_vecs = item_features

            # Проекция фичей (БЕЗ tanh — позволяем большую дисперсию)
            genre_emb = self.genre_encoder(genre_vecs)
            year_emb = self.year_encoder(year_vecs)

            # УВЕЛИЧЕННЫЕ коэффициенты для большего влияния контента
            # Это создаёт большее разнообразие в начальных эмбеддингах
            combined = id_emb + (0.5 * genre_emb) + (0.2 * year_emb)
        else:
            combined = id_emb

        # НЕ нормализуем здесь!
        return combined

    def forward(self, edge_index, item_features=None, normalize_output=False):
        """
        Forward pass с опциональной нормализацией.

        Args:
            edge_index: граф
            item_features: (genre_matrix, year_tensor) или None
            normalize_output: нормализовать ли финальные эмбеддинги
        """
        # 1. Начальные эмбеддинги (БЕЗ нормализации)
        user_emb = self.user_embedding.weight
        item_emb = self.get_item_embedding(item_features)

        x = torch.cat([user_emb, item_emb])
        embeddings = [x]

        # 2. Edge dropout (только при обучении)
        if self.training and self.edge_dropout > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=True)

        # 3. Проходим по слоям
        for conv in self.convs:
            x = conv(x, edge_index)
            # БЕЗ нормализации между слоями!

            if self.dropout > 0 and self.training:
                x = F.dropout(x, p=self.dropout, training=True)
            embeddings.append(x)

        # 4. Агрегация слоёв
        if self.use_layer_weights:
            # Softmax для нормализации весов в [0,1] с суммой 1
            weights = F.softmax(self.layer_weights, dim=0)
            final_embedding = sum(w * emb for w, emb in zip(weights, embeddings))
        else:
            final_embedding = torch.stack(embeddings, dim=0).mean(dim=0)

        # 5. Опциональная финальная нормализация
        if normalize_output:
            final_embedding = F.normalize(final_embedding, p=2, dim=1)

        return final_embedding[:self.num_users], final_embedding[self.num_users:]


class BPRLoss(nn.Module):
    """
    BPR Loss с temperature scaling.

    Temperature < 1: делает распределение "острее" (больше разница между pos и neg)
    Temperature > 1: делает распределение "мягче"
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pos_scores, neg_scores):
        # Применяем temperature scaling
        diff = (pos_scores - neg_scores) / self.temperature
        return -F.logsigmoid(diff).mean()


class InfoNCELoss(nn.Module):
    """
    Contrastive Loss для форсированного разделения эмбеддингов.

    Идея: для каждого user-item положительной пары, все остальные items
    в батче считаются негативными. Это создаёт сильный сигнал для разделения.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, user_emb, pos_item_emb, neg_item_emb=None, all_item_emb=None):
        """
        Args:
            user_emb: [batch_size, dim] — эмбеддинги пользователей
            pos_item_emb: [batch_size, dim] — позитивные items
            neg_item_emb: [batch_size, dim] — негативные items (опционально)
            all_item_emb: [num_items, dim] — все items для in-batch negatives
        """
        # Нормализуем для косинусного сходства
        user_emb = F.normalize(user_emb, p=2, dim=1)
        pos_item_emb = F.normalize(pos_item_emb, p=2, dim=1)

        # Позитивные скоры
        pos_scores = (user_emb * pos_item_emb).sum(dim=1) / self.temperature  # [batch]

        if all_item_emb is not None:
            # In-batch negatives: используем ВСЕ items в батче как негативные
            all_item_emb = F.normalize(all_item_emb, p=2, dim=1)
            all_scores = torch.matmul(user_emb, all_item_emb.t()) / self.temperature  # [batch, num_items]

            # InfoNCE: log(exp(pos) / sum(exp(all)))
            # = pos - logsumexp(all)
            loss = -pos_scores + torch.logsumexp(all_scores, dim=1)
        else:
            # Простой вариант с одним негативом
            neg_item_emb = F.normalize(neg_item_emb, p=2, dim=1)
            neg_scores = (user_emb * neg_item_emb).sum(dim=1) / self.temperature

            # Упрощённый contrastive: log(exp(pos) / (exp(pos) + exp(neg)))
            logits = torch.stack([pos_scores, neg_scores], dim=1)
            labels = torch.zeros(user_emb.size(0), device=user_emb.device, dtype=torch.long)
            loss = F.cross_entropy(logits, labels)

        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Комбинированный loss: BPR + InfoNCE.

    BPR отвечает за ранжирование, InfoNCE — за разделение эмбеддингов.
    """

    def __init__(
            self,
            bpr_temperature: float = 0.5,  # < 1 для "острого" BPR
            infonce_temperature: float = 0.1,
            infonce_weight: float = 0.1  # вес contrastive loss
    ):
        super().__init__()
        self.bpr_loss = BPRLoss(temperature=bpr_temperature)
        self.infonce_loss = InfoNCELoss(temperature=infonce_temperature)
        self.infonce_weight = infonce_weight

    def forward(self, user_emb, pos_item_emb, neg_item_emb, pos_scores, neg_scores):
        """
        Args:
            user_emb, pos_item_emb, neg_item_emb: эмбеддинги для InfoNCE
            pos_scores, neg_scores: скоры для BPR (уже вычисленные dot products)
        """
        bpr = self.bpr_loss(pos_scores, neg_scores)
        infonce = self.infonce_loss(user_emb, pos_item_emb, neg_item_emb)

        return bpr + self.infonce_weight * infonce, {
            'bpr': bpr.item(),
            'infonce': infonce.item()
        }

# ============================================================================
# АЛЬТЕРНАТИВНАЯ АРХИТЕКТУРА: LightGCN с Directional Margin
# ============================================================================

class DirectionalMarginLoss(nn.Module):
    """
    Margin-based loss, который явно требует минимальную разницу между pos и neg.

    L = max(0, margin - (pos_score - neg_score))

    Это ЗАСТАВЛЯЕТ модель создавать разницу не меньше margin.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores, neg_scores):
        diff = pos_scores - neg_scores
        loss = F.relu(self.margin - diff)
        return loss.mean()
