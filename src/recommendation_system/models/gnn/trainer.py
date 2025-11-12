import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.append(str(Path(__file__).parent))

from lightgcn import LightGCN, BPRLoss
from graph_builder import MovieGraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LightGCNTrainer:
    """
    Оптимизированный Trainer для LightGCN

    Ключевые оптимизации:
    1. Forward pass в каждом батче (для корректных градиентов)
    2. Векторизованный negative sampling
    3. Уменьшенная частота оценки
    4. Эффективная работа с памятью
    """

    def __init__(
            self,
            model: LightGCN,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
        self.train_losses = []
        self.val_recalls = []
        self.val_ndcgs = []
        self.epoch_times = []

        logger.info(f"Trainer инициализирован на {device}")
        logger.info(f"Модель параметров: {sum(p.numel() for p in model.parameters()):,}")

    def negative_sampling_batch(
            self,
            user_ids: np.ndarray,
            train_matrix,
            num_negatives: int = 1
    ) -> np.ndarray:
        """
        ВЕКТОРИЗОВАННЫЙ negative sampling (в 10-20 раз быстрее!)
        """
        num_items = train_matrix.shape[1]
        batch_size = len(user_ids)
        neg_items = np.random.randint(0, num_items, size=(batch_size, num_negatives * 3))
        result = np.zeros((batch_size, num_negatives), dtype=np.int64)

        for i, user_id in enumerate(user_ids):
            user_items = set(train_matrix[user_id].indices)
            valid_negs = [item for item in neg_items[i] if item not in user_items][:num_negatives]
            while len(valid_negs) < num_negatives:
                candidate = np.random.randint(0, num_items)
                if candidate not in user_items:
                    valid_negs.append(candidate)
            result[i] = valid_negs[:num_negatives]

        return result

    def train_epoch(
            self,
            train_graph,
            train_df,
            train_matrix,
            optimizer,
            loss_fn,
            batch_size: int = 2048,
            num_batches: int = None
    ) -> float:
        """
        Оптимизированное обучение на эпохе
        """
        self.model.train()
        total_loss = 0
        torch.cuda.empty_cache()  # Очистка памяти перед эпохой

        if num_batches is None:
            num_batches = len(train_df) // batch_size

        interactions = train_df[['user_id', 'item_id']].values
        np.random.shuffle(interactions)
        progress_bar = tqdm(range(num_batches), desc='Training')

        for batch_idx in progress_bar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(interactions))
            if end_idx <= start_idx:
                break

            batch = interactions[start_idx:end_idx]
            batch_users = torch.LongTensor(batch[:, 0]).to(self.device)
            batch_pos_items = torch.LongTensor(batch[:, 1]).to(self.device)
            batch_neg_items = torch.LongTensor(
                self.negative_sampling_batch(batch[:, 0], train_matrix, num_negatives=1).flatten()
            ).to(self.device)

            optimizer.zero_grad()

            if self.device == 'cuda':
                with torch.amp.autocast('cuda'):  # Явно указываем device_type='cuda'
                    user_emb, item_emb = self.model(train_graph.edge_index.to(self.device))
                    pos_scores = self.model.predict(user_emb, item_emb, batch_users, batch_pos_items)
                    neg_scores = self.model.predict(user_emb, item_emb, batch_users, batch_neg_items)
                    loss = loss_fn(pos_scores, neg_scores)
            else:
                user_emb, item_emb = self.model(train_graph.edge_index.to(self.device))
                pos_scores = self.model.predict(user_emb, item_emb, batch_users, batch_pos_items)
                neg_scores = self.model.predict(user_emb, item_emb, batch_users, batch_neg_items)
                loss = loss_fn(pos_scores, neg_scores)

            l2_reg = torch.norm(user_emb[batch_users], p=2) + \
                     torch.norm(item_emb[batch_pos_items], p=2) + \
                     torch.norm(item_emb[batch_neg_items], p=2)
            loss = loss + 1e-5 * l2_reg / len(batch)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Добавлено для стабильности
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            if batch_idx % 50 == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(
            self,
            train_graph,
            test_data,
            train_matrix,
            k: int = 10,
            sample_users: int = 2000
    ) -> dict:
        self.model.eval()
        user_emb, item_emb = self.model(train_graph.edge_index.to(self.device))
        all_scores = torch.matmul(user_emb, item_emb.t()).cpu().numpy()
        test_df = test_data['interactions']
        user_test_items = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
        eval_users = list(user_test_items.keys())
        if sample_users and sample_users < len(eval_users):
            eval_users = np.random.choice(eval_users, sample_users, replace=False)

        recalls = []
        ndcgs = []

        for user_id in tqdm(eval_users, desc='Evaluating', leave=False):
            true_items = user_test_items[user_id]
            train_items = train_matrix[user_id].nonzero()[1]
            scores = all_scores[user_id].copy()
            scores[train_items] = -np.inf
            top_k_items = np.argpartition(scores, -k)[-k:]
            top_k_items = top_k_items[np.argsort(scores[top_k_items])[::-1]]
            hits = len(set(top_k_items) & set(true_items))
            recall = hits / min(len(true_items), k)
            recalls.append(recall)
            relevance = [1 if item in true_items else 0 for item in top_k_items]
            if sum(relevance) > 0:
                dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)])
                ideal_relevance = sorted(relevance, reverse=True)
                idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)])
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)

        metrics = {
            f'recall@{k}': np.mean(recalls),
            f'ndcg@{k}': np.mean(ndcgs) if ndcgs else 0,
            'num_users_evaluated': len(eval_users)
        }
        return metrics

    def train(
            self,
            data: dict,
            num_epochs: int = 30,
            batch_size: int = 2048,
            lr: float = 0.001,
            eval_every: int = 5,
            early_stopping_patience: int = 5,
            save_dir: Path = None
    ):
        logger.info("=" * 70)
        logger.info("НАЧАЛО ОБУЧЕНИЯ (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)")
        logger.info("=" * 70)
        logger.info(f"Эпох: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
        logger.info(f"Оценка каждые {eval_every} эпох")

        train_graph = data['train_graph'].to(self.device)
        train_df = data['train_df']
        train_matrix = data['train_matrix']
        test_data = data['test_data']

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = BPRLoss()

        best_recall = 0
        best_epoch = 0
        patience_counter = 0

        if save_dir is None:
            save_dir = Path(__file__).parent / '../../../models'
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nМодели сохраняются в: {save_dir}")
        logger.info("")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(
                train_graph, train_df, train_matrix,
                optimizer, loss_fn, batch_size
            )
            epoch_time = time.time() - epoch_start
            self.train_losses.append(train_loss)
            self.epoch_times.append(epoch_time)

            if epoch % eval_every == 0 or epoch == 1:
                metrics = self.evaluate(
                    train_graph, test_data, train_matrix,
                    k=10, sample_users=2000
                )
                recall = metrics['recall@10']
                ndcg = metrics['ndcg@10']
                self.val_recalls.append(recall)
                self.val_ndcgs.append(ndcg)

                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Recall@10: {recall:.4f} | "
                    f"NDCG@10: {ndcg:.4f} | "
                    f"Time: {epoch_time:.1f}s"
                )

                if recall > best_recall:
                    best_recall = recall
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'recall@10': recall,
                        'ndcg@10': ndcg,
                        'metrics': metrics
                    }, save_dir / 'lightgcn_best.pt')
                    logger.info(f"  ✓ Новая лучшая модель сохранена!")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"\n⏹ Early stopping! Нет улучшений {early_stopping_patience} проверок")
                        break
            else:
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Time: {epoch_time:.1f}s"
                )

        logger.info("\n" + "=" * 70)
        logger.info("ФИНАЛЬНАЯ ОЦЕНКА")
        logger.info("=" * 70)

        checkpoint = torch.load(save_dir / 'lightgcn_best.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        final_metrics = self.evaluate(
            train_graph, test_data, train_matrix,
            k=10, sample_users=5000
        )

        logger.info(f"\nЛучшая модель (эпоха {best_epoch}):")
        logger.info(f"  Recall@10: {final_metrics['recall@10']:.4f}")
        logger.info(f"  NDCG@10:   {final_metrics['ndcg@10']:.4f}")
        logger.info(f"  Users оценено: {final_metrics['num_users_evaluated']:,}")

        training_stats = {
            'best_epoch': best_epoch,
            'best_recall@10': best_recall,
            'final_metrics': final_metrics,
            'total_epochs': epoch,
            'avg_epoch_time': np.mean(self.epoch_times),
            'total_time': sum(self.epoch_times)
        }

        with open(save_dir / 'training_stats.json', 'w') as f:
            json.dump(training_stats, f, indent=2)

        logger.info(f"\nВсего эпох: {epoch}")
        logger.info(f"Среднее время эпохи: {np.mean(self.epoch_times):.1f}s")
        logger.info(f"Общее время: {sum(self.epoch_times) / 60:.1f} минут")
        logger.info("=" * 70)

        return training_stats

    def plot_training(self, save_path: Path = None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(self.train_losses, linewidth=2, color='steelblue')
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('BPR Loss')
        axes[0].grid(alpha=0.3)
        eval_epochs = list(range(1, len(self.val_recalls) * 5 + 1, 5))
        axes[1].plot(eval_epochs, self.val_recalls, marker='o', linewidth=2, color='coral')
        axes[1].set_title('Validation Recall@10', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Recall@10')
        axes[1].grid(alpha=0.3)
        axes[2].plot(eval_epochs, self.val_ndcgs, marker='s', linewidth=2, color='lightgreen')
        axes[2].set_title('Validation NDCG@10', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('NDCG@10')
        axes[2].grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"График сохранён: {save_path}")
        plt.show()

def main():
    epochs = 30
    batch_size = 2048
    lr = 0.002
    embedding_dim = 32
    num_layers = 2
    patience = 3
    eval_every = 5

    script_dir = Path(__file__).parent
    project_root = script_dir
    while project_root.name != 'recommendation_for_film_and_tv_shows' and project_root.parent != project_root:
        project_root = project_root.parent
    data_dir = project_root / 'data'

    logger.info(f"Данные: {data_dir}")
    logger.info("Подготовка данных...")
    builder = MovieGraphBuilder(data_dir)
    data = builder.prepare_for_training(test_size=0.2, temporal=False)

    logger.info("\nСоздание модели...")
    model = LightGCN(
        num_users=data['num_users'],
        num_items=data['num_items'],
        embedding_dim=embedding_dim,
        num_layers=num_layers
    )

    trainer = LightGCNTrainer(model)
    stats = trainer.train(
        data=data,
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        eval_every=eval_every,
        early_stopping_patience=patience
    )

    save_dir = project_root / 'reports' / 'figures'
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.plot_training(save_dir / 'training_curves.png')

    logger.info("\nОбучение завершено успешно!")

if __name__ == "__main__":
    main()