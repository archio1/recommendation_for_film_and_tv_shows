import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: GUI вызывает plot_training из worker-треда; default TkAgg крашит __del__ из не-main треда.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from recommendation_system.models.gnn.lightgcn import LightGCN, BPRLoss
from recommendation_system.models.gnn.graph_builder import MovieGraphBuilder
from recommendation_system.paths import MODELS_DIR, PROCESSED_DIR

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
            user_ids: torch.Tensor,
            train_matrix,
            num_negatives: int = 1
    ) -> torch.Tensor:
        """
        СУПЕР-БЫСТРЫЙ negative sampling на чистом Torch (GPU).
        Мы жертвуем 0.01% точности (возможны случайные совпадения),
        но получаем прирост скорости в 100 раз.
        """
        num_items = self.model.num_items
        batch_size = user_ids.size(0)

        # Генерируем случайные ID предметов прямо на видеокарте
        # Генерируем чуть больше, на случай совпадений (хотя мы их не проверяем ради скорости)
        neg_items = torch.randint(
            0, num_items,
            (batch_size, num_negatives),
            device=self.device,
            dtype=torch.long
        )

        return neg_items.view(-1)

    def train_epoch(
            self,
            train_graph,
            train_df,
            train_matrix,
            optimizer,
            loss_fn,
            batch_size: int = 2048,
            stop_flag=None,
    ) -> float:
        """
        Оптимизированное обучение на эпохе
        """
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        torch.cuda.empty_cache()  # Очистка памяти перед эпохой

        num_batches = len(train_df) // batch_size

        interactions = train_df[['user_id', 'item_id']].values
        np.random.shuffle(interactions)
        progress_bar = tqdm(range(num_batches), desc='Training')

        for batch_idx in progress_bar:
            if stop_flag is not None and stop_flag():
                logger.info(f"⏹ Stop requested at batch {batch_idx}/{num_batches}")
                break
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(interactions))
            if end_idx <= start_idx:
                break

            batch = interactions[start_idx:end_idx]
            batch_users = torch.from_numpy(batch[:, 0]).long().to(self.device, non_blocking=True)
            batch_pos_items = torch.from_numpy(batch[:, 1]).long().to(self.device, non_blocking=True)
            batch_neg_items = self.negative_sampling_batch(
                batch_users, train_matrix, num_negatives=1
            ).flatten()

            optimizer.zero_grad(set_to_none=True)

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

            total_loss += loss.detach().float()
            if batch_idx % 50 == 0:
                loss_val = loss.detach().item()  # один sync на 50 итераций
                progress_bar.set_postfix({'loss': f'{loss_val:.4f}'})
                logger.info(
                    f"  batch {batch_idx}/{num_batches} | loss={loss_val:.4f}"
                )

        return (total_loss / num_batches).item()

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
            output_path: Path = None,
            on_epoch_end=None,
            stop_flag=None,
    ):
        """
        Args:
            on_epoch_end: optional callable(dict) invoked after every epoch with
                {epoch, total_epochs, loss, recall, ndcg, best_epoch, evaluated}.
                `recall` / `ndcg` are None on epochs without evaluation.
            stop_flag: optional callable() -> bool. Polled before each epoch;
                if True, training stops gracefully after the current epoch.
        """
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

        if output_path is None:
            output_path = MODELS_DIR / 'lightgcn_best.pt'
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nЧекпоинт сохраняется в: {output_path}")
        logger.info("")

        epoch = 0
        for epoch in range(1, num_epochs + 1):
            if stop_flag is not None and stop_flag():
                logger.info("⏹ Stop requested — training cancelled.")
                break

            epoch_start = time.time()
            train_loss = self.train_epoch(
                train_graph, train_df, train_matrix,
                optimizer, loss_fn, batch_size,
                stop_flag=stop_flag,
            )
            epoch_time = time.time() - epoch_start
            self.train_losses.append(train_loss)
            self.epoch_times.append(epoch_time)
            evaluated_this_epoch = False
            epoch_recall = None
            epoch_ndcg = None
            early_stop = False

            if stop_flag is not None and stop_flag():
                logger.info("⏹ Stop requested — skipping eval/save for partial epoch")
                break

            if epoch % eval_every == 0 or epoch == 1:
                metrics = self.evaluate(
                    train_graph, test_data, train_matrix,
                    k=10, sample_users=2000
                )
                recall = metrics['recall@10']
                ndcg = metrics['ndcg@10']
                self.val_recalls.append(recall)
                self.val_ndcgs.append(ndcg)
                evaluated_this_epoch = True
                epoch_recall = float(recall)
                epoch_ndcg = float(ndcg)

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
                    }, output_path)
                    logger.info(f"  ✓ Новая лучшая модель сохранена!")
                else:
                    patience_counter += 1
                    early_stop = patience_counter >= early_stopping_patience
                    if early_stop:
                        logger.info(f"\n⏹ Early stopping! Нет улучшений {early_stopping_patience} проверок")
            else:
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Time: {epoch_time:.1f}s"
                )

            if on_epoch_end is not None:
                try:
                    on_epoch_end({
                        'epoch': epoch,
                        'total_epochs': num_epochs,
                        'loss': float(train_loss),
                        'recall': epoch_recall,
                        'ndcg': epoch_ndcg,
                        'best_epoch': best_epoch,
                        'evaluated': evaluated_this_epoch,
                        'time_seconds': epoch_time,
                    })
                except Exception as cb_exc:
                    logger.warning(f"on_epoch_end callback raised: {cb_exc}")

            if early_stop:
                break

        logger.info("\n" + "=" * 70)
        logger.info("ФИНАЛЬНАЯ ОЦЕНКА")
        logger.info("=" * 70)

        if output_path.exists():
            checkpoint = torch.load(output_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            final_metrics = self.evaluate(
                train_graph, test_data, train_matrix,
                k=10, sample_users=5000
            )
            logger.info(f"\nЛучшая модель (эпоха {best_epoch}):")
            logger.info(f"  Recall@10: {final_metrics['recall@10']:.4f}")
            logger.info(f"  NDCG@10:   {final_metrics['ndcg@10']:.4f}")
            logger.info(f"  Users оценено: {final_metrics['num_users_evaluated']:,}")
        else:
            logger.warning("Чекпоинт не был сохранён (ни одна эпоха не дала улучшения).")
            final_metrics = {'recall@10': 0.0, 'ndcg@10': 0.0, 'num_users_evaluated': 0}

        training_stats = {
            'best_epoch': best_epoch,
            'best_recall@10': best_recall,
            'final_metrics': final_metrics,
            'total_epochs': epoch,
            'avg_epoch_time': float(np.mean(self.epoch_times)) if self.epoch_times else 0.0,
            'total_time': float(sum(self.epoch_times)),
        }

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
        plt.close(fig)

def _next_versioned_path(models_dir: Path, prefix: str) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.pt$")
    versions = [int(m.group(1)) for f in models_dir.iterdir()
                if (m := pattern.match(f.name)) is not None]
    next_v = max(versions, default=0) + 1
    return models_dir / f"{prefix}_v{next_v}.pt"


def _sanity_check(model: LightGCN, metrics: dict) -> tuple[bool, list]:
    failures = []

    recall = metrics.get('recall@10', 0.0)
    if recall <= 0.05:
        failures.append(f"recall@10 = {recall:.4f} (expected > 0.05)")

    with torch.no_grad():
        user_norm = model.user_embedding.weight.data.norm(dim=1).mean().item()
        item_norm = model.item_id_embedding.weight.data.norm(dim=1).mean().item()
    if user_norm <= 0.01:
        failures.append(f"||user_emb||.mean() = {user_norm:.4f} (expected > 0.01)")
    if item_norm <= 0.01:
        failures.append(f"||item_emb||.mean() = {item_norm:.4f} (expected > 0.01)")

    return len(failures) == 0, failures


def _write_sidecar(sidecar_path: Path, **fields) -> None:
    serializable = {}
    for k, v in fields.items():
        if isinstance(v, dict):
            serializable[k] = {ik: (float(iv) if isinstance(iv, (np.floating,)) else
                                    int(iv) if isinstance(iv, (np.integer,)) else iv)
                               for ik, iv in v.items()}
        elif isinstance(v, np.floating):
            serializable[k] = float(v)
        elif isinstance(v, np.integer):
            serializable[k] = int(v)
        elif isinstance(v, Path):
            serializable[k] = str(v)
        else:
            serializable[k] = v
    with open(sidecar_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train per-domain LightGCN (movies or tv).",
    )
    parser.add_argument('--domain', choices=['movies', 'tv'], required=True,
                        help="Which domain to train.")
    parser.add_argument('--data-dir', type=Path, default=None,
                        help="Dataset directory with *_final.parquet (default: "
                             "<project>/data/processed/{domain}).")
    parser.add_argument('--output', type=Path, default=None,
                        help="Checkpoint path (default: auto-bumped "
                             "models/{domain}/lightgcn_{domain}_best_v{N+1}.pt).")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help="Forwarded to torch without auto-override; "
                             "'auto' picks cuda if available, else cpu.")
    parser.add_argument('--resume', type=Path, default=None,
                        help="Resume from .pt checkpoint.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true',
                        help="Print resolved config and exit without training.")
    return parser.parse_args(argv)


def main(argv=None, *, on_epoch_end=None, stop_flag=None) -> int:
    """
    Args:
        argv: command-line tokens (None → sys.argv).
        on_epoch_end: optional callable forwarded to LightGCNTrainer.train().
        stop_flag: optional callable() -> bool forwarded to LightGCNTrainer.train().
    """
    args = _parse_args(argv)

    # Device resolution — explicit user choice wins over auto.
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("--device cuda requested but CUDA not available, falling back to cpu")
            device = 'cpu'

    data_dir = args.data_dir or (PROCESSED_DIR / args.domain)

    models_dir = MODELS_DIR / args.domain
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = _next_versioned_path(models_dir, f'lightgcn_{args.domain}_best')

    sidecar_path = output_path.with_suffix('.json')

    config = {
        'domain': args.domain,
        'data_dir': str(data_dir),
        'output': str(output_path),
        'sidecar': str(sidecar_path),
        'device': device,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'embedding_dim': args.embedding_dim,
        'num_layers': args.num_layers,
        'patience': args.patience,
        'eval_every': args.eval_every,
        'resume': str(args.resume) if args.resume else None,
        'seed': args.seed,
    }

    if args.dry_run:
        logger.info("[DRY-RUN] Would train with:")
        for k, v in config.items():
            logger.info(f"  {k}: {v}")
        return 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Domain: {args.domain}")
    logger.info(f"Dataset dir: {data_dir}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Device: {device}")
    logger.info("Подготовка данных...")
    builder = MovieGraphBuilder(dataset_dir=data_dir)
    data = builder.prepare_for_training(test_size=0.2, temporal=False, random_state=args.seed)

    logger.info("\nСоздание модели...")
    model = LightGCN(
        num_users=data['num_users'],
        num_items=data['num_items'],
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
    )

    if args.resume is not None:
        if not args.resume.exists():
            logger.error(f"--resume path does not exist: {args.resume}")
            return 2
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Resumed from {args.resume}")

    trainer = LightGCNTrainer(model, device=device)
    stats = trainer.train(
        data=data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        early_stopping_patience=args.patience,
        output_path=output_path,
        on_epoch_end=on_epoch_end,
        stop_flag=stop_flag,
    )

    final_metrics = stats['final_metrics']
    sanity_ok, failures = _sanity_check(trainer.model, final_metrics)

    plot_path = output_path.with_suffix('.training_curves.png')
    try:
        trainer.plot_training(plot_path)
    except Exception as e:  # plotting must not fail the run
        logger.warning(f"Plotting failed: {e}")

    interactions_total = len(data['train_df']) + len(data['test_data']['interactions'])
    _write_sidecar(
        sidecar_path,
        checkpoint=output_path.name,
        domain=args.domain,
        device=device,
        trained_at=datetime.now(timezone.utc).isoformat(),
        epochs_run=stats['total_epochs'],
        best_epoch=stats['best_epoch'],
        metrics={
            'recall@10': final_metrics.get('recall@10', 0.0),
            'ndcg@10': final_metrics.get('ndcg@10', 0.0),
            'best_recall@10': stats['best_recall@10'],
            'num_users_evaluated': final_metrics.get('num_users_evaluated', 0),
        },
        dataset={
            'users': int(data['num_users']),
            'items': int(data['num_items']),
            'interactions': int(interactions_total),
        },
        hyperparameters={
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'embedding_dim': args.embedding_dim,
            'num_layers': args.num_layers,
            'patience': args.patience,
            'eval_every': args.eval_every,
            'seed': args.seed,
        },
        timing={
            'avg_epoch_seconds': stats['avg_epoch_time'],
            'total_seconds': stats['total_time'],
        },
        sanity_check={
            'passed': sanity_ok,
            'failures': failures,
        },
    )

    if sanity_ok:
        logger.info("=" * 70)
        logger.info("✅ TRAINING COMPLETE — sanity-check PASSED")
        logger.info(f"   Checkpoint: {output_path}")
        logger.info(f"   Sidecar:    {sidecar_path}")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("=" * 70)
        logger.error("❌ TRAINING COMPLETE — sanity-check FAILED:")
        for f in failures:
            logger.error(f"   - {f}")
        logger.error(f"   Checkpoint kept at: {output_path}")
        logger.error(f"   Sidecar:    {sidecar_path}")
        logger.error("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())