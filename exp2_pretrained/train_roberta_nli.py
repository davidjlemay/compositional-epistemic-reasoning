#!/usr/bin/env python3
"""
Train RoBERTa cross-encoder for epistemic NLI classification (exp_6).

Task: Given (premise, hypothesis), predict entailment / non_entailment.
Model: roberta-base fine-tuned end-to-end with classification head.

Adapted from exp_5/train_nl_generation.py.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

from nli_dataset import NLIDataset, collate_fn, load_nli_dataset, load_test_set
from roberta_nli_model import RobertaNLI, HiddenStateCapture


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)


class GrokkingDetector:
    """Detects sudden improvements in validation accuracy (grokking)."""

    def __init__(self, threshold=0.05, window=5, min_gap=20):
        self.threshold = threshold
        self.window = window
        self.min_gap = min_gap
        self.grokking_events = []
        self.val_acc_history = []

    @property
    def grokking_detected(self):
        return len(self.grokking_events) > 0

    @property
    def grokking_epoch(self):
        return self.grokking_events[0][0] if self.grokking_events else None

    def update(self, val_acc, epoch):
        self.val_acc_history.append(val_acc)
        if len(self.val_acc_history) >= 2 * self.window:
            recent = np.mean(self.val_acc_history[-self.window:])
            prev = np.mean(self.val_acc_history[-(2 * self.window):-self.window])
            improvement = recent - prev
            if improvement >= self.threshold:
                if not self.grokking_events or (epoch - self.grokking_events[-1][0]) >= self.min_gap:
                    self.grokking_events.append((epoch, improvement))
                    logging.info(f"\n{'='*60}")
                    logging.info(f"GROKKING EVENT #{len(self.grokking_events)} at epoch {epoch}!")
                    logging.info(f"Val acc improved by {improvement:.4f} "
                                 f"(recent={recent:.4f}, prev={prev:.4f})")
                    logging.info(f"{'='*60}\n")


class NLITrainer:

    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir = self.output_dir / 'captured_states'
        self.states_dir.mkdir(exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.grokking_detector = GrokkingDetector()
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def setup_optimizer(self, num_training_steps: int):
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(params, lr=self.args.learning_rate)
        warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(input_ids, attention_mask)
            loss = self.criterion(out['logits'], labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item() * len(labels)
            preds = out['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        return {'loss': total_loss / total, 'accuracy': correct / total}

    @torch.no_grad()
    def evaluate(self, loader, split='val', capture_states: bool = False):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        by_complexity: dict = {}  # (n_agents, depth) -> (correct, total)

        capture = HiddenStateCapture(self.model) if capture_states else None
        if capture:
            capture.start()

        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            out = self.model(input_ids, attention_mask)
            loss = self.criterion(out['logits'], labels)
            preds = out['logits'].argmax(dim=-1)

            total_loss += loss.item() * len(labels)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            # Track accuracy by (n_agents, hypothesis_depth)
            for i in range(len(labels)):
                key = (batch['n_agents'][i].item(), batch['hypothesis_depth'][i].item())
                if key not in by_complexity:
                    by_complexity[key] = [0, 0]
                by_complexity[key][0] += int(preds[i] == labels[i])
                by_complexity[key][1] += 1

        if capture:
            capture.stop()
            states = capture.get_states()
        else:
            states = {}

        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        complexity_acc = {
            str(k): v[0] / v[1] for k, v in by_complexity.items() if v[1] > 0
        }

        return {
            'loss': total_loss / total,
            'accuracy': correct / total,
            'f1': f1,
            'complexity_accuracy': complexity_acc,
            'hidden_states': states,
        }

    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = 'latest'):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'grokking_detected': self.grokking_detector.grokking_detected,
            'grokking_epoch': self.grokking_detector.grokking_epoch,
            'args': vars(self.args),
        }
        path = self.output_dir / f'checkpoint_{tag}.pt'
        torch.save(ckpt, path)

    def save_hidden_states(self, states: dict, epoch: int, split: str):
        if not states:
            return
        path = self.states_dir / f'states_epoch_{epoch}_{split}.pt'
        torch.save(states, path)
        logging.info(f"Saved hidden states ({len(states)} layers) -> {path}")

    def train(self, train_loader, val_loader, test_loader=None):
        num_training_steps = self.args.num_epochs * len(train_loader)
        self.setup_optimizer(num_training_steps)

        logging.info(f"Training for {self.args.num_epochs} epochs on {self.device}")
        logging.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        for epoch in range(self.args.num_epochs):
            train_m = self.train_epoch(train_loader)
            self.train_metrics.append({'epoch': epoch, **train_m})

            if epoch % self.args.eval_interval == 0 or epoch == self.args.num_epochs - 1:
                # Capture states near grokking events or at intervals
                capture = (
                    self.grokking_detector.grokking_detected
                    or epoch % self.args.state_capture_interval == 0
                )
                val_m = self.evaluate(val_loader, split='val', capture_states=capture)
                self.val_metrics.append({'epoch': epoch, **{k: v for k, v in val_m.items()
                                                            if k != 'hidden_states'}})

                self.grokking_detector.update(val_m['accuracy'], epoch)

                logging.info(
                    f"Epoch {epoch:5d} | "
                    f"train_loss={train_m['loss']:.4f} train_acc={train_m['accuracy']:.4f} | "
                    f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f} "
                    f"val_f1={val_m['f1']:.4f}"
                )

                # Save hidden states (val)
                if capture and val_m['hidden_states']:
                    self.save_hidden_states(val_m['hidden_states'], epoch, 'val')

                # Also capture train states (needed for linear probing)
                if capture:
                    train_cap_m = self.evaluate(train_loader, split='train', capture_states=True)
                    if train_cap_m['hidden_states']:
                        self.save_hidden_states(train_cap_m['hidden_states'], epoch, 'train')

                # Checkpoint
                self.save_checkpoint(epoch, val_m, tag='latest')
                if val_m['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_m['accuracy']
                    self.save_checkpoint(epoch, val_m, tag='best')
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if (self.args.early_stopping_patience > 0
                        and self.epochs_without_improvement >= self.args.early_stopping_patience):
                    logging.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.args.early_stopping_patience} eval intervals)"
                    )
                    break

        # Final evaluation on test (held-out agents)
        if test_loader is not None:
            logging.info("\nEvaluating on held-out test set (unseen agents)...")
            test_m = self.evaluate(test_loader, split='test', capture_states=True)
            logging.info(
                f"Test acc={test_m['accuracy']:.4f} | f1={test_m['f1']:.4f}"
            )
            if test_m['hidden_states']:
                self.save_hidden_states(test_m['hidden_states'], -1, 'test')

            # Save test results
            test_results = {k: v for k, v in test_m.items() if k != 'hidden_states'}
            with open(self.output_dir / 'test_results.json', 'w') as f:
                json.dump(test_results, f, indent=2)

        # Save full metrics
        with open(self.output_dir / 'train_metrics.json', 'w') as f:
            json.dump(self.train_metrics, f, indent=2)
        with open(self.output_dir / 'val_metrics.json', 'w') as f:
            json.dump(self.val_metrics, f, indent=2)

        logging.info(f"Training complete. Best val acc: {self.best_val_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train RoBERTa NLI for epistemic reasoning')

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='data/mindgames_orthogonal_v4_agent_aware')
    parser.add_argument('--test_dir', type=str,
                        default='data/mindgames_orthogonal_test_held_out')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max cross-encoder input length')

    # Model
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        choices=['roberta-base', 'roberta-large'])
    parser.add_argument('--freeze_layers', type=int, default=0,
                        help='Number of bottom transformer layers to freeze (0=fine-tune all)')
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Fraction of training steps for LR warmup')

    # Evaluation and checkpointing
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--state_capture_interval', type=int, default=5,
                        help='Capture hidden states every N eval intervals')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Stop after N eval intervals without improvement (0=disabled)')

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    logging.info("Loading datasets...")
    dataset = load_nli_dataset(Path(args.data_dir))

    test_examples = None
    if Path(args.test_dir).exists():
        test_examples = load_test_set(Path(args.test_dir))

    # Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)

    # Datasets
    train_ds = NLIDataset(dataset['train'], tokenizer, max_length=args.max_length)
    val_ds = NLIDataset(dataset['validation'], tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    test_loader = None
    if test_examples:
        test_ds = NLIDataset(test_examples, tokenizer, max_length=args.max_length,
                             use_natural_language=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=2)

    # Model
    model = RobertaNLI(
        model_name=args.model_name,
        num_labels=2,
        dropout=args.dropout,
        freeze_layers=args.freeze_layers,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable:,} / {total:,}")

    # Train
    trainer = NLITrainer(model, tokenizer, args)
    trainer.train(train_loader, val_loader, test_loader)


if __name__ == '__main__':
    main()
