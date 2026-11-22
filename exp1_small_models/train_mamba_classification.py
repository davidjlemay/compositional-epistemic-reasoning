#!/usr/bin/env python3
"""
Mamba-based modal logic classification training script.
Uses Mamba (selective state space model) as alternative to Transformer attention.

KEY RESEARCH QUESTION:
Does Mamba (state space model) also preserve product topology?
Theory predicts YES because Mamba is also a continuous map!
"""

import os
import sys
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# Import our modules
from epistemic_logic_tokenizer import EpistemicLogicTokenizer
from epistemic_mamba_model import CompactMamba, HiddenStateCapture, MAMBA_AVAILABLE


class EpistemicDataset(Dataset):
    """Dataset for epistemic logic classification problems."""

    def __init__(self, examples, tokenizer, max_input_length=80, max_target_length=5):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # CLASSIFICATION TASK: Input is smcdel_problem, output is label
        input_ids = self.tokenizer.encode(example['smcdel_problem'],
                                         max_length=self.max_input_length)
        target_ids = self.tokenizer.encode(example['label'],
                                          max_length=self.max_target_length)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }


def collate_fn(batch):
    """Collate function to pad sequences to same length in batch."""
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]

    # Pad inputs
    max_input_len = max(len(ids) for ids in input_ids)
    padded_inputs = torch.zeros(len(batch), max_input_len, dtype=torch.long)
    for i, ids in enumerate(input_ids):
        padded_inputs[i, :len(ids)] = ids

    # Pad targets
    max_target_len = max(len(ids) for ids in target_ids)
    padded_targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
    for i, ids in enumerate(target_ids):
        padded_targets[i, :len(ids)] = ids

    return {
        'input_ids': padded_inputs,
        'target_ids': padded_targets
    }


class GrokkingDetector:
    """Detects grokking phenomenon during training."""

    def __init__(self, threshold=0.05, window=10, min_gap=100):
        """
        Args:
            threshold: accuracy improvement threshold to detect grokking
            window: number of epochs to check for improvement
            min_gap: minimum epochs between grokking events
        """
        self.threshold = threshold
        self.window = window
        self.min_gap = min_gap
        self.grokking_events = []  # List of (epoch, improvement) tuples
        self.val_acc_history = []

    @property
    def grokking_detected(self):
        """Backward compatibility: returns True if any grokking event detected."""
        return len(self.grokking_events) > 0

    @property
    def grokking_epoch(self):
        """Backward compatibility: returns first grokking epoch."""
        return self.grokking_events[0][0] if self.grokking_events else None

    def update(self, val_acc, epoch):
        """Update with new validation accuracy and detect grokking events."""
        self.val_acc_history.append(val_acc)

        if len(self.val_acc_history) >= 2 * self.window:
            # Check for sudden improvement
            recent_avg = np.mean(self.val_acc_history[-self.window:])
            prev_avg = np.mean(self.val_acc_history[-(2*self.window):-self.window])
            improvement = recent_avg - prev_avg

            # Check if this is a grokking event
            if improvement >= self.threshold:
                # Check minimum gap from last event
                if not self.grokking_events or (epoch - self.grokking_events[-1][0]) >= self.min_gap:
                    self.grokking_events.append((epoch, improvement))
                    logging.info(f"\n{'='*60}")
                    logging.info(f"GROKKING EVENT #{len(self.grokking_events)} DETECTED at epoch {epoch}!")
                    logging.info(f"Validation accuracy improved by {improvement:.4f}")
                    logging.info(f"Recent avg: {recent_avg:.4f}, Previous avg: {prev_avg:.4f}")
                    logging.info(f"{'='*60}\n")


class MambaTrainer:
    """Trainer for Mamba-based epistemic logic classification model."""

    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # Wrap model with hidden state capture
        self.model_wrapper = HiddenStateCapture(model)

        # Move to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_wrapper.to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_epochs
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

        # Grokking detector
        self.grokking_detector = GrokkingDetector()

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        self.start_epoch = 0

        # Early stopping
        self.early_stopping_patience = getattr(args, 'early_stopping_patience', 500)
        self.early_stopping_threshold = getattr(args, 'early_stopping_threshold', 1e-4)
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.early_stop_triggered = False

        # Setup output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Hidden states directory
        self.states_dir = self.output_dir / 'captured_states'
        self.states_dir.mkdir(exist_ok=True, parents=True)

        logging.info(f"Trainer initialized on device: {self.device}")
        logging.info(f"Architecture: Mamba (selective state space model)")
        logging.info(f"Task: Classification")
        logging.info(f"Early stopping: patience={self.early_stopping_patience}, threshold={self.early_stopping_threshold}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state."""
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("✓ Model state restored")

        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("✓ Optimizer state restored")

        # Restore scheduler state
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info("✓ Scheduler state restored")

        # Restore metrics
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        logging.info(f"✓ Restored {len(self.train_metrics)} training epochs of metrics")

        # Restore grokking detector state
        if 'grokking_events' in checkpoint:
            self.grokking_detector.grokking_events = checkpoint['grokking_events']
            logging.info(f"✓ Restored {len(self.grokking_detector.grokking_events)} grokking events")

        # Reconstruct validation accuracy history for grokking detector
        self.grokking_detector.val_acc_history = [m['accuracy'] for m in self.val_metrics]
        logging.info(f"✓ Reconstructed validation accuracy history ({len(self.val_metrics)} points)")

        # Restore early stopping state
        if 'best_val_accuracy' in checkpoint:
            self.best_val_acc = checkpoint['best_val_accuracy']
            logging.info(f"✓ Restored best validation accuracy: {self.best_val_acc:.4f}")

        # Set starting epoch
        self.start_epoch = checkpoint['epoch'] + 1
        logging.info(f"✓ Will resume from epoch {self.start_epoch}")

        return checkpoint

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            # Prepare decoder input (shift right, prepend SOS token)
            decoder_input = torch.cat([
                torch.ones(target_ids.size(0), 1, dtype=torch.long, device=self.device),
                target_ids[:, :-1]
            ], dim=1)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model_wrapper(input_ids, decoder_input)

            # Compute loss
            loss = self.criterion(
                output.reshape(-1, self.tokenizer.vocab_size),
                target_ids.reshape(-1)
            )

            # Backward pass
            loss.backward()
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()

            # Calculate accuracy
            predictions = output.argmax(dim=-1)
            mask = (target_ids != 0)  # Ignore padding
            correct = ((predictions == target_ids) & mask).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{total_correct / max(total_tokens, 1):.4f}'
            })

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / max(total_tokens, 1)

        return {'loss': avg_loss, 'accuracy': avg_acc}

    def evaluate(self, dataloader, capture_states=False, split_name='val'):
        """Evaluate on validation/test set."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        if capture_states:
            self.model_wrapper.enable_capture()

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Evaluating {split_name}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)

                # Prepare decoder input
                decoder_input = torch.cat([
                    torch.ones(target_ids.size(0), 1, dtype=torch.long, device=self.device),
                    target_ids[:, :-1]
                ], dim=1)

                # Forward pass
                output = self.model_wrapper(input_ids, decoder_input)

                # Compute loss
                loss = self.criterion(
                    output.reshape(-1, self.tokenizer.vocab_size),
                    target_ids.reshape(-1)
                )

                total_loss += loss.item()

                # Calculate accuracy
                predictions = output.argmax(dim=-1)
                mask = (target_ids != 0)
                correct = ((predictions == target_ids) & mask).sum().item()
                total_correct += correct
                total_tokens += mask.sum().item()

        if capture_states:
            self.model_wrapper.disable_capture()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / max(total_tokens, 1)

        return {'loss': avg_loss, 'accuracy': avg_acc}

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_accuracy': self.best_val_acc,
            'args': vars(self.args),
            'grokking_detected': self.grokking_detector.grokking_detected,
            'grokking_epoch': self.grokking_detector.grokking_epoch,
            'grokking_events': self.grokking_detector.grokking_events,
            'architecture': 'mamba'  # Mark as Mamba for analysis
        }

        # Save latest checkpoint
        latest_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)

    def save_captured_states(self, epoch, split_name):
        """Save captured hidden states."""
        states = self.model_wrapper.get_captured_states()

        # Concatenate all batches
        encoder_states = torch.cat(states['encoder_states'], dim=0) if states['encoder_states'] else None
        decoder_states = torch.cat(states['decoder_states'], dim=0) if states['decoder_states'] else None

        states_dict = {
            'encoder_states': encoder_states,
            'decoder_states': decoder_states,
            'epoch': epoch,
            'architecture': 'mamba'
        }

        save_path = self.states_dir / f'states_epoch_{epoch}_{split_name}.pt'
        torch.save(states_dict, save_path)
        logging.info(f"Saved captured states to {save_path}")

    def train(self, dataset):
        """Main training loop."""
        # Create dataloaders
        train_dataset = EpistemicDataset(
            dataset['train'],
            self.tokenizer,
            self.args.max_input_length,
            self.args.max_target_length
        )
        val_dataset = EpistemicDataset(
            dataset['validation'],
            self.tokenizer,
            self.args.max_input_length,
            self.args.max_target_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn
        )

        logging.info("Starting training...")
        logging.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

        # Determine best validation accuracy from history (if resuming)
        self.best_val_acc = max([m['accuracy'] for m in self.val_metrics], default=0.0)
        if self.best_val_acc > 0:
            logging.info(f"Best validation accuracy from previous training: {self.best_val_acc:.4f}")

        for epoch in range(self.start_epoch, self.args.num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_metrics.append(train_metrics)

            # Evaluate
            if epoch % self.args.eval_interval == 0 or epoch == 0:
                # Determine if we should capture states
                capture_states = self.args.capture_states

                val_metrics = self.evaluate(val_loader, capture_states=capture_states, split_name='val')
                self.val_metrics.append(val_metrics)

                # Save captured states
                if capture_states:
                    self.save_captured_states(epoch, 'val')

                    # Also capture train states
                    _ = self.evaluate(train_loader, capture_states=True, split_name='train')
                    self.save_captured_states(epoch, 'train')

                # Log metrics
                logging.info(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['accuracy']:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}"
                )

                # Update grokking detector
                self.grokking_detector.update(val_metrics['accuracy'], epoch)

                # Check for NaN/Inf - immediate stop
                if np.isnan(val_metrics['loss']) or np.isinf(val_metrics['loss']):
                    logging.error(f"NaN/Inf detected in validation loss at epoch {epoch}!")
                    logging.error("Stopping training to prevent model corruption.")
                    self.early_stop_triggered = True
                    break

                # Early stopping based on validation accuracy plateau
                improvement = val_metrics['accuracy'] - self.best_val_acc
                if improvement > self.early_stopping_threshold:
                    # Significant improvement
                    self.best_val_acc = val_metrics['accuracy']
                    self.epochs_without_improvement = 0
                    is_best = True
                else:
                    # No significant improvement
                    self.epochs_without_improvement += self.args.eval_interval
                    is_best = False

                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)

                # Check early stopping
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logging.info(f"\n{'='*60}")
                    logging.info(f"Early stopping triggered at epoch {epoch}!")
                    logging.info(f"No improvement for {self.epochs_without_improvement} epochs")
                    logging.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
                    logging.info(f"{'='*60}\n")
                    self.early_stop_triggered = True
                    break

            # Update learning rate
            self.scheduler.step()

        if self.early_stop_triggered:
            logging.info("Training stopped early!")
        else:
            logging.info("Training completed!")


def load_symbolic_dataset(data_dir: Path) -> dict:
    """Load symbolic dataset from JSON files."""
    logging.info(f"Loading symbolic dataset from {data_dir}")

    with open(data_dir / "train.json", 'r') as f:
        train_examples = json.load(f)

    with open(data_dir / "val.json", 'r') as f:
        val_examples = json.load(f)

    dataset = {
        'train': train_examples,
        'validation': val_examples,
        'test': val_examples
    }

    logging.info(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples")
    return dataset


def create_symbolic_tokenizer(dataset: dict) -> EpistemicLogicTokenizer:
    """Create tokenizer from symbolic dataset."""
    logging.info("Building symbolic tokenizer...")

    sample_formulas = []
    for split_name, examples in dataset.items():
        for example in examples[:1000]:
            sample_formulas.append(example['smcdel_problem'])
            sample_formulas.append(example['label'])

    tokenizer = EpistemicLogicTokenizer()
    tokenizer.build_vocabulary_from_data(sample_formulas)

    logging.info(f"Built tokenizer with {tokenizer.vocab_size} tokens")
    return tokenizer


def main():
    """Main training function."""
    import argparse
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--d_model', type=int, default=32, help='Model dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=1)
    parser.add_argument('--num_decoder_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--capture_states', type=bool, default=True)
    parser.add_argument('--early_stopping_patience', type=int, default=500,
                        help='Number of epochs without improvement before stopping')
    parser.add_argument('--early_stopping_threshold', type=float, default=1e-4,
                        help='Minimum improvement in validation accuracy to reset patience')

    # Data parameters
    parser.add_argument('--max_input_length', type=int, default=80)
    parser.add_argument('--max_target_length', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)

    # System parameters
    parser.add_argument('--data_dir', type=str, default='data/mindgames_orthogonal_v4_classification',
                        help='Directory containing train.json and val.json')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--run_name', type=str, default='mamba_classification')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--disable_progress_bar', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()

    # Check Mamba availability
    if not MAMBA_AVAILABLE:
        print("ERROR: mamba-ssm package not installed!")
        print("Install with: pip install mamba-ssm")
        print("Requires CUDA and causal-conv1d.")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Set random seed
    if args.seed is not None:
        import random
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to: {args.seed}")

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # Load dataset
    data_dir = Path(args.data_dir)
    dataset = load_symbolic_dataset(data_dir)

    # Create tokenizer
    tokenizer = create_symbolic_tokenizer(dataset)
    tokenizer.save_vocab(Path(args.output_dir) / 'tokenizer.json')

    # Create model
    logger.info("Creating Mamba model...")
    model = CompactMamba(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    logger.info(f"Architecture: Mamba (selective state space model)")
    logger.info(f"Task: Classification (entailment/non-entailment)")
    logger.info(f"Theoretical prediction: Product topology should be preserved (continuous map)")

    # Create trainer
    trainer = MambaTrainer(model, tokenizer, args)

    # Load checkpoint if resuming
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Resuming training from epoch {trainer.start_epoch}")
    else:
        logger.info("Starting fresh training from epoch 0")

    # Start training
    logger.info("Starting Mamba grokking training for classification...")
    trainer.train(dataset)

    # Save final results
    results_file = Path(args.output_dir) / "final_results.json"
    results = {
        'architecture': 'mamba',
        'task': 'classification',
        'final_train_metrics': trainer.train_metrics[-1] if trainer.train_metrics else None,
        'final_val_metrics': trainer.val_metrics[-1] if trainer.val_metrics else None,
        'best_val_accuracy': trainer.best_val_acc,
        'grokking_detected': trainer.grokking_detector.grokking_detected,
        'grokking_epoch': trainer.grokking_detector.grokking_epoch,
        'grokking_events': [(epoch, float(improvement)) for epoch, improvement in trainer.grokking_detector.grokking_events],
        'num_grokking_events': len(trainer.grokking_detector.grokking_events),
        'total_epochs': len(trainer.train_metrics),
        'early_stopping_triggered': trainer.early_stop_triggered,
        'early_stopping_patience': args.early_stopping_patience,
        'model_parameters': total_params,
        'vocab_size': tokenizer.vocab_size
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training completed! Results saved to {results_file}")


if __name__ == "__main__":
    main()
