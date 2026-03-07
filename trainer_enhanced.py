"""
trainer_enhanced.py
===================
TinyTransformerTrainer — trainer for EnhancedOcclusionAwareTransformer with
the full three-term loss:

    L = L_class  +  0.1 * L_gate_align  −  0.01 * L_gate_div

where
  L_class      is cross-entropy over the 3 driver states.
  L_gate_align teaches eye/mouth gate MLPs to output low values when
               the occlusion estimator reports high occlusion probability
               (gate target = 0.3 + 0.7 * (1 − p_occ)).
  L_gate_div   is negative gate variance, encouraging diverse gate outputs
               across the four regions.

Usage
-----
    from trainer_enhanced import TinyTransformerTrainer
    trainer = TinyTransformerTrainer(model, device='cuda')
    trainer.train_epoch(train_loader, epoch=0)
    trainer.evaluate(val_loader)
"""

import traceback
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import move_batch_to_device


class TinyTransformerTrainer:
    """
    Trainer for EnhancedOcclusionAwareTransformer.

    Parameters
    ----------
    model        : the gated transformer instance.
    device       : 'cpu' or 'cuda'.
    learning_rate: initial AdamW learning rate.
    """

    def __init__(self, model, device: str = 'cpu',
                 learning_rate: float = 3e-5):
        self.model  = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate,
            weight_decay=1e-4, betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=1e-6,
        )
        self.classification_criterion = nn.CrossEntropyLoss()

        self.loss_weights = {
            'classification': 1.0,
            'gate_occ':       0.1,
            'gate_div':       0.01,
        }

        self.history: Dict = {
            'epoch_losses':          [],
            'classification_losses': [],
            'accuracies':            [],
            'learning_rates':        [],
        }

        print(f' TinyTransformerTrainer initialized on {device}')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        return move_batch_to_device(batch, self.device)

    def _compute_losses(self, outputs: Dict, batch: Dict) -> Dict:
        labels = batch['label'].view(-1)
        if labels.dtype != torch.long:
            labels = labels.long()

        class_loss = self.classification_criterion(
            outputs['class_logits'], labels)

        # Gate factors [B, 4]: [face, left_eye, right_eye, mouth]
        gf = outputs['gate_factors']

        # Gate alignment loss
        occ = batch.get('occlusion_targets', None)
        if isinstance(occ, torch.Tensor):
            eye_prob   = occ[:, 0].clamp(0, 1)
            mouth_prob = occ[:, 1].clamp(0, 1)
            eye_target   = 0.3 + 0.7 * (1.0 - eye_prob)
            mouth_target = 0.3 + 0.7 * (1.0 - mouth_prob)
            gate_occ_reg = (F.mse_loss(gf[:, 1], eye_target) +
                            F.mse_loss(gf[:, 2], eye_target) +
                            F.mse_loss(gf[:, 3], mouth_target))
        else:
            gate_occ_reg = torch.tensor(0.0, device=self.device)

        # Diversity regularisation (maximise gate variance → negative)
        gate_div_reg = -torch.var(gf, dim=1).mean()

        total_loss = (self.loss_weights['classification'] * class_loss
                      + self.loss_weights['gate_occ'] * gate_occ_reg
                      + self.loss_weights['gate_div'] * gate_div_reg)

        return {
            'total_loss':          total_loss,
            'classification_loss': class_loss,
            'gate_occ_reg':        gate_occ_reg,
            'gate_div_reg':        gate_div_reg,
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def train_epoch(self, train_loader, epoch: int) -> Dict:
        self.model.train()
        epoch_losses, epoch_class_losses = [], []
        correct, total = 0, 0

        print(f'\n EPOCH {epoch + 1} TRAINING')
        for batch_idx, batch in enumerate(train_loader):
            try:
                batch = self._move_batch_to_device(batch)
                self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(batch['features'], batch['occlusion_info'])
                losses  = self._compute_losses(outputs, batch)
                total_loss = losses['total_loss']

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f' Invalid loss at batch {batch_idx}, skipping...')
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_losses.append(total_loss.item())
                epoch_class_losses.append(losses['classification_loss'].item())

                pred   = outputs['predicted_class'].view(-1)
                labels = batch['label'].view(-1)
                correct += (pred == labels).sum().item()
                total   += labels.size(0)

                if ((batch_idx + 1) % max(1, len(train_loader) // 3) == 0
                        or batch_idx == len(train_loader) - 1):
                    print(f'  Batch {batch_idx + 1}/{len(train_loader)} '
                          f'Loss={total_loss.item():.4f} '
                          f'Class={losses["classification_loss"].item():.4f}')
            except Exception as e:
                print(f' Error in batch {batch_idx}: {e}')
                traceback.print_exc()

        self.scheduler.step()

        if epoch_losses:
            avg_loss  = float(np.mean(epoch_losses))
            avg_class = float(np.mean(epoch_class_losses))
            accuracy  = correct / max(total, 1) * 100.0
            lr        = self.optimizer.param_groups[0]['lr']

            self.history['epoch_losses'].append(avg_loss)
            self.history['classification_losses'].append(avg_class)
            self.history['accuracies'].append(accuracy)
            self.history['learning_rates'].append(lr)

            print(f'\n Epoch {epoch + 1}: Loss={avg_loss:.4f} '
                  f'Class={avg_class:.4f} Acc={accuracy:.1f}% LR={lr:.2e}')
            return {'avg_loss': avg_loss, 'accuracy': accuracy,
                    'class_loss': avg_class}

        return {'avg_loss': float('inf'), 'accuracy': 0.0,
                'class_loss': float('inf')}

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, loader, name: str = 'VAL') -> Dict:
        self.model.eval()
        print(f'\n DETAILED EVALUATION [{name}]')
        all_preds, all_labels = [], []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch   = self._move_batch_to_device(batch)
                outputs = self.model(batch['features'], batch['occlusion_info'],
                                     return_attention=True)

                pred   = outputs['predicted_class'].view(-1).detach().cpu().numpy()
                labels = batch['label'].view(-1).detach().cpu().numpy()
                all_preds.extend(pred.tolist())
                all_labels.extend(labels.tolist())

                probs = outputs['class_probs'].detach().cpu().numpy()
                gates = outputs['gate_factors'].detach().cpu().numpy()
                j = 0
                if len(pred) > 0:
                    cname = (batch['class_name'][j]
                             if isinstance(batch.get('class_name'), list)
                             else str(batch.get('class_name')))
                    conf = float(probs[j][int(pred[j])])
                    print(f'  Sample {i+1}-1 ({cname}): '
                          f'pred={int(pred[j])} conf={conf:.3f} '
                          f'gates={np.round(gates[j], 2)}')

        acc = (float(np.mean(np.array(all_preds) == np.array(all_labels)) * 100)
               if all_labels else 0.0)
        print(f' OVERALL ACCURACY: {acc:.1f}%')
        return {'accuracy': acc, 'predictions': all_preds, 'labels': all_labels}

    # ── Full training loop ────────────────────────────────────────────────────

    def train(self, train_loader, val_loader, num_epochs: int = 20,
              patience: int = 20, save_path: str = 'best_transformer.pth') -> Dict:
        """Train for up to *num_epochs* with early stopping on val accuracy."""
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.train_epoch(train_loader, epoch)
            val_result = self.evaluate(val_loader, name='VAL')
            val_acc = val_result['accuracy']

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), save_path)
                print(f' New best: {best_val_acc:.1f}%  → saved to {save_path}')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f' Early stopping after {epoch + 1} epochs.')
                break

        print(f' Training complete. Best val accuracy: {best_val_acc:.1f}%')
        return {'best_val_accuracy': best_val_acc, 'history': self.history}
