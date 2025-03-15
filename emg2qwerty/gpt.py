from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform


import torch.nn.functional as F
from transformers import GPT2Model
from einops import rearrange
from emg2qwerty.embed import DataEmbedding


class GPTFeatureExtractor(nn.Module):
    def __init__(self, num_features, d_model, gpt_layers=3):
        super().__init__()
        self.d_model = d_model
        
        self.gpt2 = GPT2Model.from_pretrained("gpt2", output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:gpt_layers]  # Only use first few layers for efficiency
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            elif "h.2" in name or "h.1" in name:  # 解冻最后2个 Transformer 块 (index 2)
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.embedding = DataEmbedding(c_in=num_features, d_model=d_model, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x shape: (T, N, num_features)
        T, N, C = x.shape
        x = x.permute(1, 0, 2)

        x = self.embedding(x, None)  # Map to d_model
        outputs = self.gpt2(inputs_embeds=x).last_hidden_state  # Extract features
        outputs = F.gelu(outputs)  # Apply activation
        outputs = self.norm(outputs)
        outputs = rearrange(outputs, 'n t d -> t n d')  # Convert back to (T', N, d_model)
        
        return outputs

class GPTCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: Dict,
        lr_scheduler: Dict,
        decoder: Dict,
        d_model: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        
        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Dropout(0.1),
            nn.Flatten(start_dim=2),
            GPTFeatureExtractor(
                num_features=num_features, 
                d_model=d_model
            ),
            nn.Linear(d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {f"{phase}_metrics": metrics.clone(prefix=f"{phase}/") for phase in ["train", "val", "test"]}
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )









