import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from transformers import Wav2Vec2Model
from torchaudio.transforms import Resample
from omegaconf import DictConfig
from typing import Any, ClassVar
from collections.abc import Sequence

from hydra.utils import instantiate
from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates


class Wave2VecCTCModule(pl.LightningModule):
    NUM_BANDS: int = 2
    ELECTRODE_CHANNELS: int = 16

    def __init__(
        self,
        pretrained_model: str,  # "facebook/wav2vec2-large-960h" or "facebook/wav2vec2-base-960h"
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        freeze_feature_extractor: bool = True,
        cache_dir: str = "/home/user/emg2qwerty/cache",  # 预训练模型存储路径
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained Wave2Vec 2.0 model from transformers
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            pretrained_model,
            cache_dir=cache_dir
        )

        if not torch.cuda.is_available():
            self.feature_extractor.to('cpu')
        
        new_conv_layers = [
            # 适当减少 kernel_size 和 stride，确保适用于 2kHz
            torch.nn.Conv1d(1, 512, kernel_size=5, stride=2, bias=False),  # 原: (10, 5)
            torch.nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False), # 原: (3, 2)
            torch.nn.Conv1d(512, 512, kernel_size=2, stride=2, bias=False), # 原: (3, 2)
            torch.nn.Conv1d(512, 512, kernel_size=2, stride=1, bias=False), # 原: (3, 2)
            torch.nn.Conv1d(512, 512, kernel_size=2, stride=1, bias=False), # 原: (2, 2)
            torch.nn.Conv1d(512, 512, kernel_size=2, stride=1, bias=False), # 原: (2, 2)
        ]
        self.feature_extractor.feature_extractor.conv_layers = torch.nn.ModuleList(new_conv_layers)

        # 适配 base 和 large 版本的 hidden_size
        num_features = self.feature_extractor.config.hidden_size  # 自动获取维度

        # 适配 base (12 层) 和 large (24 层) 版本的 Transformer 层数
        total_layers = len(self.feature_extractor.encoder.layers)  # 获取模型总层数
        self.feature_extractor.encoder.layers = self.feature_extractor.encoder.layers[: min(6, total_layers)]  # 限制使用层数

        # 冻结特征提取部分
        # if freeze_feature_extractor:
        #     for param in self.feature_extractor.feature_extractor.parameters():
        #         param.requires_grad = False
        

        # 适配输出维度
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 2, charset().num_classes),  # 自动适配 hidden_size
            nn.LogSoftmax(dim=-1)
        )

        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor, input_lengths) -> torch.Tensor:
        # Reshape tensor
        T, B, _, _ = inputs.shape  # [T, B, 2, 16]
        left_hand = inputs[:, :, 0, :]  # (T, B, 16)
        right_hand = inputs[:, :, 1, :]  # (T, B, 16)
        left_hand = left_hand.mean(dim=-1).permute(1, 0)  # (B, T)
        right_hand = right_hand.mean(dim=-1).permute(1, 0)  # (B, T)

        # attention mask
        attention_mask = torch.arange(T, device=input_lengths.device).expand(B, T) < input_lengths.unsqueeze(1)  # (B, T)
        attention_mask = attention_mask.long()

        # wave2vec extract feature
        left_features = self.feature_extractor(left_hand, attention_mask=attention_mask).last_hidden_state  # (B, T', D)
        right_features = self.feature_extractor(right_hand, attention_mask=attention_mask).last_hidden_state  # (B, T', D)
        features = torch.cat([left_features, right_features], dim=-1)  # (B, T', 2*D)
        output = self.classifier(features).permute(1, 0, 2)  # (T', B, num_chars)
        return output

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch size

        emissions = self.forward(inputs, input_lengths)

        # Compute emission lengths post-processing
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff
        if (emission_lengths < 0).any():
            a = 1

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
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

