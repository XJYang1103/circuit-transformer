"""Training helpers kept separate from the circuit search algorithm."""
import os
import copy
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from .utils import MPDataset

from .pytorch_transformer import CustomSchedule, KerasAdam, masked_accuracy, masked_loss


class GeneratorDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()


class ReplicaObjective(torch.nn.Module):
    """Reduce valid tokens *within* a replica, as the original Keras loss does."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels):
        logits = self.model(inputs)
        return masked_loss(labels, logits), masked_accuracy(labels, logits)


class ReplicaDataParallel(torch.nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        features, labels = inputs
        if labels.shape[0] == 0 or not device_ids:
            raise ValueError('Cannot scatter an empty batch or use an empty device list.')
        device_ids = device_ids[:labels.shape[0]]
        quotient, remainder = divmod(labels.shape[0], len(device_ids))
        scattered, offset = [], 0
        # Match tf.distribute's balanced split, rather than torch.chunk's
        # potentially smaller final chunk / fewer chunks for partial batches.
        for index, device_id in enumerate(device_ids):
            count = quotient + (index < remainder)
            end = offset + count
            device = device_id if isinstance(device_id, torch.device) else torch.device('cuda', device_id)
            scattered.append(({key: value[offset:end].to(device) for key, value in features.items()},
                                labels[offset:end].to(device)))
            offset = end
        return tuple(scattered), tuple(dict(kwargs or {}) for _ in device_ids)


def train_circuit(owner, train_data_dir, ckpt_save_path=None, validation_split=0.1,
                    epochs=10, initial_epoch=0, batch_size=128, profile=False,
                    distributed=False, latest_ckpt_only=False, log_dir='tensorboard',
                    excluded_files=None, num_workers=8):
    if not 0 <= validation_split < 1:
        raise ValueError('validation_split must be in [0, 1)')
    if num_workers < 0:
        raise ValueError('num_workers must be nonnegative')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not 0 <= initial_epoch < epochs:
        raise ValueError('Expected 0 <= initial_epoch < epochs.')
    files = sorted(os.listdir(train_data_dir))
    np.random.seed(0)
    np.random.shuffle(files)
    excluded = set(excluded_files or [])
    files = [os.path.join(train_data_dir, name) for name in files if name not in excluded]
    if any(not os.path.isfile(path) for path in files):
        raise ValueError('Training data must contain files, not subdirectories.')
    split = int(len(files) * (1 - validation_split))
    if split == 0:
        raise ValueError('The training split contains no examples.')
    if num_workers > 1 and split < 2:
        raise ValueError('The asynchronous loader needs at least two training examples (it yields N-1 per epoch).')
    if ckpt_save_path is not None:
        os.makedirs(ckpt_save_path, exist_ok=True)
    model = owner._transformer
    if distributed:
        # The original MirroredStrategy branch creates a fresh model, restoring
        # only ckpt_path. Keep that initialization behavior.
        model = owner._get_torch_transformer().to(owner.device)
        if owner.ckpt_path is not None:
            from .checkpoints import load_checkpoint
            load_checkpoint(model, owner.ckpt_path)
    old_cache, old_training = model.return_cache, model.training
    objective = ReplicaObjective(model)
    if distributed and owner.device.type == 'cuda' and torch.cuda.device_count() > 1:
        if owner.device.index not in (None, 0):
            raise ValueError('distributed=True requires the primary model on cuda:0')
        objective = ReplicaDataParallel(objective)
    optimizer = KerasAdam(model.parameters())
    schedule = CustomSchedule(owner.embedding_width)
    # Keras compile automatically enables dynamic loss scaling for mixed_float16.
    scaler = torch.amp.GradScaler(owner.device.type, init_scale=32768,
                                    growth_interval=2000, enabled=owner.mixed_precision)
    iterations, batches_seen = 0, 0
    save_freq = split * (epochs - initial_epoch) // batch_size
    profiler = (torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir))
                if profile else nullcontext())
    history = []
    encoder = copy.copy(owner)
    encoder._transformer = None
    encoder._transformer_inference = None
    dataset = MPDataset(files, encoder.load_and_encode_formatted,
                        num_processes=num_workers, validation_split=validation_split)
    loaders = [DataLoader(GeneratorDataset(generator), batch_size=batch_size, num_workers=0)
                for generator in (dataset.train_generator, dataset.validation_generator)]
    try:
        model.return_cache = False
        with profiler as trace:
            for epoch in range(initial_epoch, epochs):
                metrics = {}
                for phase, loader in zip(('train', 'validation'), loaders):
                    model.train(phase == 'train')
                    total_loss = total_accuracy = examples = metric_batches = 0
                    with torch.set_grad_enabled(phase == 'train'):
                        for inputs, labels in loader:
                            labels = labels.to(owner.device)
                            loss, accuracy = objective(inputs, labels)
                            # Keras averages the scalar per-replica masked losses,
                            # rather than pooling valid-token counts across replicas.
                            loss, accuracy = loss.mean(), accuracy.mean()
                            if phase == 'train':
                                optimizer.zero_grad(set_to_none=True)
                                optimizer.param_groups[0]['lr'] = float(schedule(iterations))
                                scaler.scale(loss).backward()
                                old_scale = scaler.get_scale()
                                scaler.step(optimizer)
                                scaler.update()
                                if scaler.get_scale() < old_scale:
                                    optimizer.skipped_step()
                                iterations += 1
                                batches_seen += 1
                                if trace is not None:
                                    trace.step()
                                if ckpt_save_path is not None and latest_ckpt_only and (save_freq == 0 or batches_seen % save_freq == 0):
                                    from .checkpoints import save_checkpoint
                                    save_checkpoint(model, os.path.join(ckpt_save_path, f'model-{epoch + 1:04d}.pt'))
                            count = labels.shape[0]
                            total_loss += float(loss.detach()) * count
                            # Keras MeanMetricWrapper sees a scalar accuracy,
                            # so each batch contributes one metric observation.
                            total_accuracy += float(accuracy.detach())
                            metric_batches += 1
                            examples += count
                    if examples:
                        metrics[phase + '_loss'] = total_loss / examples
                        metrics[phase + '_accuracy'] = total_accuracy / metric_batches
                history.append(metrics)
                print(f'Epoch {epoch + 1}/{epochs}: {metrics}')
                if ckpt_save_path is not None and not latest_ckpt_only:
                    from .checkpoints import save_checkpoint
                    save_checkpoint(model, os.path.join(ckpt_save_path, f'model-{epoch + 1:04d}.pt'))
    finally:
        dataset.close()
        model.return_cache = old_cache
        model.train(old_training)
    owner.training_history = history
    if not distributed:
        owner._weights_ready = True
    return None  # The original public train() method returns None.
