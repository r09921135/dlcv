import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, load_state_dict, rename_parallel_state_dict, autocast, use_fp16
import model.model as module_arch
# import pathlib
import torch.nn.functional as F
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config

        # add_extra_info will return info about individual experts. This is crucial for individual loss. If this is false, we can only get a final mean logits.
        self.add_extra_info = config._config.get('add_extra_info', False)
        print("self.add_extra_info",self.add_extra_info)

        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        if use_fp16:
            self.logger.warn("FP16 is enabled. This option should be used with caution unless you make sure it's working and we do not provide guarantee.")
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.hard_rot_criterion_expert1 = torch.nn.CrossEntropyLoss()
        self.soft_rot_criterion_expert1_to_expert2 = torch.nn.KLDivLoss(reduction='batchmean')
        self.hard_rot_criterion_expert2 = torch.nn.CrossEntropyLoss()
        self.soft_rot_criterion_expert2_to_expert3 = torch.nn.KLDivLoss(reduction='batchmean')
        self.hard_rot_criterion_expert3 = torch.nn.CrossEntropyLoss()
        self.temperature = 5
        # self.hard_rot_criterion = torch.nn.CrossEntropyLoss()

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        breakpoint()
        self.model.train()
        self.real_model._hook_before_iter()
        self.train_metrics.reset()
        # print(self.epochs)
        # print(epoch)
        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)

        for batch_idx, data in enumerate(self.data_loader):
            # val_log = self._valid_epoch(epoch)
            (data, rot_data, rot_label), target = data
            data, target = data.to(self.device), target.to(self.device)
            rot_data, rot_label = rot_data.to(self.device), rot_label.to(self.device, dtype=torch.long)
            rot_label = [x + target[i]*4 for i, x in enumerate(rot_label)]
            rot_label = torch.stack(rot_label)
            # print(target)
            # print(rot_label)
            self.optimizer.zero_grad()

            with autocast():
                if self.real_model.requires_target:
                    output = self.model(data, target=target) 

                    output, loss = output   
                else:
                    extra_info = {}
                    output, rot_logits = self.model(data, epoch=epoch, num_epochs=self.epochs, rot_x=rot_data)

                    if self.add_extra_info:
                        if isinstance(output, dict):
                            logits = output["logits"]
                            extra_info.update({
                                "logits": logits.transpose(0, 1)
                            })
                        else:
                            extra_info.update({
                                "logits": self.real_model.backbone.logits
                            })

                    if isinstance(output, dict):
                        output = output["output"]

                    # weight between soft label and hard label
                    alpha_weighting = 1.0 - (epoch/self.epochs)**2
                    if self.add_extra_info:
                        loss = self.criterion(output_logits=output, target=target, extra_info=extra_info, epoch=epoch, num_epochs=self.epochs)

                        # Self-Supervised Knowldge Distillation 
                        loss_rot_expert1 = self.hard_rot_criterion_expert1(rot_logits[0], rot_label)
                        loss_rot_expert2 = self.soft_rot_criterion_expert1_to_expert2(F.log_softmax(rot_logits[1]/self.temperature, dim=1), F.softmax(rot_logits[0]/self.temperature, dim=1)) \
                            * (self.temperature * self.temperature) * (1 - alpha_weighting) + alpha_weighting * self.hard_rot_criterion_expert2(rot_logits[1], rot_label)
                        loss_rot_expert3 = self.soft_rot_criterion_expert2_to_expert3(F.log_softmax(rot_logits[2]/self.temperature, dim=1), F.softmax(rot_logits[1]/self.temperature, dim=1)) \
                            * (self.temperature * self.temperature) * (1 - alpha_weighting) + alpha_weighting * self.hard_rot_criterion_expert3(rot_logits[2], rot_label)
                        
                        loss = loss + loss_rot_expert1 + loss_rot_expert2 + loss_rot_expert3
                    else:
                        loss = self.criterion(output_logits=output, target=target, extra_info=extra_info, epoch=epoch, num_epochs=self.epochs)
                        
                        # Self-Supervised Knowldge Distillation 
                        loss_rot_expert1 = self.hard_rot_criterion_expert1(rot_logits[0], rot_label)
                        loss_rot_expert2 = self.soft_rot_criterion_expert1_to_expert2(F.log_softmax(rot_logits[1]/self.temperature, dim=1), F.softmax(rot_logits[0]/self.temperature, dim=1)) \
                            * (self.temperature * self.temperature) * (1 - alpha_weighting) + alpha_weighting * self.hard_rot_criterion_expert2(rot_logits[1], rot_label)
                        loss_rot_expert3 = self.soft_rot_criterion_expert2_to_expert3(F.log_softmax(rot_logits[2]/self.temperature, dim=1), F.softmax(rot_logits[1]/self.temperature, dim=1)) \
                            * (self.temperature * self.temperature) * (1 - alpha_weighting) + alpha_weighting * self.hard_rot_criterion_expert3(rot_logits[2], rot_label)
                        
                        loss = loss + loss_rot_expert1 + loss_rot_expert2 + loss_rot_expert3
            
            if not use_fp16:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target, return_length=True))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} max group LR: {:.7f} min group LR: {:.7f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    max([param_group['lr'] for param_group in self.optimizer.param_groups]),
                    min([param_group['lr'] for param_group in self.optimizer.param_groups])))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
           
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                if isinstance(output, dict):
                    output = output["output"]
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, return_length=True))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
