import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from src.sslsteal.sslsteal_utils import save_config_file, accuracy, save_checkpoint
from loss import soft_cross_entropy, wasserstein_loss, soft_nn_loss, pairwise_euclid_distance, SupConLoss, barlow_loss
import time
from collections import defaultdict

torch.manual_seed(0)

class SimCLR(object):

    def __init__(self, stealing=False, victim_model=None, victim_head = None, entropy_model = None, watermark_mlp = None, logdir='', loss=None, timing_metrics=None, *args,
                 **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.timing_metrics = timing_metrics or defaultdict(list)  # Store timing metrics
        # self.log_dir = 'runs/' + logdir
        if watermark_mlp is not None:
            self.watermark_mlp = watermark_mlp.to(self.args.device)
        # if stealing:
        #     if self.args.defence == "True":
        #         self.log_dir2 = f"./checkpoint/{os.getenv('USER')}/SimCLR/{self.args.epochs}{self.args.archstolen}{self.args.losstype}DEFENCE/"  # save logs here. # edited by Dev
        #     else:
        #         if self.args.victimhead == "True":
        #             self.log_dir2 = f"./checkpoint/{os.getenv('USER')}/SimCLR/{self.args.epochs}{self.args.archstolen}{self.args.losstype}STEALWVICH/" # save logs here. # edited by Dev
        #         else:
        #             self.log_dir2 = f"./checkpoint/{os.getenv('USER')}/SimCLR/{self.args.epochs}{self.args.archstolen}{self.args.losstype}STEAL/"  # save logs here. # edited by Dev
        # else:
        #     self.log_dir2 = f"./checkpoint/{os.getenv('USER')}/SimCLR/{self.args.epochs}{self.args.arch}{self.args.losstype}TRAIN/" # edited by Dev
        self.log_dir2 = logdir
        self.stealing = stealing
        self.loss = loss
        logname = 'training.log'
        if self.stealing:
            logname = f'training{self.args.datasetsteal}{self.args.num_queries}.log'
        if os.path.exists(os.path.join(self.log_dir2, logname)):
            if self.args.clear == "True":
                os.remove(os.path.join(self.log_dir2, logname))
        else:
            try:
                try:
                    os.mkdir(self.log_dir2, exist_ok=False)
                except:
                    try:
                        from pathlib import Path                                    # added by Dev
                        output_file = Path(os.path.join(self.log_dir2, logname))    # added by Dev
                        output_file.parent.mkdir(exist_ok=True, parents=True)       # added by Dev
                    except:
                        os.mkdir(self.log_dir2)
            except:
                print(f"Error creating directory at {self.log_dir2}")
        logging.basicConfig(
            filename=os.path.join(self.log_dir2, logname),
            level=logging.DEBUG)
        if self.stealing:
            self.victim_model = victim_model.to(self.args.device)
            if self.args.defence == "True":
                self.victim_head = victim_head.to(self.args.device)
        if self.loss in ["infonce", "infonce2"]:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        elif self.loss == "softce":
            self.criterion = soft_cross_entropy
        elif self.loss == "wasserstein":
            self.criterion = wasserstein_loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss().to(self.args.device)
        elif self.loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss == "softnn":
            self.criterion = soft_nn_loss
            self.tempsn = self.args.temperaturesn
        elif self.loss == "supcon":
            self.criterion = SupConLoss(temperature=self.args.temperature)
        elif self.loss == "symmetrized":
            self.criterion = nn.CosineSimilarity(dim=1)
        elif self.loss == "barlow": # method from barlow twins
            self.criterion = barlow_loss
        else:
            raise RuntimeError(f"Loss function {self.loss} not supported.")
        self.criterion2 = nn.CosineSimilarity(dim=1) # for the defence

    def get_model_flops(self, model, input_shape):
        """Estimate FLOPs for a model (simplified estimation)"""
        def flop_count(module, input, output):
            if isinstance(module, nn.Linear):
                return module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                return (module.in_channels * module.out_channels * 
                       module.kernel_size[0] * module.kernel_size[1] * 
                       output.shape[-2] * output.shape[-1])
            elif isinstance(module, nn.BatchNorm2d):
                return output.numel()
            else:
                return 0
        
        model.eval()
        total_flops = 0
        
        def hook_fn(module, input, output):
            nonlocal total_flops
            total_flops += flop_count(module, input, output)
        
        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(hook_fn))
        
        with torch.no_grad():
            dummy_input = torch.randn(input_shape).cuda()
            model(dummy_input)
        
        for hook in hooks:
            hook.remove()
        
        return total_flops


    def info_nce_loss(self, features):
        n = int(features.size()[0] / self.args.batch_size)
        labels = torch.cat(
            [torch.arange(self.args.batch_size) for i in range(n)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
            self.args.device)
        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, watermark_loader=None):
        print("_" * 100)
        print(f"Training on {len(train_loader)} batches for {self.args.epochs} epochs")
        print("_" * 100)
        
        scaler = GradScaler("cuda", enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.log_dir2, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {torch.cuda.is_available()}.")
        logging.info(f"Args: {self.args}")

        for epoch_counter in range(self.args.epochs):
            total_queries = 0
            for images, truelabels in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast("cuda", enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    if self.loss == "softnn":
                        loss = self.criterion(self.args, features,
                                              pairwise_euclid_distance, self.tempsn)
                    elif self.loss == "supcon":
                        labels = truelabels
                        bsz = labels.shape[0]
                        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                        features = torch.cat(
                            [f1.unsqueeze(1), f2.unsqueeze(1)],
                            dim=1)
                        loss = self.criterion(features, labels)
                    else:
                        loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                n_iter += 1
            if watermark_loader is not None:
                watermark_accuracy = 0
                for counter, (images, _) in enumerate(tqdm(watermark_loader)):
                    images = torch.cat(images, dim=0)

                    images = images.to(self.args.device)

                    with autocast("cuda", enabled=self.args.fp16_precision):
                        x = self.model.backbone.conv1(images)
                        x = self.model.backbone.bn1(x)
                        x = self.model.backbone.relu(x)
                        x = self.model.backbone.maxpool(x)
                        x = self.model.backbone.layer1(x)
                        x = self.model.backbone.layer2(x)
                        x = self.model.backbone.layer3(x)
                        x = self.model.backbone.layer4(x)
                        x = self.model.backbone.avgpool(x)
                        features = torch.flatten(x, 1)
                        logits = self.watermark_mlp(features)
                        labels = torch.cat([torch.zeros(self.args.batch_size),
                                            torch.ones(self.args.batch_size)],
                                           dim=0).long().to(self.args.device)
                        loss = self.criterion(logits, labels)
                        w_top1 = accuracy(logits, labels, topk=(1,))
                        watermark_accuracy += w_top1[0]

                    self.optimizer.zero_grad()

                    scaler.scale(loss).backward()

                    scaler.step(self.optimizer)
                    scaler.update()
                watermark_accuracy /= (counter + 1)
                logging.debug(f"Epoch: {epoch_counter}\t Watermark Acc: {watermark_accuracy}")

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\t")

        logging.info("Training has finished.")
        # save model checkpoints
        if watermark_loader is None:
            checkpoint_name = f'{self.args.dataset}_checkpoint_{self.args.epochs}_{self.args.losstype}.pth.tar'
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False,
                filename=os.path.join(self.log_dir2, checkpoint_name))
        else:
            checkpoint_name = f'{self.args.dataset}_checkpoint_{self.args.epochs}_{self.args.losstype}WATERMARK.pth.tar'
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'watermark_state_dict': self.watermark_mlp.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False,
                filename=os.path.join(self.log_dir2, checkpoint_name))
        logging.info(
            f"Model checkpoint and metadata has been saved at {self.log_dir2}")

    def steal(self, train_loader, num_queries, watermark_loader=None):
        print("_" * 100)
        print("Stealing using SimCLR with Enhanced Performance Monitoring")
        print(f"Stealing on {len(train_loader)} batches for {self.args.epochs} epochs")
        print(f"Loss type: {self.loss}")
        print("_" * 100)
        
        # Initialize comprehensive timing metrics
        training_start_time = time.time()
        self.timing_metrics['epoch_times'] = []
        self.timing_metrics['victim_forward_times'] = []
        self.timing_metrics['surrogate_forward_times'] = []
        self.timing_metrics['loss_computation_times'] = []
        self.timing_metrics['backward_pass_times'] = []
        self.timing_metrics['training_losses'] = []
        self.timing_metrics['victim_queries_per_epoch'] = []
        self.timing_metrics['victim_flops_per_epoch'] = []
        self.timing_metrics['surrogate_flops_per_epoch'] = []
        
        self.model.train()
        self.victim_model.eval()
        if self.args.defence == "True":
            self.victim_head.eval()
        if watermark_loader is not None:
            self.watermark_mlp.eval()
        scaler = GradScaler("cuda", enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.log_dir2, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR stealing for {self.args.epochs} epochs.")
        logging.info(f"Using loss type: {self.loss}")
        logging.info(f"Training with gpu: {torch.cuda.is_available()}.")
        logging.info(f"Args: {self.args}")

        # Estimate FLOPs for victim and surrogate models (done once)
        sample_batch = next(iter(train_loader))[0]
        if isinstance(sample_batch, list):
            sample_batch = torch.cat(sample_batch, dim=0)
        sample_input = sample_batch[:1]  # Single sample
        victim_flops_per_sample = self.get_model_flops(self.victim_model, sample_input.shape)
        surrogate_flops_per_sample = self.get_model_flops(self.model, sample_input.shape)
        self.timing_metrics['victim_flops_per_sample'] = victim_flops_per_sample
        self.timing_metrics['surrogate_flops_per_sample'] = surrogate_flops_per_sample
        print(f"Victim model FLOPs per sample: {victim_flops_per_sample:,}")
        print(f"Surrogate model FLOPs per sample: {surrogate_flops_per_sample:,}")

        total_victim_queries = 0
        epochs_completed = 0

        for epoch_counter in range(self.args.epochs):
            epoch_start_time = time.time()
            epoch_victim_forward_time = 0
            epoch_surrogate_forward_time = 0
            epoch_loss_computation_time = 0
            epoch_backward_pass_time = 0
            epoch_loss_sum = 0
            epoch_batches = 0
            victim_queries_this_epoch = 0
            victim_flops_this_epoch = 0
            surrogate_flops_this_epoch = 0
            
            total_queries = 0
            all_reps = None
            y_true = []
            y_pred = []
            y_pred_raw = []
            
            for images, truelabels in tqdm(train_loader, desc=f"Epoch {epoch_counter+1}/{self.args.epochs}"):
                #  # TODO will cause error for resnet surrogate, skip the step for resent surrogate
                # if self.args.surrogate_arch in ["clip-vitb32", "clip-vitb16", "dino-vitb16", "dino-vits16"]:
                #     images = [images]
                if isinstance(images, list):
                    images = torch.cat(images, dim=0)
                images = images.to(self.args.device)
                batch_size = images.shape[0]
                
                # Victim forward pass timing - Count queries based on loss type
                victim_start_time = time.time()
                with torch.no_grad():
                    query_features = self.victim_model(images) # victim model representations
                    # print("victim model", query_features.shape)
                victim_forward_time = time.time() - victim_start_time
                epoch_victim_forward_time += victim_forward_time
                
                victim_queries_this_epoch += batch_size
                victim_flops_this_epoch += victim_flops_per_sample * batch_size
                
                if self.args.defence == "True" and self.loss in ["softnn", "infonce"]: # first type of perturbation defence
                    query_features2 = self.victim_head(images)
                    all_reps = torch.t(query_features2[0].reshape(-1,1))
                    for i in range(1, query_features.shape[0]):
                        sims = self.criterion2(query_features2[i].expand(all_reps.shape[0], all_reps.shape[1]), all_reps)
                        sims = ((sims+1)/2)

                        maxval = sims.max()
                        maxpos = torch.argmax(sims)
                        if i < query_features.shape[0]/2:
                            y_true.append(0)
                        else:
                            if i - query_features.shape[0]/2 == maxpos.item():
                                y_true.append(1)
                            else:
                                y_true.append(0)
                        y_pred_raw.append(maxval.item())
                        if maxval.item() > 0.8:
                            y_pred.append(1)
                            if self.args.sigma > 0:
                                query_features[i] = torch.empty(
                                    query_features[i].size()).normal_(mean=1000,
                                                                      std=self.args.sigma).to(
                                    self.args.device)  # instead of adding, completely change the representation
                        else:
                            y_pred.append(0)
                        all_reps = torch.cat([all_reps, torch.t(query_features2[i].reshape(-1,1))], dim=0)
                elif self.args.defence == "True": # Second type of perturbation defence
                    if self.args.sigma > 0:
                        query_features += torch.empty(query_features.size()).normal_(mean=self.args.mu,std=self.args.sigma).to(self.args.device)  # add random noise to embeddings
                
                # Surrogate forward pass timing
                surrogate_start_time = time.time()
                if self.loss != "symmetrized":
                    features = self.model(images) # current stolen model representation: 512x512 (512 images, 512/128 dimensional representation if head not used / if head used)
                    # print("attck model", features.shape)
                    # raise Exception()
                    surrogate_flops_this_epoch += surrogate_flops_per_sample * batch_size
                surrogate_forward_time = time.time() - surrogate_start_time
                epoch_surrogate_forward_time += surrogate_forward_time
                
                # Loss computation timing
                loss_start_time = time.time()
                if self.loss == "softce":
                    loss = self.criterion(features,F.softmax(features, dim=1)) 
                elif self.loss == "infonce":
                    all_features = torch.cat([features, query_features], dim=0)
                    logits, labels = self.info_nce_loss(all_features)
                    loss = self.criterion(logits, labels)
                elif self.loss == "bce":
                    loss = self.criterion(features, torch.round(torch.sigmoid(query_features))) # torch.round to convert it to one hot style representation
                elif self.loss == "softnn":
                    all_features = torch.cat([features, query_features], dim=0)
                    loss = self.criterion(self.args, all_features, pairwise_euclid_distance, self.tempsn)
                elif self.loss == "supcon":
                    all_features = torch.cat([F.normalize(features, dim=1) , F.normalize(query_features, dim=1) ], dim=0)
                    labels = truelabels.repeat(2) # for victim and stolen features
                    bsz = labels.shape[0]
                    f1, f2 = torch.split(all_features, [bsz, bsz], dim=0)
                    all_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)],
                                         dim=1)
                    loss = self.criterion(all_features, labels)
                elif self.loss == "symmetrized":
                    #https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py#L294
                    # p is the output from the predictor (i.e. stolen model in this case)
                    # z is the output from the victim model (so the direct representation)
                    x1 = images[:int(len(images)/2)]
                    x2 = images[int(len(images)/2):]
                    p1, p2, _, _ = self.model(x1, x2)
                    y1 = self.victim_model(x1).detach()
                    y2 = self.victim_model(x2).detach() # raw representations from victim
                    z1 = self.model.encoder.fc(y1)
                    z2 = self.model.encoder.fc(y2) # pass representations through attacker's encoder
                    loss = -(self.criterion(p1, z2).mean() + self.criterion(p2,
                                                                  z1).mean()) * 0.5
                    # Count FLOPs for symmetrized loss
                    surrogate_flops_this_epoch += surrogate_flops_per_sample * batch_size * 2  # Two forward passes
                elif self.loss == "barlow":
                    x1 = images[:int(len(images) / 2)]
                    x2 = images[int(len(images) / 2):]
                    p1 = self.model(x1)
                    p2 = self.model(x2)
                    y1 = self.victim_model(x1).detach()
                    y2 = self.victim_model(x2).detach()
                    P1 = torch.cat([p1, y1], dim=0) # combine all representations on the first view
                    P2 = torch.cat([p2, y2], dim=0) # combine all representations on the second view
                    loss = self.criterion(P1, P2, self.args.device)
                    # Count FLOPs for barlow loss
                    surrogate_flops_this_epoch += surrogate_flops_per_sample * batch_size * 2  # Two forward passes
                else:
                    loss = self.criterion(features, query_features)
                
                loss_computation_time = time.time() - loss_start_time
                epoch_loss_computation_time += loss_computation_time
                
                # Backward pass timing
                backward_start_time = time.time()
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                backward_pass_time = time.time() - backward_start_time
                epoch_backward_pass_time += backward_pass_time

                n_iter += 1
                total_queries += len(images)
                epoch_loss_sum += loss.item()
                epoch_batches += 1
                
                if total_queries >= num_queries:
                    break

            # Store epoch metrics
            epoch_total_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss_sum / epoch_batches if epoch_batches > 0 else 0
            
            self.timing_metrics['epoch_times'].append(epoch_total_time)
            self.timing_metrics['victim_forward_times'].append(epoch_victim_forward_time)
            self.timing_metrics['surrogate_forward_times'].append(epoch_surrogate_forward_time)
            self.timing_metrics['loss_computation_times'].append(epoch_loss_computation_time)
            self.timing_metrics['backward_pass_times'].append(epoch_backward_pass_time)
            self.timing_metrics['training_losses'].append(avg_epoch_loss)
            self.timing_metrics['victim_queries_per_epoch'].append(victim_queries_this_epoch)
            self.timing_metrics['victim_flops_per_epoch'].append(victim_flops_this_epoch)
            self.timing_metrics['surrogate_flops_per_epoch'].append(surrogate_flops_this_epoch)
            
            total_victim_queries += victim_queries_this_epoch
            epochs_completed += 1
            
            print(f'Epoch {epoch_counter+1}/{self.args.epochs} - Loss: {avg_epoch_loss:.6f}', flush=True)
            print(f'Epoch {epoch_counter+1} timing - Total: {epoch_total_time:.2f}s, Victim: {epoch_victim_forward_time:.2f}s, Surrogate: {epoch_surrogate_forward_time:.2f}s, Loss: {epoch_loss_computation_time:.2f}s, Backward: {epoch_backward_pass_time:.2f}s', flush=True)
            print(f'Victim queries this epoch: {victim_queries_this_epoch:,}, Total FLOPs: {victim_flops_this_epoch + surrogate_flops_this_epoch:,} ({(victim_flops_this_epoch + surrogate_flops_this_epoch)/1e9:.2f} GFLOPs)', flush=True)
            print()

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            if self.args.defence == "True":
                try:
                    import sklearn.metrics
                    f1 = sklearn.metrics.f1_score(np.array(y_true),
                                                  np.array(y_pred))
                    print("f1 score", f1)
                    fpr, tpr, thresholds = sklearn.metrics.roc_curve(np.array(y_true), np.array(y_pred_raw), pos_label=1)
                    print("auc",  sklearn.metrics.auc(fpr, tpr))
                except ImportError:
                    print("sklearn not available for defense metrics")

            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\t")

        # Store total metrics in timing_metrics for access from main
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        self.timing_metrics['total_training_time'] = total_training_time
        self.timing_metrics['total_victim_queries'] = total_victim_queries
        self.timing_metrics['total_victim_flops'] = sum(self.timing_metrics['victim_flops_per_epoch'])
        self.timing_metrics['total_surrogate_flops'] = sum(self.timing_metrics['surrogate_flops_per_epoch'])
        self.timing_metrics['num_epochs'] = epochs_completed
        self.timing_metrics['total_victim_forward_time'] = sum(self.timing_metrics['victim_forward_times'])
        self.timing_metrics['total_surrogate_forward_time'] = sum(self.timing_metrics['surrogate_forward_times'])
        self.timing_metrics['total_loss_computation_time'] = sum(self.timing_metrics['loss_computation_times'])
        self.timing_metrics['total_backward_pass_time'] = sum(self.timing_metrics['backward_pass_times'])

        print(f"Total training completed in {total_training_time:.2f} seconds")
        print(f"Total victim queries: {total_victim_queries:,}")

        logging.info("Stealing has finished.")
        # save model checkpoints
        checkpoint_name = f'stolen_checkpoint_{self.args.num_queries}_{self.loss}_{self.args.datasetsteal}.pth.tar'
        save_checkpoint({
            'epoch': self.args.epochs,
            'victim_arch': self.args.victim_arch,
            'surrogate_arch': self.args.surrogate_arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False,
            filename=os.path.join(self.log_dir2, checkpoint_name))
        logging.info(
            f"Stolen model checkpoint and metadata has been saved at {self.log_dir2}.")
        if watermark_loader is not None:
            self.watermark_mlp.eval()
            self.model.eval()
            watermark_accuracy = 0
            for counter, (x_batch, _) in enumerate(watermark_loader):
                if isinstance(x_batch, list):
                    x_batch = torch.cat(x_batch, dim=0)
                x_batch = x_batch.to(self.args.device)
                logits = self.watermark_mlp(self.model(x_batch))
                y_batch = torch.cat([torch.zeros(self.args.batch_size),
                                     torch.ones(self.args.batch_size)],dim=0).long().to(self.args.device)
                top1 = accuracy(logits, y_batch, topk=(1,))
                watermark_accuracy += top1[0]
            watermark_accuracy /= (counter + 1)
            print(f"Watermark accuracy is {watermark_accuracy.item()}.")
            logging.info(f"Watermark accuracy is {watermark_accuracy.item()}.")
