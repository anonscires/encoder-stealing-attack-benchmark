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

torch.manual_seed(0)
from simclr import SimCLR

class CustomSimCLR(SimCLR):

    def __init__(self, stealing=False, victim_model=None, victim_head = None, entropy_model = None, watermark_mlp = None, logdir='', loss=None, *args,
                 **kwargs):
        super().__init__(stealing=stealing, victim_model=victim_model, victim_head=victim_head, 
                 entropy_model=entropy_model, watermark_mlp=watermark_mlp, logdir=logdir, 
                 loss=loss, *args, **kwargs)
        

    def train(self, train_loader, watermark_loader=None):

        raise Exception("Training has not been implemented yet for CustomSimCLR.")


    def steal(self, train_loader, num_queries, watermark_loader=None):

        print("_" * 100)
        print("Stealing using CustomSimCLR")
        print(f"Stealing on {len(train_loader)} batches for {self.args.epochs} epochs")
        print("_" * 100)

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


        # embed_path = "" #TODO: Specify the correct path to the class embeddings file
        # if not os.path.exists(embed_path):
        #     raise FileNotFoundError(f"Class-embeds file not found: {embed_path}")
        # class_embeds = torch.load(embed_path, map_location="cuda")

        # if not torch.is_tensor(class_embeds):
        #     class_embeds = torch.tensor(class_embeds)
        
        # class_embeds = class_embeds.to("cuda")
        # print("Loaded class embeddings shape:", getattr(class_embeds, "shape", None))

        for epoch_counter in range(self.args.epochs):
            total_queries = 0
            all_reps = None
            y_true = []
            y_pred = []
            y_pred_raw = []
            for victim_images, surrogate_images, truelabels in tqdm(train_loader):

                if "vit" in self.args.victim_arch:
                    victim_images = [victim_images]

                # TODO will cause error for resnet surrogate, skip the step for resent surrogate
                if "vit" in self.args.surrogate_arch:
                    surrogate_images = [surrogate_images]

                victim_images = torch.cat(victim_images, dim=0)
                victim_images = victim_images.to(self.args.device)

                surrogate_images = torch.cat(surrogate_images, dim=0)
                surrogate_images = surrogate_images.to(self.args.device)

                # Use gradients if GradCAM defense is enabled, otherwise disable gradients
                if hasattr(self.args, 'gradcam') and self.args.gradcam != "False":
                    # GradCAM defense requires gradients for saliency computation
                    query_features = self.victim_model(victim_images) # victim model representations
                else:
                    with torch.no_grad():
                        query_features = self.victim_model(victim_images) # victim model representations
                        # print("victim model", query_features.shape)
                if self.args.defence == "True" and self.loss in ["softnn", "infonce"]: # first type of perturbation defence
                    query_features2 = self.victim_head(victim_images)
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
                if self.loss != "symmetrized":
                    features = self.model(surrogate_images) # current stolen model representation: 512x512 (512 images, 512/128 dimensional representation if head not used / if head used)
                    # print("attck model", features.shape)
                    # raise Exception()
                if self.loss == "softce":
                    loss = self.criterion(features,F.softmax(features, dim=1)) 
                elif self.loss == "infonce":
                    all_features = torch.cat([features, query_features], dim=0)
                    logits, labels = self.info_nce_loss(all_features)
                    vision_loss = self.criterion(logits, labels)

                    # #TODO: add if condition for text encoder stealing
                    #  # Build a (batch_size, embed_dim) tensor of class embeddings corresponding to y
                    # y_idx = truelabels.long().squeeze()
                    # if y_idx.dim() == 0:
                    #     y_idx = y_idx.unsqueeze(0)
                    # # Make sure indices are on the same device as class_embeds for indexing
                    # y_idx = y_idx.to(class_embeds.device)
                    # batch_class_embeds = class_embeds[y_idx].detach()  # shape: (batch_size, embed_dim)
                    # # Ensure result is on the training device
                    # batch_class_embeds = batch_class_embeds.to("cuda")


                    # all_text_features = torch.cat([features, batch_class_embeds], dim=0)
                    # text_logits, text_labels = self.info_nce_loss(all_text_features)
                    # text_loss = self.criterion(text_logits, text_labels)

                    loss = vision_loss #+ text_loss
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
                    raise Exception("Symmetrized loss not implemented yet.")
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
                elif self.loss == "barlow":
                    raise Exception("Barlow loss not implemented yet.")
                    x1 = images[:int(len(images) / 2)]
                    x2 = images[int(len(images) / 2):]
                    p1 = self.model(x1)
                    p2 = self.model(x2)
                    y1 = self.victim_model(x1).detach()
                    y2 = self.victim_model(x2).detach()
                    P1 = torch.cat([p1, y1], dim=0) # combine all representations on the first view
                    P2 = torch.cat([p2, y2], dim=0) # combine all representations on the second view
                    loss = self.criterion(P1, P2, self.args.device)
                else:
                    loss = self.criterion(features, query_features)
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                n_iter += 1
                total_queries += len(victim_images)
                
                if total_queries >= num_queries:
                    break

            # Clear GPU cache after each epoch
            torch.cuda.empty_cache()
            
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            if self.args.defence == "True":
                raise Exception("Defence not implemented yet.")
                f1 = sklearn.metrics.f1_score(np.array(y_true),
                                              np.array(y_pred))
                print("f1 score", f1)
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(np.array(y_true), np.array(y_pred_raw), pos_label=1)
                print("auc",  sklearn.metrics.auc(fpr, tpr))

            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\t")
            
            # Print consumed GPU memory
            if torch.cuda.is_available():
                allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
                reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
                max_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
                available_memory = max_memory - allocated_memory
                print(f"GPU Memory - Allocated: {allocated_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB, Available: {available_memory:.2f} GB, Total: {max_memory:.2f} GB")
                logging.info(f"GPU Memory - Allocated: {allocated_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB, Available: {available_memory:.2f} GB, Total: {max_memory:.2f} GB")

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
                x_batch = torch.cat(x_batch, dim=0)
                x_batch = x_batch.to(self.args.device)
                logits = self.watermark_mlp(self.model(x_batch))
                y_batch = torch.cat([torch.zeros(self.args.batch_size),
                                     torch.ones(self.args.batch_size)],dim=0).long().to(self.args.device)
                top1 = accuracy(logits, y_batch, topk=(1,))
                watermark_accuracy += top1[0]
            watermark_accuracy /= (counter + 1)
            print(f"Watermark accuracy is {watermark_accuracy.item()}.")
            logging.info(f"Watermark accuracy is {watermark_accuracy.item()}.")\
            