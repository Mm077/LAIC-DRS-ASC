import os
import os.path as osp
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import datetime
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import (
    load_pretrained_weights, load_checkpoint,MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.optim import build_optimizer, build_lr_scheduler
import random
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from utils.draw_confusion import compute_metrics

from trainers.ClusMatch.ClusMatchThresholing import ClusMatchThresholing
from trainers.ClusMatch.PseudoLabeling import PseudoLabeling

_tokenizer = _Tokenizer()
from tqdm import tqdm

# 以下7项是来自 ClusMatch 的损失类定义
def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

class CELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, reduction='none'):
        return ce_loss(logits, targets, reduction)

def consistency_loss(logits, targets, name='ce', mask=None):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()

class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None):
        return consistency_loss(logits, targets, name, mask)

def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # print(f'label_hist: {label_hist}')
    
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()

def negative_loss(positive_mask,negative_mask,logits_x_ulb_s):
    probs = torch.softmax(logits_x_ulb_s, dim=-1)
    
    acceptmask=(negative_mask).sum(dim=1).ge(0.5).to(logits_x_ulb_s.dtype)
    all_negative_loss = -(1 - (negative_mask * probs).sum(1)).clamp(1e-8,1).log()
    negative_loss = all_negative_loss*acceptmask#*(1/neg_mask_cnt)

    loss = negative_loss.mean()#+h_ne#+2.0*positive_loss.mean()
    if torch.isnan(negative_loss).sum() > 0:
        negative_loss=replace_inf_to_zero(negative_loss)
        print(negative_loss)

    return loss

# 来自tac的InfoNCE函数
class DistillLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(DistillLoss, self).__init__()
        # 这里的n是batch size
        # self.n = 32
        self.class_num = class_num
        self.temperature = temperature
        # self.mask = self.mask_correlated_clusters(class_num).cuda()
        # self.mask = self.mask_correlated_clusters(self.n).cuda()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_clusters(self):
        # N = 2 * class_num
        n = self.n
        N = 2 * n
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        # for i in range(class_num):
            # mask[i, class_num + i] = 0
            # mask[class_num + i, i] = 0
        for i in range(n):
            mask[i, n + i] = 0
            mask[n + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        
        # 使用列作为instance
        # c_i = c_i.t()
        # c_j = c_j.t()
        # n = c_i.shape[1]        
        # N = 2 * self.class_num
        # N = 2 * n    

        # 使用行作为instance
        self.n = n = c_i.shape[0]
        N = 2 * n
        # 改到这里初始化mask：
        self.mask = self.mask_correlated_clusters().cuda()
        c = torch.cat((c_i, c_j), dim=0)

        c = F.normalize(c, dim=1)
        sim = c @ c.T / self.temperature
        # sim_i_j = torch.diag(sim, self.class_num)
        # sim_j_i = torch.diag(sim, -self.class_num)
        sim_i_j = torch.diag(sim, n)
        sim_j_i = torch.diag(sim, -n)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels) / N

        return loss


class GeneralizedCrossEntropy(nn.Module):
    """Computes the generalized cross-entropy loss, from `
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>`_
    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """
    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q
        return torch.mean(loss)

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        # print("Before permute, x.shape =", x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner_for_usl_1st(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # n_cls: number of classes
        n_cls = len(classnames)
        # n_ctx: number of context tokens embedding
        n_ctx = cfg.TRAINER.COOP.N_CTX
        # init ctx, for example: a photo of a {}

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]


        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        print("Initializing class-specific contexts")
        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        
        # [SOS] a photo of a X X X . [...] [EOS]
        
        # classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = ["a photo of a X X " + "." for name in classnames]
        # prompts = ["a photo of a" + "X " * (n_ctx) + "." for name in classnames]
        # prompts = ["a photo of a" + "X " * (n_ctx) + "." for i in range(n_cls)]
        # prompts = ["a photo of a person doing" + "X " * (n_ctx) + "." for i in range(n_cls)]
        # prompt_used_ = "a centered satellite photo of " 
        prompt_used_ = "a photo of a " 
        # prompt_used_ = "a photo of a person doing " 
        # prompt_used_ = " "
        prompts = [prompt_used_ + "X " * (n_ctx) + "." for i in range(n_cls)]
        # prompts = [prompt_used_ + "X " * (n_ctx) + "texture" for i in range(n_cls)]
        counts_prompts_nouns = len(prompt_used_.split())
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]


        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            # embedding: [ncls, 77, 512]
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # for example: [SOS] a cat sitting on the mat . [EOS]
        # token_prefix: [ncls, 5, 512], 输入前是[SOS]
        self.register_buffer("token_prefix", embedding[:, :counts_prompts_nouns + 1, :])  # [SOS] a photo of a 
        # self.register_buffer("token_prefix", embedding[:, :7, :])  # [SOS] a photo of a person doing
        # self.register_buffer("token_prefix", embedding[:, :7, :])  # [SOS] a photo of a person doing
        # token_suffix: [ncls, 70, 512], 输入前是[]
        self.register_buffer("token_suffix", embedding[:, counts_prompts_nouns + 1 + n_ctx:, :])  # CLS, EOS
        # self.register_buffer("token_suffix", embedding[:, 7 + n_ctx:, :])  # CLS, EOS
        # self.register_buffer("token_suffix", embedding[:, 6 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        # ctx: [ncls, 16, 512]
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 5, dim)
                    ctx,     # (n_cls, nctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        # elif self.class_token_position == "middle":
        #     half_n_ctx = self.n_ctx // 2
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i : i + 1, :, :]
        #         class_i = suffix[i : i + 1, :name_len, :]
        #         suffix_i = suffix[i : i + 1, name_len:, :]
        #         ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
        #         ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,     # (1, 1, dim)
        #                 ctx_i_half1,  # (1, n_ctx//2, dim)
        #                 class_i,      # (1, name_len, dim)
        #                 ctx_i_half2,  # (1, n_ctx//2, dim)
        #                 suffix_i,     # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        # elif self.class_token_position == "front":
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i : i + 1, :, :]
        #         class_i = suffix[i : i + 1, :name_len, :]
        #         suffix_i = suffix[i : i + 1, name_len:, :]
        #         ctx_i = ctx[i : i + 1, :, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,  # (1, 1, dim)
        #                 class_i,   # (1, name_len, dim)
        #                 ctx_i,     # (1, n_ctx, dim)
        #                 suffix_i,  # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        if cfg.PROMPT_STAGE == "FIRST":
            self.prompt_learner = PromptLearner_for_usl_1st(cfg, classnames, clip_model)

        else:
            raise ValueError("Unsupported PROMPT_STAGE")

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits

@TRAINER_REGISTRY.register()
class CoOp_NEG(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE_loss = GeneralizedCrossEntropy(q=0.8)

        # 来自clusmatch的损失
        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()
        # 来自clusmatch的方法
        # 这里的参数先写死，后续可以改成可配置的
        self.ClusMatchThresholing = ClusMatchThresholing(10)
        self.PseudoLabeling = PseudoLabeling()
        self.ContrastiveLoss = DistillLoss(10, 0.5)

        # 用于保存此次训练的每个ep的结果：
        self.test_history = []

        

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # 指示当前是第几轮epoch
        self.epoch_this = 0
        print(f"initializing epoch_this = {self.epoch_this}")

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building CustomCLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Freezing all except prompt_learner")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        if torch.cuda.device_count() > 1:
            print("Multi-GPU training enabled")
            self.model = nn.DataParallel(self.model)


    
    def forward_backward(self, forward_type, batch_x, batch_u, batch_u_2):

        # TODO：这里获取每个batch需要计算的样本
        # _x：有标签样本 _u, _2：无标签样本的两种增强方法
        image_x, label_x, pl_label_x, idx_x = self.parse_batch_train(batch_x)
        image_u, image_u_2, _, idx_u = self.parse_batch_train(batch_u)
        # image_u_2, _, idx_u_2 = self.parse_batch_train(batch_u_2)
        
        # print(f"idx_u[:10]: {idx_u[:10]}, idx_u_2[:10]: {idx_u_2[:10]}[:10]")

        prec = self.cfg.TRAINER.COOP.PREC
        
        if prec == "amp":
            return 0
            with autocast():
                output = self.model(image)
                loss = self._compute_loss(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        elif forward_type == "warmup":
            # warmup时先用全部的正标签初始化
            output = self.model(image_x)
            loss = self._compute_loss(output, label_x)
            self.model_backward_and_update(loss)
        
        elif forward_type == "s_e_c":
            num_lb = image_x.size(0)  # 有标签的样本数量

            # TODO:此时用负伪标签损失
            inputs = torch.cat([image_x, image_u, image_u_2], dim=0)
            outputs = self.model(inputs).float()
            # 分别是强弱版本的 logits 和 feats 的输出
            logits_x_lb = outputs[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = outputs[num_lb:].chunk(2)
            logits_all = torch.cat([logits_x_lb, logits_x_ulb_w], dim=0)
            # feats_x_lb = outputs['feat'][:num_lb]
            # feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)

            # 有标签监督损失 替换为原来使用的MAE损失试试什么情况：
            sup_loss = self._compute_loss(logits_x_lb, label_x)

            # 此处计算来自TAC的entropy loss
            def entropy_2(logit):
                logit = logit.mean(dim=0)
                logit_ = torch.clamp(logit, min=1e-9)
                b = logit_ * torch.log(logit_)
                return -b.sum()
            # ent_loss2 = entropy_2(torch.softmax(logits_x_lb, dim=1)) + entropy_2(torch.softmax(logits_x_ulb_s, dim=1)) + entropy_2(torch.softmax(logits_x_ulb_w, dim=1))
            # ent_loss2 = entropy_2(logits_x_ulb_s) + entropy_2(logits_x_ulb_w)
            ent_loss2 = entropy_2(torch.softmax(logits_all, dim=1))
            # ent_loss2 = entropy_2(torch.softmax(logits_x_lb, dim=1))

            # 此处计算来自tac的强弱增强一致性损失
            def consistency_loss(anchors, neighbors):
                b, n = anchors.size()
                similarity = torch.bmm(anchors.view(b, 1, n), neighbors.view(b, n, 1)).squeeze()
                ones = torch.ones_like(similarity)
                consistency_loss = F.binary_cross_entropy(similarity, ones)

                return consistency_loss
            # consis_loss2 = consistency_loss(torch.softmax(logits_x_ulb_s, dim=1), torch.softmax(logits_x_ulb_w, dim=1))
            consis_loss2 = F.mse_loss(torch.softmax(logits_x_ulb_s, dim=1), torch.softmax(logits_x_ulb_w, dim=1))

            # 计算总的损失：
            self.sup_loss_weight = 1.0 # 监督损失
            self.ent_loss_2_weight = 0.1 # 0.1 # tac的熵正则
            self.consis_loss_2_weight = 2.0 # tac的强弱增强一致性损失
            
            # 打印一下损失项的值
            # print(f"sup_loss_weight: {self.sup_loss_weight}, ent_loss_weight: {self.ent_loss_2_weight}, consis_loss_weight: {self.consis_loss_2_weight}")

            loss = (
                self.sup_loss_weight * sup_loss 
                - self.ent_loss_2_weight * ent_loss2
                + self.consis_loss_2_weight * consis_loss2
                )
            
            self.model_backward_and_update(loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        # 这里只测试有标注信息的label
        # return {"loss": loss.item(), "acc": compute_accuracy(output, label)[0].item()}
        if forward_type == "use_neg" or forward_type == "s_e_c":
            output = outputs[:num_lb]
        # return {"loss": loss.item(), "acc": compute_accuracy(output, label_x)[0].item()}
        return {"loss": loss.item()}

    def _compute_loss(self, output, label):
        if self.cfg.USE_LOSS == "MAE":
            return self.GCE_loss(output, label)
        elif self.cfg.USE_LOSS == "CE":
            return F.cross_entropy(output, label)
        else:
            raise RuntimeError("Unsupported loss type")
    
    def save_ctx(self, save_path):
        if hasattr(self.model.prompt_learner, "ctx"):
            torch.save({"ctx": self.model.prompt_learner.ctx.detach().cpu()}, save_path)
        elif hasattr(self.model.prompt_learner, "ctx_yy"):
            print(f"ctx_yy: {self.model.prompt_learner.ctx_yy.shape}, ctx_xx: {self.model.prompt_learner.ctx_xx.shape}")
            torch.save({
                "ctx_yy": self.model.prompt_learner.ctx_yy.detach().cpu(),
                "ctx_xx": self.model.prompt_learner.ctx_xx.detach().cpu()
            }, save_path)
        else:
            raise RuntimeError("No ctx found to save.")

    
        # trans images to GPU
   
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        pl_label = batch["pl_label"]
        index = batch["index_filename"]
        input = input.to(self.device)
        label = label.to(self.device)
        pl_label = pl_label.to(self.device)
        index = index.to(self.device)
        # 对应有强弱增强版本的情况：
        if "img2" in batch:
            input2 = batch["img2"]
            input2 = input2.to(self.device)
            return input, input2, label, index
        return input, label, pl_label, index
    
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        index = batch["index_filename"]
        input = input.to(self.device)
        label = label.to(self.device)
        index = index.to(self.device)
        return input, label, index

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        # TODO:这里没有决定用哪个
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)
        train_loader_u_2_iter = iter(self.train_loader_u_2)

        end = time.time()

        # 决定当前epoch采用什么forward_type
        epoch_this = self.epoch
        if epoch_this < self.cfg.TRAINER.COOP.FIRST_STAGE_EP:
            forward_type = "warmup"
        elif epoch_this >= self.cfg.TRAINER.COOP.FIRST_STAGE_EP:
            forward_type = "use_neg"

        # !!!!!此处直接使用第三种forwardtype
        # forward_type = "pll"
        forward_type = "s_e_c"
    

        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            try:
                batch_u_2 = next(train_loader_u_2_iter)
            except StopIteration:
                train_loader_u_2_iter = iter(self.train_loader_u_2)
                batch_u_2 = next(train_loader_u_2_iter)

            data_time.update(time.time() - end)

            
            loss_summary = self.forward_backward(forward_type, batch_x, batch_u, batch_u_2)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"EPOCH [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                # info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                # info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
        # 结束一个epoch
        self.epoch_this += 1
    
    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test(save_text_embeddings=True)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    @torch.no_grad()
    def test(self, split=None, save_text_embeddings=False, ep=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if save_text_embeddings:
            print("saving text embeddings")
            model = self.model
            # prompt_learner = model.prompt_learner
            prompts = model.prompt_learner()
            tokenized_prompts = model.tokenized_prompts
            text_features = model.text_encoder(prompts, tokenized_prompts)

            if ep is not None:
                self.prompt_save_path = self.cfg.PROMPT_SAVEPATH + f'prompt_ep{ep}.pt'
            else:
                self.prompt_save_path = self.cfg.PROMPT_SAVEPATH + 'prompt.pt'
            # self.prompt_save_path = f"./{self.cfg.DATASET.SUBSAMPLE_CLASSES}.pt"
            torch.save(text_features, self.prompt_save_path)
            print(f"text embeddings saved to {self.prompt_save_path}")
  
    def print_test_table(self):
        print("Results:")
        print("-" * 50)
        print(f"{'Epoch':^8} | {'ACC':^10} | {'NMI':^10} | {'ARI':^10}")
        for row in self.test_history:
            print(f"{row['epoch']:^8} | "
                f"{row['acc']:^10.4f} | "
                f"{row['nmi']:^10.4f} | "
                f"{row['ari']:^10.4f}")
        print("-" * 50)

    @torch.no_grad()
    def test_self(self, split=None, save_text_embeddings=False, ep=None):  
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
            split = "test"

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"=> Evaluate on the *{split}* set, use for self")
        
        pre_labels = []
        labels = []
        test_self = True

        if self.prompt_save_path is not None and os.path.exists(self.prompt_save_path):

            text_features = torch.load(self.prompt_save_path)

            print(f"Evaluate using text prompt from {self.prompt_save_path}")
            for batch_idx, batch in enumerate(data_loader):
                image, label, _ = self.parse_batch_test(batch)
                labels.append(label)

                image_features = self.model.image_encoder(image.type(self.model.dtype))                    
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logits = image_features @ text_features.t()
                pre_label = torch.argmax(F.softmax(logits, dim=1), dim=1)
                pre_labels.append(pre_label)
        else:
            print("Get no text prompts this training!!")
            test_self = False
        
        if test_self:
    
            pre_labels = torch.cat(pre_labels, dim=0)
            labels = torch.cat(labels, dim=0) 

            # 聚类性能指标     
            acc, nmi, _, ari,  = compute_metrics(labels.cpu(), pre_labels.cpu(), min_samples_per_class=5, superclass_mapping=None, 
                                                 print_results=False,return_reassign=False, draw_matrix=False, 
                                                 fname=(self.cfg.PROMPT_SAVEPATH + f'{ep}' if ep is not None else self.cfg.PROMPT_SAVEPATH) )
            self.test_history.append({
                "epoch": ep,
                "acc": acc,
                "nmi": nmi,
                "ari": ari
            })
            self.print_test_table()
        
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
        self.test(split="test", save_text_embeddings=True, ep=self.epoch + 1)
        # 接下来，测试当前ep的性能：
        self.test_self(split="test", save_text_embeddings=False, ep=self.epoch + 1)




    @torch.no_grad()
    def eval_pred(self, split=None, save_text_embeddings=False):
        from generate_partialP import generate_partialP
        from collections import defaultdict
        import numpy as np
        import os
        import torch.nn.functional as F
        # 绘制最大 logit 值的直方图
        def plot_histogram(max_values, bins=10, fname=None):
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            counts, bin_edges, _ = plt.hist(max_values, bins=bins, edgecolor='black', alpha=0.7)
            plt.title('Histogram of Maximum Logit Values')
            plt.xlabel('Max Logit Value')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.5)
            for i in range(len(counts)):
                plt.text((bin_edges[i] + bin_edges[i+1]) / 2, counts[i], str(int(counts[i])), 
                        ha='center', va='bottom')
            plt.tight_layout()
            plt.savefig(fname)
        
        
        lab2cname = self.lab2cname # dict {label: classname}
        num_classes = len(lab2cname)
          
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        split = "test"
        if split == "test" and self.test_loader is not None:
            data_loader = self.test_loader
        else:
            split = "train_u"  # in case val_loader is None
            data_loader = self.train_loader_u
        print(f"=> Evaluate on the *{split}* set, use for self")
        
        pre_labels = []
        logits_ = []
        labels = []
        max_pred_values = []
        sample_records = []
        idxs = []
        test_self = True

        # 开始使用训练好的Prompt作为分类器进行划分
 
        testname = None
        self.prompt_save_path = testname + '.pt'
        # self.prompt_save_path = './prompt-CIFAR20_USL-seed6-0shots-ONLY_NAME_X-vit_b32-MAE-b16knn5-bs=32-ep=20.pt'
        if self.prompt_save_path is not None and os.path.exists(self.prompt_save_path):
            print(f"Evaluate using text prompt from {self.prompt_save_path}")
            print(f"data_loader length: {len(data_loader)}")
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                image, label, index = self.parse_batch_test(batch)
                idxs.append(index)
                labels.append(label)

                image_features = self.model.image_encoder(image.type(self.model.dtype))
                text_features = torch.load(self.prompt_save_path)
    
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logits = image_features @ text_features.t()

                softmax_probs = F.softmax(logits, dim=1)
                max_probs, pred_labels = softmax_probs.max(dim=1)

                pre_labels.append(pred_labels)
                logits_.append(logits)
                max_pred_values.extend(max_probs.cpu().numpy())

                # 🔺记录每个样本信息：index, 预测标签, 概率, 真实标签
                for idx, pred, prob, gt in zip(index.cpu().numpy(), pred_labels.cpu().numpy(), max_probs.cpu().numpy(), label.cpu().numpy()):
                    sample_records.append((idx, pred, prob, gt)) 
        else:
            print("Get no text prompts this training!!")
            test_self = False

        if test_self:

            logits_ = torch.cat(logits_, dim=0)
            pre_labels = torch.cat(pre_labels, dim=0)
            labels = torch.cat(labels, dim=0)
            idxs = torch.cat(idxs, dim=0)


            labels_sorted = torch.empty_like(labels)
            for i, k in enumerate(idxs):
                labels_sorted[k] = labels[i]

            pls_sorted = torch.empty_like(pre_labels)
            for i, k in enumerate(idxs):
                pls_sorted[k] = pre_labels[i]
            
            logits_sorted = torch.empty_like(logits_)
            for i, k in enumerate(idxs):
                logits_sorted[k] = logits_[i]
            
            acc, nmi, _, ari,  = compute_metrics(labels_sorted.cpu(), pls_sorted.cpu(), min_samples_per_class=5, superclass_mapping=None, 
                                                 print_results=False,return_reassign=False, draw_matrix=True, fname='./confusion_matrix/cifar10-/train_5w_test_5w')

            print(f"ACC: {acc:.4f}")
            print(f"NMI: {nmi:.4f}")
            print(f"ARI: {ari:.4f}")

            # 分析此时logits得到的pll
            print("分析此时训练结果得到的partial label")
            generate_partialP(logits_sorted, pls_sorted)

        print('eval preds finished')


def train_two_stage(cfg, trainer):
    print("\n==> 🔵 Stage 1: Training with fixed prefix")
    
    cfg.PROMPT_STAGE = "FIRST"
    trainer1 = trainer(cfg)
    trainer1.train()

    ctx_path = os.path.join(cfg.OUTPUT_DIR, "ctx_stage1.pt")
    trainer1.save_ctx(ctx_path)

    print("\n==> 🟢 Stage 2: Training new prompt with frozen ctx")
    
    cfg.PROMPT_STAGE = "SECOND"
    cfg.PROMPT_CTX_PATH = ctx_path
    trainer2 = trainer(cfg)
    trainer2.train()
