import os, math, time, json, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, utils as tv_utils
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights

from PIL import Image
from tqdm.auto import tqdm

try:
    from torch import amp
    USE_NEW_AMP = True
except Exception:
    amp = None
    USE_NEW_AMP = False

@contextmanager
def preserve_train_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()

def save_ckpt(path: str, model, opt, scaler, scheduler, epoch: int, step: int, best_score: float, args: dict):
    torch.save({
        "epoch": epoch,
        "step": step,
        "best_score": best_score,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": args,
    }, path)

class ProofOfQuality:
    def __init__(self, out_dir: str):
        self.out_dir = os.path.join(out_dir, "quality_proof")
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_file = os.path.join(self.out_dir, "q_logs.csv")
        if not os.path.isfile(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("step,avg_q,min_q,max_q,margin_mean\n")

    def save_visual_samples(self, images: torch.Tensor, q: torch.Tensor, step: int):
        # images: [B, 3, 112, 112], q: [B]
        images = images.detach().cpu()
        q = q.detach().cpu()
        idx = torch.argsort(q)
        worst = (images[idx[:8]] * 0.5 + 0.5).clamp(0.0, 1.0) # [8, 3, 112, 112]
        best  = (images[idx[-8:]] * 0.5 + 0.5).clamp(0.0, 1.0) # [8, 3, 112, 112]
        tv_utils.save_image(worst, os.path.join(self.out_dir, f"step_{step}_low_q.jpg"), nrow=4)
        tv_utils.save_image(best,  os.path.join(self.out_dir, f"step_{step}_high_q.jpg"), nrow=4)

    def append_csv(self, step: int, avg_q: float, min_q: float, max_q: float, margin_mean: float):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{step},{avg_q:.6f},{min_q:.6f},{max_q:.6f},{margin_mean:.6f}\n")

def apply_corruption_tensor(x: torch.Tensor, corruption: str) -> torch.Tensor:
    # x: [B, 3, 112, 112]
    if corruption == "none": return x
    if corruption == "lowlight": return (x * 0.6).clamp(-1, 1) # [B, 3, 112, 112]
    if corruption == "noise":
        return (x + torch.randn_like(x) * 0.07).clamp(-1, 1) # [B, 3, 112, 112]
    if corruption == "blur":
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) # [B, 3, 112, 112]
    return x

# ===================== Model =====================

class ResNet50Embed(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.fc = nn.Linear(2048, d, bias=False)
        self.bn = nn.BatchNorm1d(d)
        self.prelu = nn.PReLU(d)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # x: [B, 3, 112, 112]
        x = self.backbone(x).flatten(1) # [B, 2048]
        f = self.prelu(self.bn(self.fc(x))) # [B, D]
        fn = F.normalize(f, p=2, dim=1) # [B, D]
        return f, fn



class SubCenterArcFace(nn.Module):
    def __init__(self, D: int, C: int, K: int = 2, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.C, self.K, self.scale, self.margin = C, K, scale, margin
        self.W = nn.Parameter(torch.empty(self.C * self.K, int(D))) # [C*K, D]
        nn.init.xavier_uniform_(self.W)

    def forward(self, f_raw: torch.Tensor, y: torch.Tensor, q: Optional[torch.Tensor], use_q_margin: bool):
        # f_raw: [B, D], y: [B], q: [B]
        x = F.normalize(f_raw.float(), p=2, dim=1) # [B, D]
        w = F.normalize(self.W.float(), p=2, dim=1) # [C*K, D]
        cos_all = F.linear(x, w).clamp(-1 + 1e-7, 1 - 1e-7) # [B, C*K]
        B = cos_all.size(0)
        cos_max, _ = cos_all.view(B, self.C, self.K).max(dim=2) # [B, C]

        cos_y = cos_max.gather(1, y.view(-1, 1)).squeeze(1) # [B]
        sin_y = torch.sqrt(torch.clamp(1.0 - cos_y * cos_y, min=0.0)) # [B]

        if q is not None and use_q_margin:
            m_i = self.margin * q.to(cos_y.dtype).view(-1) # [B]
            phi = cos_y * torch.cos(m_i) - sin_y * torch.sin(m_i) # [B]
        else:
            phi = cos_y * math.cos(self.margin) - sin_y * math.sin(self.margin) # [B]

        logits = cos_max.clone() # [B, C]
        logits.scatter_(1, y.view(-1, 1), phi.view(-1, 1)) # [B, C]
        return logits * self.scale # [B, C]

class FaceNet(nn.Module):
    def __init__(self, C, D=512, K=2, scale=64.0, margin=0.5, q_ema_beta=0.99, q_min=0.2, q_max=1.0):
        super().__init__()
        self.backbone = ResNet50Embed(D)
        self.head = SubCenterArcFace(D, C, K=K, scale=scale, margin=margin)
        self.q_ema_beta, self.q_min, self.q_max = q_ema_beta, q_min, q_max
        self.register_buffer("ema_norm", torch.tensor(1.0)) # [1]

    def _compute_q(self, f_raw: torch.Tensor) -> torch.Tensor:
        # f_raw: [B, D]
        with torch.no_grad():
            n = f_raw.float().norm(dim=1) # [B]
            bmean = n.mean().clamp_min(1e-6) # [1]
            self.ema_norm.mul_(self.q_ema_beta).add_((1.0 - self.q_ema_beta) * bmean)
            q = (n / self.ema_norm.clamp_min(1e-6)).clamp(self.q_min, self.q_max) # [B]
            return q

    def forward(self, x: torch.Tensor, y: torch.Tensor, use_q_margin: bool):
        # x: [B, 3, 112, 112], y: [B]
        f_raw, f_norm = self.backbone(x) # [B, D], [B, D]
        q = self._compute_q(f_raw) if self.training else torch.ones(x.size(0), device=x.device) # [B]
        logits = self.head(f_raw, y, q=q, use_q_margin=use_q_margin) # [B, C]
        return logits, f_raw, f_norm, q

# ===================== Eval Logic =====================

@dataclass
class MS1MPair:
    i1: int; i2: int; same: int

@torch.no_grad()
def ms1m_verif_eval(model: FaceNet, ds_full, pairs: List[MS1MPair], device, corruption="none"):
    model.eval()
    tfm = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    emb: Dict[int, torch.Tensor] = {}

    uniq = sorted({p.i1 for p in pairs} | {p.i2 for p in pairs})
    for idx in uniq:
        path, _ = ds_full.samples[idx]
        x = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device) # [1, 3, 112, 112]
        x = apply_corruption_tensor(x, corruption)
        _, e = model.backbone(x) # [1, D]
        emb[idx] = e.cpu()

    sims, labs = [], []
    for p in pairs:
        s = F.cosine_similarity(emb[p.i1], emb[p.i2]).item()
        sims.append(s); labs.append(p.same)

    sims_t, labs_t = torch.tensor(sims), torch.tensor(labs) # [N], [N]
    best_acc = 0.0
    for th in torch.linspace(-1, 1, 200):
        acc = ((sims_t >= th).long() == labs_t).float().mean().item()
        if acc > best_acc: best_acc = acc
    return {"acc": best_acc, "mean_pos": sims_t[labs_t==1].mean().item(), "mean_neg": sims_t[labs_t==0].mean().item()}

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, required=True)
    ap.add_argument("--max_class_idx", type=int, default=1000)
    ap.add_argument("--out", type=str, default="./runs_face")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--q_gamma", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.out, "tb"))
    proof = ProofOfQuality(args.out)

    tfm = transforms.Compose([transforms.Resize((112, 112)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    ds_full = ImageFolder(args.train_root, transform=tfm)
    indices = [i for i, y in enumerate(ds_full.targets) if y < args.max_class_idx]
    dl = DataLoader(Subset(ds_full, indices), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = FaceNet(args.max_class_idx, margin=args.margin).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dl)*args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    ce_none = nn.CrossEntropyLoss(reduction="none")

    verif_pairs = [MS1MPair(random.choice(indices), random.choice(indices), random.randint(0,1)) for _ in range(400)]

    step, best_score = 0, 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device) # [B, 3, 112, 112], [B]

            with torch.cuda.amp.autocast(enabled=True):
                logits, f_raw, f_norm, q = model(x, y, use_q_margin=True) # logits: [B, C], q: [B]
                loss_ce = ce_none(logits, y) # [B]
                wgt = torch.pow(q.clamp_min(1e-6), args.q_gamma) # [B]
                loss = (loss_ce * wgt).mean() # [1]

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            if step % 50 == 0:
                writer.add_scalar("Loss/train", loss.item(), step)
                writer.add_scalar("Q/mean", q.mean().item(), step)
                proof.append_csv(step, q.mean().item(), q.min().item(), q.max().item(), args.margin * q.mean().item())
                pbar.set_postfix(loss=f"{loss.item():.3f}", q=f"{q.mean():.2f}")

            if step % 500 == 0:
                proof.save_visual_samples(x, q, step)
                model.eval()
                res = ms1m_verif_eval(model, ds_full, verif_pairs, device)
                writer.add_scalar("Acc/clean", res["acc"], step)
                if res["acc"] > best_score:
                    best_score = res["acc"]
                    save_ckpt(os.path.join(args.out, "best.pt"), model, opt, scaler, scheduler, epoch, step, best_score, vars(args))
                model.train()

            step += 1

if __name__ == "__main__":
    main()
