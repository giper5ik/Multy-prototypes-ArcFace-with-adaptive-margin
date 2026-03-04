import os, json, argparse, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image


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
        x = self.backbone(x).flatten(1)
        f = self.prelu(self.bn(self.fc(x)))
        fn = F.normalize(f, p=2, dim=1)
        return f, fn

class DummyHead(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args, **kwargs):
        raise RuntimeError("Head not used in eval.")

class FaceNetEval(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = ResNet50Embed(embed_dim)
        self.head = DummyHead()
        self.register_buffer("ema_norm", torch.tensor(1.0))

    def forward(self, x):
        return self.backbone(x)[1]


def apply_corruption_tensor(x: torch.Tensor, corruption: str) -> torch.Tensor:
    if corruption == "none":
        return x
    if corruption == "lowlight":
        return (x * 0.6).clamp(-1, 1)
    if corruption == "noise":
        return (x + torch.randn_like(x) * 0.07).clamp(-1, 1)
    if corruption == "blur":
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x


@dataclass
class Pair:
    i1: int
    i2: int
    same: int

def build_per_class_indices(ds: ImageFolder, min_id: int, max_id_exclusive: int) -> Dict[int, List[int]]:
    per: Dict[int, List[int]] = {}
    for idx, y in enumerate(ds.targets):
        y = int(y)
        if min_id <= y < max_id_exclusive:
            per.setdefault(y, []).append(idx)
    return per

def choose_val_ids(per: Dict[int, List[int]], num_ids: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    valid = [cid for cid, idxs in per.items() if len(idxs) >= 2]
    rng.shuffle(valid)
    return valid[:num_ids]

def make_pairs(per: Dict[int, List[int]], val_ids: List[int], pairs_per_id: int, num_neg: int, seed: int) -> List[Pair]:
    rng = random.Random(seed)
    pairs: List[Pair] = []

    for cid in val_ids:
        idxs = per[cid]
        for _ in range(pairs_per_id):
            a, b = rng.sample(idxs, 2)
            pairs.append(Pair(a, b, 1))

    for _ in range(num_neg):
        c1, c2 = rng.sample(val_ids, 2)
        a = rng.choice(per[c1])
        b = rng.choice(per[c2])
        pairs.append(Pair(a, b, 0))

    rng.shuffle(pairs)
    return pairs


def eval_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

class IdxDS(torch.utils.data.Dataset):
    def __init__(self, ds: ImageFolder, indices: List[int]):
        self.ds = ds
        self.indices = indices
        self.tfm = eval_transform()
    def __len__(self): return len(self.indices)
    def __getitem__(self, k):
        idx = self.indices[k]
        path, _y = self.ds.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        return x, idx

@torch.no_grad()
def compute_embeddings(model: FaceNetEval, ds: ImageFolder, uniq_indices: List[int], device: torch.device,
                       bs: int, nw: int, corruption: str, tta_flip: bool) -> Dict[int, torch.Tensor]:
    loader = DataLoader(IdxDS(ds, uniq_indices), batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    emb: Dict[int, torch.Tensor] = {}

    model.eval()
    for x, idxs in loader:
        x = x.to(device, non_blocking=True)
        x = apply_corruption_tensor(x, corruption)
        _, e1 = model.backbone(x)
        if tta_flip:
            xf = torch.flip(x, dims=[3])
            _, e2 = model.backbone(xf)
            e = F.normalize(e1 + e2, p=2, dim=1)
        else:
            e = e1
        e = e.detach().float().cpu()
        for ii, vec in zip(idxs.tolist(), e):
            emb[int(ii)] = vec
    return emb


def best_threshold(sims: torch.Tensor, labs: torch.Tensor) -> float:
    best_acc = -1.0
    best_th = 0.0
    for th in torch.linspace(-1.0, 1.0, 2001):
        pred = (sims >= th).long()
        acc = (pred == labs).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_th = float(th.item())
    return best_th

def eval_pairs_holdout(emb: Dict[int, torch.Tensor], pairs: List[Pair], seed: int) -> Dict[str, float]:
    sims = []
    labs = []
    same_s = []
    diff_s = []

    for p in pairs:
        if p.i1 not in emb or p.i2 not in emb:
            continue
        s = float(F.cosine_similarity(emb[p.i1].view(1,-1), emb[p.i2].view(1,-1)).item())
        sims.append(s)
        labs.append(p.same)
        (same_s if p.same==1 else diff_s).append(s)

    sims_t = torch.tensor(sims)
    labs_t = torch.tensor(labs)

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(sims_t.numel(), generator=g)
    cut = sims_t.numel() // 2
    fit = perm[:cut]
    test = perm[cut:]

    th = best_threshold(sims_t[fit], labs_t[fit])
    pred = (sims_t[test] >= th).long()
    acc = (pred == labs_t[test]).float().mean().item()

    mean_same = float(torch.tensor(same_s).mean().item()) if same_s else 0.0
    mean_diff = float(torch.tensor(diff_s).mean().item()) if diff_s else 0.0

    return {
        "acc": float(acc),
        "th": float(th),
        "mean_same": mean_same,
        "mean_diff": mean_diff,
        "n_pairs_used": int(sims_t.numel())
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--train_max_id", type=int, default=40000)
    ap.add_argument("--val_id_span", type=int, default=20000, help="Validate on [train_max_id, train_max_id+val_id_span)")

    ap.add_argument("--embed_dim", type=int, default=512)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num_ids", type=int, default=2000)
    ap.add_argument("--pairs_per_id", type=int, default=1)
    ap.add_argument("--num_neg", type=int, default=2000)
    ap.add_argument("--tta_flip", action="store_true")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--out_json", type=str, default="results_ms1m_out_of_id.json")
    args = ap.parse_args()

    device = torch.device(args.device)

    ds = ImageFolder(args.train_root, transform=None)
    min_id = int(args.train_max_id)
    max_id = int(args.train_max_id + args.val_id_span)

    per = build_per_class_indices(ds, min_id=min_id, max_id_exclusive=max_id)
    val_ids = choose_val_ids(per, args.num_ids, args.seed)
    if len(val_ids) < args.num_ids:
        print(f"[WARN] requested num_ids={args.num_ids}, available={len(val_ids)} in range [{min_id},{max_id}).")

    pairs = make_pairs(per, val_ids, args.pairs_per_id, args.num_neg, args.seed)
    uniq = sorted({p.i1 for p in pairs} | {p.i2 for p in pairs})

    print(f"[VAL_IDS] range=[{min_id},{max_id}) chosen={len(val_ids)}")
    print(f"[PAIRS] total={len(pairs)} uniq_imgs={len(uniq)} pos={len(val_ids)*args.pairs_per_id} neg={args.num_neg}")

    model = FaceNetEval(embed_dim=args.embed_dim).to(device)

    ck = torch.load(args.ckpt, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)


    results = {}
    for corr in ["none", "blur", "noise", "lowlight"]:
        emb = compute_embeddings(model, ds, uniq, device=device,
                                 bs=args.batch_size, nw=args.num_workers,
                                 corruption=corr, tta_flip=args.tta_flip)
        m = eval_pairs_holdout(emb, pairs, seed=args.seed)
        results[corr] = m
        print(f"[OUT-OF-ID {corr}] acc={m['acc']:.4f} th={m['th']:.3f} same={m['mean_same']:.3f} diff={m['mean_diff']:.3f} n={m['n_pairs_used']}")

    out = {
        "train_root": args.train_root,
        "ckpt": args.ckpt,
        "train_max_id": args.train_max_id,
        "val_id_span": args.val_id_span,
        "seed": args.seed,
        "num_ids": args.num_ids,
        "pairs_per_id": args.pairs_per_id,
        "num_neg": args.num_neg,
        "tta_flip": bool(args.tta_flip),
        "results": results,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[SAVE]", args.out_json)


if __name__ == "__main__":
    main()
