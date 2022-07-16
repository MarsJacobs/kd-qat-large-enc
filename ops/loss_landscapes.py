import copy
import re

import numpy as np
import torch
from tqdm import tqdm
import ops.norm as norm
import ops.tests as tests
import ops.meters as meters
from torch.nn import CrossEntropyLoss, MSELoss

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return torch.sum((- targets_prob * student_likelihood), dim=-1).mean()

def normalize_filter(bs, ws):
    bs = {k: v.float() for k, v in bs.items()}
    ws = {k: v.float() for k, v in ws.items()}

    norm_bs = {}
    for k in bs:
        ws_norm = torch.norm(ws[k], dim=0, keepdim=True)
        bs_norm = torch.norm(bs[k], dim=0, keepdim=True)
        norm_bs[k] = ws_norm / (bs_norm + 1e-7) * bs[k]

    return norm_bs


def ignore_bn(ws):
    ignored_ws = {}
    for k in ws:
        if len(ws[k].size()) < 2:
            ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
        else:
            ignored_ws[k] = ws[k]
    return ignored_ws


def ignore_running_stats(ws):
    return ignore_kw(ws, ["num_batches_tracked"])


def ignore_kw(ws, kws=None):
    kws = [] if kws is None else kws

    ignored_ws = {}
    for k in ws:
        if any([re.search(kw, k) for kw in kws]):
            ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
        else:
            ignored_ws[k] = ws[k]
    return ignored_ws


def rand_basis(ws, gpu=True):
    return {k: torch.randn(size=v.shape, device="cuda" if gpu else None) for k, v in ws.items()}


def create_bases(model, kws=None, gpu=True, ws0=None):
    kws = [] if kws is None else kws
    # ws0 = copy.deepcopy(model.state_dict())
    bases = [rand_basis(ws0, gpu) for _ in range(2)]  # Use two bases
    bases = [normalize_filter(bs, ws0) for bs in bases]
    bases = [ignore_bn(bs) for bs in bases]
    bases = [ignore_kw(bs, kws) for bs in bases]

    return bases


def get_loss_landscape(model, n_ff, dataset, transform=None,
                       bases=None, kws=None,
                       cutoffs=(0.0, 0.9), bins=np.linspace(0.0, 1.0, 11), verbose=False, period=10, gpu=True,
                       x_min=-1.0, x_max=1.0, n_x=11, y_min=-1.0, y_max=1.0, n_y=11, teacher_model=None, kd_type=None):
    model = model.cuda() if gpu else model.cpu()
    model = copy.deepcopy(model)
    ws0 = copy.deepcopy(model.state_dict())
    kws = [] if kws is None else kws
    bases = create_bases(model, kws, gpu, ws0=ws0) if bases is None else bases
    xs = np.linspace(x_min, x_max, n_x)
    ys = np.linspace(y_min, y_max, n_y)
    ratio_grid = np.stack(np.meshgrid(xs, ys), axis=0).transpose((1, 2, 0))

    metrics_grid = {}

    for ratio in tqdm(ratio_grid.reshape([-1, 2])):
        ws = copy.deepcopy(ws0)
        gs = [{k: r * bs[k] for k in bs} for r, bs in zip(ratio, bases)]
        gs = {k: torch.sum(torch.stack([g[k] for g in gs]), dim=0) + ws[k] for k in gs[0]}
        model.load_state_dict(gs)
        # print("Grid: ", ratio, end=", ")
        nll_meter = meters.AverageMeter("nll")
        # MSKIM

        loss = 0.
        tmp_loss = 0.
        for step, batch in enumerate(dataset):
            batch = tuple(t.to("cuda") for t in batch)
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
            with torch.no_grad():
                if kd_type is not None:
                    teacher_logits, teacher_atts, teacher_reps, teacher_probs, teacher_values = teacher_model(input_ids, segment_ids, input_mask)
                student_logits, student_atts, student_reps, student_probs, student_values = model(input_ids, segment_ids, input_mask, teacher_outputs=None)
            
            if kd_type == "pred":
                loss = soft_cross_entropy(student_logits,teacher_logits)
            elif kd_type == "trm":
                for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
                    tmp_loss = MSELoss()(student_rep, teacher_rep)
                    loss += tmp_loss
            else:
                lprobs = torch.nn.functional.log_softmax(student_logits, dim=-1)
                loss = torch.nn.functional.nll_loss(lprobs, label_ids, reduction='sum')
            
            nll_meter.update(loss.item())
            if verbose:
                print("NLL: {0:0.2f}".format(nll_meter.avg))
            
            metrics = nll_meter.avg


        # *metrics, cal_diag = tests.test(model, n_ff, dataset, transform=transform,
        #                                 cutoffs=cutoffs, bins=bins, verbose=verbose, period=period, gpu=gpu)
        l1, l2 = norm.l1(model, gpu).item(), norm.l2(model, gpu).item()
        metrics_grid[tuple(ratio)] = (l1, l2, metrics)
        # metrics_grid[tuple(ratio)] = (metrics)

    return metrics_grid
