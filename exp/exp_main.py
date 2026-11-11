import os
import time
import math
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import OepfTSF
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

warnings.filterwarnings("ignore")


def horizon_weights(H: int, mode: str, device, dtype):
    if mode == "none":
        w = torch.ones(H, 1, device=device, dtype=dtype)
    else:
        h = torch.arange(1, H + 1, device=device, dtype=dtype).view(H, 1)
        if mode == "atan":
            w = (-torch.atan(h) + (math.pi / 4.0) + 1.0)
        elif mode == "sqrt":
            w = 1.0 / torch.sqrt(h)
        elif mode == "linear":
            w = (H + 1.0 - h) / H
        else:
            raise ValueError(f"Unknown horizon_weighting='{mode}', choose from none/atan/sqrt/linear")
    return w / (w.mean() + 1e-8)


def weighted_mae(pred: torch.Tensor, true: torch.Tensor, w: torch.Tensor):
    w = w.view(1, -1, 1)
    return (torch.abs(pred - true) * w).mean()


def weighted_mse(pred: torch.Tensor, true: torch.Tensor, w: torch.Tensor):
    w = w.view(1, -1, 1)
    return (((pred - true) ** 2) * w).mean()


def weighted_huber(pred: torch.Tensor, true: torch.Tensor, w: torch.Tensor, delta: float = 1.0):
    w = w.view(1, -1, 1)
    loss = F.smooth_l1_loss(pred, true, beta=delta, reduction="none")
    return (loss * w).mean()


def _set_model_attr(model, name: str, value):
    """Support DataParallel."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, name):
        setattr(m, name, value)


def _linear_warmup(epoch_idx: int, warmup_epochs: int, target: float):
    if warmup_epochs <= 0:
        return float(target)
    t = min(1.0, float(epoch_idx + 1) / float(warmup_epochs))
    return float(target) * t


def _cosine_anneal(epoch_idx: int, total_epochs: int, start: float, end: float):
    if total_epochs <= 1:
        return float(end)
    t = float(epoch_idx) / float(total_epochs - 1)
    c = 0.5 * (1.0 - math.cos(math.pi * t))
    return float(start) + (float(end) - float(start)) * c


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.con_loss_coe_cls_1 = args.con_cls_1
        self.con_loss_coe_cls_2 = args.con_cls_2
        self.con_loss_coe_time = args.con_time

        self.horizon_weighting = str(getattr(args, "horizon_weighting", "atan"))


        self.use_amp = bool(getattr(args, "use_amp", False)) and (self.device.type == "cuda")
        amp_dtype_str = str(getattr(args, "amp_dtype", "bf16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)


        self.train_loss_type = str(getattr(args, "train_loss_type", "mix")).lower()
        self.mix_mse_weight = float(getattr(args, "mix_mse_weight", 0.85))
        self.huber_delta = float(getattr(args, "huber_delta", 1.0))
        self.grad_clip = float(getattr(args, "grad_clip", 0.0))


        self.dist_warmup_epochs = int(getattr(args, "dist_warmup_epochs", 10))
        self.support_warmup_epochs = int(getattr(args, "support_warmup_epochs", 10))

        self.sigma_anneal_from = float(getattr(args, "sigma_anneal_from", getattr(args, "label_sigma_ratio", 1.0)))
        self.sigma_anneal_to = float(getattr(args, "sigma_anneal_to", getattr(args, "label_sigma_ratio", 1.0)))

        self.tau_anneal_from = getattr(args, "tau_anneal_from", None)
        self.tau_anneal_to = getattr(args, "tau_anneal_to", None)

    def _build_model(self):
        model_dict = {"OepfTSF": OepfTSF}
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        return mse_criterion, mae_criterion

    def _train_main_loss(self, outputs, batch_y_cut):
        w = horizon_weights(self.args.pred_len, self.horizon_weighting, outputs.device, outputs.dtype)
        if self.train_loss_type == "wmae":
            return weighted_mae(outputs, batch_y_cut, w)
        elif self.train_loss_type == "wmse":
            return weighted_mse(outputs, batch_y_cut, w)
        elif self.train_loss_type == "whuber":
            return weighted_huber(outputs, batch_y_cut, w, delta=self.huber_delta)
        elif self.train_loss_type == "mix":
            mse_part = weighted_mse(outputs, batch_y_cut, w)
            mae_part = weighted_mae(outputs, batch_y_cut, w)
            a = float(self.mix_mse_weight)
            return a * mse_part + (1.0 - a) * mae_part
        else:
            raise ValueError("train_loss_type must be one of: wmae/wmse/whuber/mix")

    @torch.no_grad()
    def _eval_loop(self, data_loader, metric_mode: str):
        self.model.eval()
        total = []
        for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            f_dim = -1 if self.args.features == "MS" else 0
            y_target = batch_y[:, -self.args.pred_len:, f_dim:]

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs, _, _, _ = self.model(batch_x, None)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]

                if metric_mode == "mse":
                    loss = F.mse_loss(outputs, batch_y_cut)
                elif metric_mode == "mae":
                    loss = F.l1_loss(outputs, batch_y_cut)
                else:
                    raise ValueError("metric_mode must be 'mse' or 'mae'")

            total.append(float(loss.item()))
        self.model.train()
        return float(np.average(total))

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            dist_target = float(getattr(self.args, "dist_sup_weight", 0.0))
            sup_target = float(getattr(self.args, "support_reg_weight", 0.0))

            dist_eff = _linear_warmup(epoch, self.dist_warmup_epochs, dist_target)
            sup_eff = _linear_warmup(epoch, self.support_warmup_epochs, sup_target)
            sigma_eff = _cosine_anneal(epoch, self.args.train_epochs, self.sigma_anneal_from, self.sigma_anneal_to)

            _set_model_attr(self.model, "dist_sup_weight_eff", dist_eff)
            _set_model_attr(self.model, "support_reg_weight_eff", sup_eff)
            _set_model_attr(self.model, "label_sigma_ratio_eff", sigma_eff)

            if self.tau_anneal_from is not None and self.tau_anneal_to is not None:
                tau_eff = _cosine_anneal(epoch, self.args.train_epochs,
                                         float(self.tau_anneal_from), float(self.tau_anneal_to))
                _set_model_attr(self.model, "support_tau_eff", tau_eff)

            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                f_dim = -1 if self.args.features == "MS" else 0
                y_target = batch_y[:, -self.args.pred_len:, f_dim:]

                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs, con_loss_cls_1, con_loss_cls_2, con_loss_time = self.model(batch_x, y_target)

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss_main = self._train_main_loss(outputs, batch_y_cut)
                    loss = loss_main + (
                        self.con_loss_coe_cls_1 * con_loss_cls_1
                        + self.con_loss_coe_cls_2 * con_loss_cls_2
                        + self.con_loss_coe_time * con_loss_time
                    )

                train_loss.append(float(loss.item()))

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / max(iter_count, 1)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                # backward/step
                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(model_optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    model_optim.step()

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time}")
            train_loss_avg = float(np.average(train_loss))


            val_mse = self._eval_loop(vali_loader, metric_mode="mse")
            val_mae = self._eval_loop(vali_loader, metric_mode="mae")
            test_mse = self._eval_loop(test_loader, metric_mode="mse")
            test_mae = self._eval_loop(test_loader, metric_mode="mae")

            print(
                f"Epoch: {epoch+1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss_avg:.7f} | "
                f"Val MSE: {val_mse:.7f} (early-stop) | Val MAE: {val_mae:.7f} | "
                f"Test MSE: {test_mse:.7f} | Test MAE: {test_mae:.7f}"
                f"dist_w:{dist_eff:.4f} sup_w:{sup_eff:.4f} sigma_ratio:{sigma_eff:.4f}"
            )


            early_stopping(val_mse, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        try:
            os.remove(best_model_path)
        except OSError:
            pass

        return self.model

    @torch.no_grad()
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            ckpt = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        preds, trues = [], []
        folder_path = "./test_results/" + setting + "/"
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs, _, _, _ = self.model(batch_x, None)

            f_dim = -1 if self.args.features == "MS" else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)

            if i % 20 == 0:
                inp = batch_x.detach().cpu().numpy()
                gt = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                pd_ = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd_, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse = metric(preds, trues)
        print(f"mse:{mse}, mae:{mae}")

        with open("result.txt", "a") as f:
            f.write(setting + "  \n")
            f.write(f"mse:{mse}, mae:{mae}")
            f.write("\n\n")
        return
