import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.window.rolling import Window
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors

import torch
from torch.nn import functional as F
from torch import nn
from torch.nn.modules.loss import MSELoss
from tqdm import tqdm


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


def make_shuffled_id(ids):
    """
    Args:
        ids (pd.Series): id
    """
    uid = ids.unique().tolist()
    sfl_uid = np.random.permutation(uid)
    uid_to_sfl_uid = dict(zip(uid, sfl_uid))

    sfl_uids = ids.map(uid_to_sfl_uid)
    return sfl_uids


class Feature:
    def __init__(self):
        return

    def transform(self, df):
        ft = pd.DataFrame([], index=df.index)

        #ft["x_u_in"] = np.log(df["u_in"] + 1)
        ft["x_u_in"] = df["u_in"]
        ft['x_u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
        #ft['x_u_in-ave'] = df["u_in"] - df.groupby('breath_id')['u_in'].rolling(window=3, min_periods=1, center=True).mean().fillna(0).reset_index(level=0, drop=True)

        ft["x_u_out"] = df["u_out"]

        #ft["x_R"] = (df["R"] - 5) / 45
        ft["x_R"] = df["R"]
        #ft["x_R5"] = (df["R"] == 5) * 1
        #ft["x_R20"] = (df["R"] == 20) * 1
        #ft["x_R50"] = (df["R"] == 50) * 1
        #ft["x_R"] = df["R"].map({5: -1, 20: 0, 50: 1})

        #ft["x_C"] = (df["C"] - 10) / 40
        ft["x_C"] = df["C"]
        #ft["x_C10"] = (df["C"] == 10) * 1
        #ft["x_C20"] = (df["C"] == 20) * 1
        #ft["x_C50"] = (df["C"] == 50) * 1
        #ft["x_C"] = df["C"].map({10: -1, 20: 0, 50: 1})


        return ft


class Target:
    def __init__(self):
        return

    def transform(self, df):
        y = df[["pressure"]].copy()
        return y

    def inverse(self, y):
        inv_y = y.copy()
        return inv_y


class EnsembleModel:
    def __init__(self, models):
        self.models = models
        return

    def predict(self, x):
        pred = None
        for mdl in self.models:
            if pred is None:
                pred = mdl.predict(x) / len(self.models)
            else:
                pred = pred + mdl.predict(x) / len(self.models)
        return pred


class Model:
    def __init__(
        self, 
        scaling_prms,
        dnn_prms, 
        tr_prms, 
        nn_prms, 
        seq_len=80,
        use_seq_len=32, 
        train_batch_size=128,
        pred_batch_size=128,
        dev="cuda",
        ):
        """
        Args:
            nn_prms (dict): nearest neighbor
        """
        self.use_seq_len = use_seq_len
        self.dnn = DNN(**dnn_prms)
        self.tr_prms = tr_prms
        self.seq_len = seq_len
        self.dev = dev

        self.x_scalers = [RobustScaler() for i in range(scaling_prms["n_scaler"])]
        self.y_scaler = RobustScaler()

        self.train_batch_size = train_batch_size
        self.pred_batch_size = pred_batch_size

        self.gnrnb = VentilatorGroupNearestNeighbors(nn_prms, remove_nearest=False)

        return

    def set_dev(self, dev):
        self.dev = dev
        self.dnn.to(dev)

    def fit(self, tr_x, tr_y, vl_x, vl_y):
        """
        Args:
            tr_x (pd.DataFrame): columns=x feature
        """
        # group
        tr_group = self.make_group(tr_x)
        vl_group = self.make_group(vl_x)

        # fit scaler
        # x
        if len(self.x_scalers) > 0:
            tmp_x = self.extract_use_seq_len(tr_x)
            for i, scl in enumerate(self.x_scalers):
                scl.fit(tmp_x[:, i: i+1])
        # y
        tmp_y = self.extract_use_seq_len(tr_y)
        self.y_scaler.fit(tmp_y)

        # scaling
        tr_x = self.scaling(tr_x, self.x_scalers)
        tr_y = self.scaling(tr_y, [self.y_scaler])
        vl_x = self.scaling(vl_x, self.x_scalers)
        vl_y = self.scaling(vl_y, [self.y_scaler])

        # to seq
        # (sample, seq, feat)
        tr_x = self.to_sequence(tr_x)
        tr_y = self.to_sequence(tr_y)
        vl_x = self.to_sequence(vl_x)
        vl_y = self.to_sequence(vl_y)

        # fit nn
        self.gnrnb.fit(tr_x[:, :self.use_seq_len, 0], tr_y[:, :self.use_seq_len], tr_group)
        score = self.gnrnb.score(vl_x[:, :self.use_seq_len, 0], vl_y[:, :self.use_seq_len], vl_group)

        """
        # fit dnn
        # dataset
        tr_ds = VentilatorDataset(tr_x, tr_y, self.use_seq_len)
        tr_dl = get_dataloader(tr_ds, self.train_batch_size, shuffle=True, drop_last=True)
        vl_ds = VentilatorDataset(vl_x, vl_y, self.use_seq_len)
        vl_dl = get_dataloader(vl_ds, self.pred_batch_size, shuffle=False, drop_last=False)

        # train
        self.dnn = self.dnn.to(self.dev)
        trainer = Trainer(dev=self.dev, **self.tr_prms)
        score = trainer.run(self.dnn, tr_dl, vl_dl)
        """

        return score

    def predict(self, x):
        """
        Args:
            x (pd.DataFrame): columns=x feature
        """
        # group
        group = self.make_group(x)

        # scaling
        x = self.scaling(x, self.x_scalers)

        # to seq
        x = self.to_sequence(x)

        # nbnr predicting
        pred = self.gnrnb.predict(x[:, :self.use_seq_len, 0], group)

        """
        # dnn predicting
        ds = VentilatorDataset(x, None, self.use_seq_len)
        dl = get_dataloader(ds, self.pred_batch_size, shuffle=False, drop_last=False)

        predictor = Predictor(self.dev, do_print=True)
        pred = predictor.run(self.dnn, dl) # (breath_id, seq, 1)
        """

        # 0 pad in out of seq
        pred = pred[:, :, 0] # (breath_id, seq)
        pred = np.pad(pred, [(0, 0), (0, self.seq_len - self.use_seq_len)], "constant")

        # inverse transform of y
        pred = self.y_scaler.inverse_transform(pred)

        # flatten
        pred = np.ravel(pred)

        return pred

    def scaling(self, x, scalers):
        """
        Args:
            x (pd.DataFrame): 
        """
        x = x.copy()
        if len(scalers) > 0:
            for i, scl in enumerate(scalers):
                x.iloc[:, i: i+1] = scl.transform(x.iloc[:, i: i+1])
        return x

    def to_sequence(self, x):
        """
        Args:
            x (pd.DataFrame): columns=feature
        Returns:
            ndarray: (breath_id, seq, feat)
        """
        seq = x.values.reshape([-1, self.seq_len, x.shape[1]]).copy()
        return seq

    def extract_use_seq_len(self, x):
        """
        Args:
            x (pd.DataFrame): columns=x feature
        """
        seq = self.to_sequence(x)
        seq = seq[:, :self.use_seq_len, :]
        ex_x = seq.reshape([-1, seq.shape[2]])
        return ex_x

    def make_group(self, x):
        group = x["x_R"].astype(str) + "+" + x["x_C"].astype(str)
        group = self.to_sequence(pd.DataFrame(group))
        group = group[:, 0, 0]
        return group


class DNN(nn.Module):
    def __init__(self, n_feat, n_channel, dropout, n_rc_layer):
        super(DNN, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_feat, 64),
            nn.ReLU(),
        )
        self.rc = nn.GRU(64, n_channel, batch_first=True, bidirectional=True, num_layers=n_rc_layer, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(n_channel * 2, 64), 
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq, feat)
        Returns:
            y: (batch, seq, 1)
        """
        h = x
        h = self.mlp(h)
        h = self.rc(h)[0]
        h = self.head(h)
        return h


class IdentityScaler:
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if type(x) is pd.DataFrame:
            return x.values.copy()
        elif type(x) is pd.Series:
            return x.values.copy()
        else:
            return x.copy()

    def inverse_transform(self, x):
        if type(x) is pd.DataFrame:
            return x.values.copy()
        elif type(x) is pd.Series:
            return x.values.copy()
        else:
            return x.copy()


class VentilatorLoss(nn.Module):
    def __init__(self):
        super(VentilatorLoss, self).__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, input, target):
        loss = self.l1loss(input, target)
        return loss


class VentilatorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, use_seq_len):
        """
        Args:
            x (np.ndarray): (breath_id, seq, feature)
            y (np.ndarray): (breath_id, seq, 1)
            use_seq_len (int): モデルに入力するシーケンス長
        """
        self.x = x[:, :use_seq_len, :].astype('float32')
        if y is not None:
            self.y = y[:, :use_seq_len, :].astype('float32')
        else:
            self.y = None

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

    def __len__(self):
        return len(self.x)


def get_dataloader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=0):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


class Trainer:
    def __init__(self, criterion, opt, opt_params, sch_params, epochs, use_step_rate=1, dev="cuda", grad_accum_steps=1, prefix='', save_best=True, maximize_score=True):
        """
        Args:
            opt: 'sgd', 'adam', 'adamw'
        """
        self.criterion = criterion
        self.opt_params = opt_params
        self.opt = optimizer(opt)

        self.sch_params = sch_params
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR if sch_params is not None else None

        self.epochs = epochs
        self.use_step_rate = use_step_rate if use_step_rate is not None else 1
        self.grad_accum_steps = grad_accum_steps
        self.save_best = save_best
        self.maximize_score = maximize_score

        self.dev = dev
        self.prefix = prefix

    def run(self, model, tr_loader, vl_loader):
        steps_per_epoch = int(len(tr_loader) * self.use_step_rate) // self.grad_accum_steps

        optimizer = self.opt(model.parameters(), **self.opt_params)

        scheduler = self.scheduler(
            optimizer, 
            total_steps=steps_per_epoch * self.epochs,
            **self.sch_params,
        ) if self.scheduler is not None else None

        model_path = self.prefix + 'model.pth'
        log_path = self.prefix + 'tr_log.csv'

        loglist = []
        best_score = None
        for ep in range(self.epochs):
            print('\nepoch ', ep)

            for param_group in optimizer.param_groups:
                print('lr ', param_group['lr'])
                now_lr = param_group['lr']

            tr_log = self.run_epoch(model, optimizer, scheduler, tr_loader, train=True)
            vl_log = self.run_epoch(model, None, None, vl_loader, train=False)

            print()
            self.print_result(tr_log, 'Train')
            self.print_result(vl_log, 'Valid')

            # best score
            score = vl_log['score']
            if not np.isnan(score):
                if best_score is None:
                    best_score = score
                    print('Update best score :', best_score)

                    if self.save_best:
                        self.save_model(model, model_path)
                else:
                    if (self.maximize_score and best_score < score) or (not self.maximize_score and best_score > score):
                        best_score = score
                        print('Update best score :', best_score)

                        if self.save_best:
                            self.save_model(model, model_path)

            if not self.save_best:
                self.save_model(model, model_path)

            # save log
            columns = ['ep', 'lr'] + ['tr_' + k for k in tr_log.keys()] + ['vl_' + k for k in tr_log.keys()]
            loglist.append([ep, now_lr] + list(tr_log.values()) + list(vl_log.values()))
            pd.DataFrame(loglist, columns=columns).to_csv(log_path)

        # load best
        model.load_state_dict(torch.load(model_path))
        model.to(self.dev)

        return best_score

    def run_epoch(self, model, optimizer, scheduler, loader, train=True):
        if train:
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        total_loss = 0
        score_calculater = ScoreCalculater()

        if not train:
            batch_idx_limit = len(loader)
        else:
            batch_idx_limit = int(len(loader) * self.use_step_rate)

        for batch_idx, data in enumerate(tqdm(loader)):
            if batch_idx >= batch_idx_limit:
                break

            x = data[0].to(self.dev)
            y = data[1].to(self.dev)

            with torch.set_grad_enabled(train):
                output = model(x)

                loss = self.criterion(output, y)
                loss = loss / self.grad_accum_steps

                if train:
                    loss.backward()
                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                        # scheduler
                        if scheduler is not None:
                            scheduler.step()

                total_loss += loss.item() * self.grad_accum_steps

            with torch.no_grad():
                score_calculater.add_results(y.cpu().numpy(), output.cpu().numpy())

        # loss
        total_loss = total_loss / (batch_idx + 1)
        # score
        score = score_calculater.calc_score()

        result = dict(loss=total_loss)
        result.update(score)
        return result

    def print_result(self, result, title):
        print(title + ' Loss: %.4f | Score: %.5f' % (result['loss'], result['score']))
        print(result)

    def save_model(self, model, model_path):
        torch.save(model.to('cpu').state_dict(), model_path)
        model = model.to(self.dev)
        print('Save model :', model_path)


class ScoreCalculater:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def add_results(self, y_true, y_pred):
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def calc_score(self):
        y_true = np.concatenate(self.y_true) # (batch, seq, y)
        y_pred = np.concatenate(self.y_pred) # (batch, seq, y)

        scores = {}
        scr = self._mae(y_true, y_pred)
        scores['score'] = scr
        scores['mae'] = scr

        return scores
    
    def _apply_multi_axis(self, t, p, func, kwarg={}):
        n = t.shape[1]
        scr = []
        for i in range(n):
            scr.append(func(t[:, i], p[:, i], **kwarg))
        scr = np.average(scr)
        return scr

    def _mae(self, t, p):
        return np.abs(t - p).mean()


def optimizer(opt):
    if opt == 'sgd':
        return torch.optim.SGD
    elif opt == 'adam':
        return torch.optim.Adam
    elif opt == 'adamw':
        return torch.optim.AdamW
    else:
        return None


class Predictor:
    def __init__(self, dev="cuda", do_print=False):
        self.dev = dev
        self.do_print = do_print

    def run(self, model, loader):
        model.eval()
        preds = []

        if self.do_print:
            ite = enumerate(tqdm(loader))
        else:
            ite = enumerate(loader)

        for batch_idx, data in ite:
            x = data.to(self.dev)

            with torch.set_grad_enabled(False):
                output = model(x)
            preds.append(output.cpu().numpy())

        preds = np.concatenate(preds)
        return preds


class VentilatorNearestNeighbors:
    def __init__(self, nn_params, remove_nearest=False):
        # class sklearn.neighbors.NearestNeighbors(*, n_neighbors=5, radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None)
        self.nrnb = NearestNeighbors(**nn_params)
        self.remove_nearest = remove_nearest

    def fit(self, x, y):
        """
        Args:
            x (np.ndarray): (sample, seq)
            y (np.ndarray): (sample, seq, 1)
        """
        self.x = x.copy()
        self.y = y.copy()

        self.nrnb.fit(self.x)
        return self

    def predict(self, x):
        return self.average_kneighbors(x)

    def average_kneighbors(self, x):
        """
        Args:
            x (pd.DataFrame): (sample, feature)
        """
        # (sample, n_neighbors)
        ind = self.nrnb.kneighbors(x, return_distance=False)
        if self.remove_nearest:
            # リークを防ぐため一番近い結果(自分自身)を除く
            ind = ind[:, 1:]

        ind = np.ravel(ind)

        # (sample * n_neighbors, seq)
        av_neighbor = self.y[ind]
        # (sample, n_neighbors, seq)
        av_neighbor = av_neighbor.reshape([x.shape[0], -1, av_neighbor.shape[1], 1])
        av_neighbor = av_neighbor.mean(axis=1)

        return av_neighbor

    def score(self, x, y):
        pred = self.predict(x)
        score = np.abs(pred - y).mean()
        return score


class VentilatorGroupNearestNeighbors:
    def __init__(self, nn_params, remove_nearest=False):
        # class sklearn.neighbors.NearestNeighbors(*, n_neighbors=5, radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None)
        self.nn_params = nn_params
        self.remove_nearest = remove_nearest

        self.group_idxs = None
        self.vnrnbs = None

    def fit(self, x, y, group):
        """
        Args:
            x (np.ndarray): (sample, seq)
            y (np.ndarray): (sample, seq, 1)
            group (np.ndarray): (sample,)
        """
        self.group_idxs = np.unique(group).tolist()

        self.vnrnbs = {}
        for gr_idx in self.group_idxs:
            is_in_gr = (group == gr_idx)
            gr_x = x[is_in_gr]
            gr_y = y[is_in_gr]

            vnn = VentilatorNearestNeighbors(self.nn_params, self.remove_nearest)
            vnn.fit(gr_x, gr_y)

            self.vnrnbs[gr_idx] = vnn

        return self

    def predict(self, x, group):
        pred = np.zeros_like(x)[:, :, None]

        for gr_idx in self.group_idxs:
            is_in_gr = (group == gr_idx)
            gr_x = x[is_in_gr]
            
            gr_pred = self.vnrnbs[gr_idx].predict(gr_x)
            pred[is_in_gr] = gr_pred

        return pred

    def score(self, x, y, group):
        pred = self.predict(x, group)
        score = np.abs(pred - y).mean()
        return score


class Evaluation:
    def __init__(self, oof_df):
        self.oof_df = oof_df
        self.uniq_folds = sorted(self.oof_df["fold"].unique().tolist())
        return

    def _get_fold_result(self, df, fold):
        return df[df["fold"] == fold].copy()

    def _get_inspiratory_phase(self, df):
        return df[df["u_out"] == 0].copy()

    def _mae(self, df):
        y_true = df["pressure"]
        y_pred = df["pred"]
        mae = np.abs(y_true - y_pred).mean()
        return mae

    def calc_scores(self):
        rslts = {}
        for fold in self.uniq_folds:
            rslt = {}

            df = self._get_fold_result(self.oof_df, fold)
            df = self._get_inspiratory_phase(df)

            rslt["mae"] = self._mae(df)

            rslts[fold] = rslt

        keys = rslts[list(rslts.keys())[0]].keys()
        av_rslt = {k: np.average([v[k] for v in rslts.values()]) for k in keys}
        rslts["av"] = av_rslt

        return rslts

    def calc_breath_mae(self):
        df = self._get_inspiratory_phase(self.oof_df)
        df["error"] = np.abs(df["pressure"] - df["pred"])

        breath_mae = df[["breath_id", "error"]].groupby("breath_id").mean()
        return breath_mae

    def plot_breath(self, breath_id):
        bdf = self.oof_df[self.oof_df["breath_id"]==breath_id].copy()
        bdf = bdf.sort_values("id")

        bdf = bdf[["u_in", "pressure", "pred", "R", "C"]]
        bdf[["u_in", "pressure", "pred"]].plot()
        R = bdf["R"].iloc[0]
        C = bdf["C"].iloc[0]
        plt.title(f"breath_id {breath_id} (R{R},C{C})")
        plt.show()


def test1():
    tr_file = "input/ventilator-pressure-prediction/train.csv"
    tr_df = pd.read_csv(tr_file)
    tr_df = tr_df.iloc[:80*100]

    ft = Feature()
    tr_x = ft.transform(tr_df)
    tg = Target()
    tr_y = tg.transform(tr_df)

    model_prms = {
        "scaling_prms": dict(n_scaler=1),
        "dnn_prms": dict(n_feat=tr_x.shape[1], n_channel=64, dropout=0.0, n_rc_layer=2), 
        "tr_prms": dict(
            criterion=nn.L1Loss(),
            opt="adam",
            opt_params=dict(lr=0.001, weight_decay=1e-4),
            sch_params=None, #{'max_lr': lr, 'pct_start':0.1, 'div_factor':5, 'final_div_factor': 10000}, # initial_lr = max_lr/div_factor, min_lr = initial_lr/final_div_factor
            epochs=2, 
            prefix="test_",
            save_best=False,
            maximize_score=False,
        ), 
        "nn_prms": dict(
            n_neighbors=1,
        ),
        "seq_len": 80,
        "use_seq_len": 32, 
        "train_batch_size": 8,
        "pred_batch_size": 8,
    }

    model = Model(**model_prms)
    model.fit(tr_x, tr_y, tr_x, tr_y)

    pred = model.predict(tr_x)


def test2():
    tr_file = "input/ventilator-pressure-prediction/train.csv"
    tr_df = pd.read_csv(tr_file)
    tr_df = tr_df.iloc[:80*5]

    tr_df["shuffled_breath_id"] = make_shuffled_id(tr_df["breath_id"])
    tr_df.to_csv("test_tr_df.csv")


def test3():
    oof_df = pd.read_csv("experiments/exp_v00_00_00/oof.csv")
    ev = Evaluation(oof_df)
    score = ev.calc_scores()
    print(score)

    breath_mae = ev.calc_breath_mae()
    breath_mae = breath_mae.sort_values("error", ascending=False)
    print(breath_mae)

    ev.plot_breath(breath_mae.index[0])
    ev.plot_breath(breath_mae.index[1])
    ev.plot_breath(breath_mae.index[2])


def test4():
    tr_file = "input/ventilator-pressure-prediction/train.csv"
    tr_df = pd.read_csv(tr_file)
    tr_df = tr_df.iloc[:80*100]

    ft = Feature()
    tr_x = ft.transform(tr_df)
    tg = Target()
    tr_y = tg.transform(tr_df)

    nn_params = dict(
        n_neighbors=3,
    )
    nrnb = VentilatorNearestNeighbors(nn_params)

    tr_x_seq = tr_x[["x_u_in"]].values.reshape([-1, 80]).copy()
    tr_y_seq = tr_y.values.reshape([-1, 80, 1]).copy()

    nrnb.fit(tr_x_seq, tr_y_seq)
    nrnb.average_kneighbors(tr_x_seq)



if __name__ == "__main__":
    test1()
    #test2()
    #test3()
    #test4()
