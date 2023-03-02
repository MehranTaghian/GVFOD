import os
from multiprocessing import Pool

import click

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression as LR


@click.command()
@click.argument("src", nargs=1)
@click.argument("dst", nargs=1)
@click.argument("failures", nargs=-1)
def main(src, dst, failures):
    p = Pool()
    for file in os.listdir(src):
        if file.endswith(".json"):
            p.apply_async(evaluate_results, [os.path.join(src, file), dst,
                                             [precision_score, recall_score, f1_score], failures])

    p.close()
    p.join()


def evaluate_results(json_filepath, dst, metrics, failures):
    res = pd.read_json(json_filepath)
    res["Class"] = res["Algorithm"].map(
        lambda alg: "Temporal" if alg in ["GVFOD", "MarkovChain", "HMM"] else "Multivariate")

    outlier_names = parse_outlier_names_from_columns(res.columns)

    for abn_str in outlier_names[1:]:
        if (abn_str not in failures) and (len(failures) > 0):
            continue
        for metric in metrics:
            nor_str = outlier_names[0]
            metric_name = metric.__name__
            res[f"{metric_name}_{abn_str}"] = score(metric,
                                                    res["c_" + nor_str],
                                                    res["ic_" + nor_str],
                                                    res["c_" + abn_str],
                                                    res["ic_" + abn_str])

            # Here it just wants to calculate the f1-score at the 1000 training size
            if metric_name == "f1_score" and abn_str == "loose_l1":
                selection = res.loc[
                    (res["Algorithm"] == "GVFOD") & (res["Training Size"].isin([973, 1076])),
                    ["Training Size", f"{metric_name}_{abn_str}"]]

                model = LR()
                model.fit(selection["Training Size"].values.reshape(-1, 1),
                          selection[f"{metric_name}_{abn_str}"].values.reshape(-1, 1))

                print(f"F1 score @ 1000, {os.path.split(json_filepath)[1]}: {model.predict(np.array([[1000]]))}")

        res.to_csv(os.path.join(dst, 'res.csv'))

    return res


def parse_outlier_names_from_columns(columns):
    outlier_names = []
    for cname in columns:
        if cname.startswith("c_"):
            outlier_names.append("_".join(cname.split("_")[1:]))
    return outlier_names


def score(metric, true_negative, false_positive, true_positive, false_negative):
    scores = []
    for tn, fp, tp, fn in zip(true_negative, false_positive, true_positive, false_negative):
        true_vec, pred_vec = _create_true_pred_vec_from_count(tn, fp, tp, fn)
        scores.append(metric(true_vec, pred_vec))

    return scores


def _create_true_pred_vec_from_count(true_negative, false_positive, true_positive, false_negative):
    n_neg_true = true_negative + false_positive
    n_pos_true = true_positive + false_negative

    true_vec = np.concatenate([np.zeros(n_neg_true), np.ones(n_pos_true)]).astype(int)
    pred_vec = np.concatenate([np.zeros(true_negative),
                               np.ones(false_positive),
                               np.zeros(false_negative),
                               np.ones(true_positive)]).astype(int)

    return true_vec, pred_vec


if __name__ == "__main__":
    main()
