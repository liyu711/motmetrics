import pandas as pd
import motmetrics as mm
import numpy as np
import os


def load_sequence_information(sequences, ground_truth_dir, hypothesis_dir):
    result = []
    for sequence in sequences:
        ground_truth = os.path.join(ground_truth_dir, sequence)
        hypothesis = os.path.join(hypothesis_dir, sequence)
        gt = pd.read_csv(ground_truth, header=None, sep=',')
        gt = filter_out_small_bbox(gt)
        hp = pd.read_csv(hypothesis, header=None, sep=",")
        hp = filter_out_small_bbox(hp)
        acc = mm.MOTAccumulator(auto_id=True)
        frames = gt[0].unique()
        for frame in frames:
            a = []
            b = []
            gt_frame = gt.loc[gt[0] == frame]
            hp_frame = hp.loc[hp[0] == frame]
            for row in gt_frame.iterrows():
                box = row[1][2:6].values
                a.append(box)
            a = np.array(a)

            for row in hp_frame.iterrows():
                box = row[1][2:6].values
                b.append(box)
            b = np.array(b)

            obj_gt = gt_frame[1].values
            obj_hp = hp_frame[1].values

            C = mm.distances.iou_matrix(a, b, max_iou=0.5)
            acc.update(
                obj_gt,
                obj_hp,
                C
            )
        result.append(acc)
    return result


def filter_out_small_bbox(df):
    # result_df = df.loc[(df[4] >= 30) & (df[5] >= 30)]
    result_df = df[df[4] * df[5] >= 2500]
    # result_df = result_df[(result_df[4] >= 30) & (result_df[5] >= 30)]
    return result_df



if __name__ == '__main__':
    ground_truth_dir = "cruw_test_gt"
    hypothesis_dir = "output1"
    sequence = os.listdir(hypothesis_dir)
    accs = load_sequence_information(sequence, ground_truth_dir, hypothesis_dir)

    mh = mm.metrics.create()
    summary = mh.compute(accs[0], metrics=['num_frames', 'mota', 'motp'], name='acc')
    print(summary)

    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        names=sequence,
        generate_overall=True
    )

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


