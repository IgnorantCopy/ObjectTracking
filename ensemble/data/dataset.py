import torch

from ensemble.rd.data.preprocess import *
from ensemble.rd.data.dataset import RDMap
from ensemble.track.data.preprocessor import TrajectoryPreprocessor


class FusedDataset(RDMap):
    def __init__(self, batch_files: list[BatchFile], image_transform=None, track_transform=None,
                 image_seq_len=180, track_seq_len=29):
        super().__init__(batch_files, image_transform, track_transform, image_seq_len, track_seq_len)

    def __getitem__(self, item):
        # load rd map
        batch_file, point_index, images, extra_features, missing_rate, image_mask, cls = super().__getitem__(item)
        point_file = batch_file.point_file
        track_file = batch_file.track_file

        # load point and track data
        merged_data = self._load_track_data(point_file, track_file)
        if len(merged_data) == 0:
            return batch_file, None, None, None, None, None, None, cls
        if merged_data.dtype != np.float32:
            merged_data = merged_data.astype(np.float32)
        if self.track_transform:
            merged_data = self.track_transform(merged_data)

        return batch_file, point_index, images, merged_data, extra_features, missing_rate, image_mask, cls

    def _load_track_data(self, point_filepath, track_filepath):
        """
        使用 polars 高效加载、合并数据并进行特征工程
        :param point_filepath: 点迹文件路径
        :param track_filepath: 航迹文件路径
        :return: 合并和处理后的特征数据 (NumPy Array)
        """
        preprocessor = TrajectoryPreprocessor(seq_len=self.track_seq_len)
        preprocessed_data, _, _ = preprocessor.process_single_trajectory(point_filepath, track_filepath)

        return preprocessed_data

    @staticmethod
    def collate_fn(batch):
        batch_files, point_indices, stacked_images, stacked_track_features, stacked_extra_features, \
            stacked_missing_rate, image_masks, labels = [], [], [], [], [], [], [], []
        for (batch_file, point_index, images, track_features, extra_features, missing_rate, image_mask, cls) in batch:
            if images is None:
                continue
            batch_files.append(batch_file)
            point_indices.append(point_index)
            stacked_images.append(images)
            stacked_track_features.append(track_features)
            stacked_extra_features.append(extra_features)
            stacked_missing_rate.append(missing_rate)
            image_masks.append(image_mask)
            labels.append(cls)
        point_indices = torch.from_numpy(np.stack(point_indices, axis=0))
        stacked_images = torch.from_numpy(np.stack(stacked_images, axis=0))
        stacked_track_features = torch.from_numpy(np.stack(stacked_track_features, axis=0))
        stacked_extra_features = torch.from_numpy(np.stack(stacked_extra_features, axis=0))
        stacked_missing_rate = torch.from_numpy(np.stack(stacked_missing_rate, axis=0))
        image_masks = torch.from_numpy(np.stack(image_masks, axis=0))
        labels = torch.tensor(labels, dtype=torch.long)
        return batch_files, point_indices, stacked_images, stacked_track_features, stacked_extra_features, \
                stacked_missing_rate, image_masks, labels


if __name__ == '__main__':
    from tqdm import tqdm
    import glob
    import re
    import plotly.graph_objects as go

    data_root = "D:/DataSets/挑战杯_揭榜挂帅_CQ-08赛题_数据集"
    point_files = glob.glob(os.path.join(data_root, "点迹/PointTracks_*.txt"))
    missing = {i: [] for i in range(4)}
    for point_file in tqdm(point_files):
        match_result = re.match(r"PointTracks_(\d+)_(\d+)_(\d+).txt", os.path.basename(point_file))
        batch_id = int(match_result.group(1))
        label = int(match_result.group(2))
        if label > 4:
            continue
        num_points = int(match_result.group(3))
        raw_file = os.path.join(data_root, f"原始回波/{batch_id}_Label_{label}.dat")
        track_file = os.path.join(data_root, f"航迹/Tracks_{batch_id}_{label}_{num_points}.txt")
        batch_file = BatchFile(batch_id, label, raw_file, point_file, track_file)
        rd_matrices, ranges, velocities, missing_rates = process_batch(batch_file)
        if rd_matrices is None or len(rd_matrices) == 0:
            missing[label-1].append(1)
        else:
            missing[label-1].append(missing_rates[-1])
    for k, v in missing.items():
        fig = go.Figure(data=[go.Histogram(x=v, histnorm='probability', nbinsx=30)])
        fig.update_layout(
            title_text="缺失率分布",
            xaxis_title_text="缺失率",
            yaxis_title_text="占比",
            bargap=0.2,
            bargroupgap=0.1
        )
        fig.show()