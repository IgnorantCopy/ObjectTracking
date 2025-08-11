import torch

from ensemble.rd.data.preprocess import *
from ensemble.rd.data.dataset import RDMap
from ensemble.track.data.preprocessor import TrajectoryPreprocessor


class FusedDataset(RDMap):
    def __init__(self, batch_files: list[BatchFile], image_transform=None, track_transform=None,
                 image_seq_len=180, track_seq_len=29, test=False):
        super().__init__(batch_files, image_transform, track_transform, image_seq_len, track_seq_len)
        self.test = test

    def __getitem__(self, item):
        # load rd map
        batch_file, point_index, images, extra_features, missing_rate, image_mask, cls = super().__getitem__(item)
        point_file = batch_file.point_file
        track_file = batch_file.track_file

        # load point and track data
        merged_data, num_points = self._load_track_data(point_file, track_file)
        if len(merged_data) == 0:
            return batch_file, None, None, None, None, None, None, None, cls
        if merged_data.dtype != np.float32:
            merged_data = merged_data.astype(np.float32)
        if self.track_transform:
            merged_data = self.track_transform(merged_data)

        return batch_file, point_index, images, merged_data, num_points, extra_features, missing_rate, image_mask, cls

    def _load_track_data(self, point_filepath, track_filepath):
        """
        使用 polars 高效加载、合并数据并进行特征工程
        :param point_filepath: 点迹文件路径
        :param track_filepath: 航迹文件路径
        :return: 合并和处理后的特征数据 (NumPy Array)
        """
        preprocessor = TrajectoryPreprocessor(seq_len=self.track_seq_len, test=self.test)
        preprocessed_data, _, _ = preprocessor.process_single_trajectory(point_filepath, track_filepath)
        num_points = os.path.basename(point_filepath).split('_')[-1].split('.')[0]
        return preprocessed_data, int(num_points)

    @staticmethod
    def collate_fn(batch):
        batch_files, point_indices, stacked_images, stacked_track_features, stacked_num_points, \
            stacked_extra_features, stacked_missing_rate, image_masks, labels, fail = [], [], [], [], [], [], [], [], [], []
        for (batch_file, point_index, images, track_features, num_points, extra_features, missing_rate, image_mask, cls) in batch:
            batch_files.append(batch_file)
            stacked_track_features.append(track_features)
            stacked_num_points.append(num_points)
            labels.append(cls)
            if images is None:
                fail.append(True)
                continue
            fail.append(False)
            point_indices.append(point_index)
            stacked_images.append(images)
            stacked_extra_features.append(extra_features)
            stacked_missing_rate.append(missing_rate)
            image_masks.append(image_mask)
        point_indices = torch.from_numpy(np.stack(point_indices, axis=0))
        stacked_images = torch.from_numpy(np.stack(stacked_images, axis=0))
        stacked_track_features = torch.from_numpy(np.stack(stacked_track_features, axis=0))
        stacked_num_points = torch.tensor(stacked_num_points, dtype=torch.int)
        stacked_extra_features = torch.from_numpy(np.stack(stacked_extra_features, axis=0))
        stacked_missing_rate = torch.from_numpy(np.stack(stacked_missing_rate, axis=0))
        image_masks = torch.from_numpy(np.stack(image_masks, axis=0))
        labels = torch.tensor(labels, dtype=torch.long)
        fail = torch.tensor(fail, dtype=torch.bool)
        return {
            "batch_files": batch_files,
            "point_indices": point_indices,
            "images": stacked_images,
            "track_features": stacked_track_features,
            "num_points": stacked_num_points,
            "extra_features": stacked_extra_features,
            "missing_rate": stacked_missing_rate,
            "image_masks": image_masks,
            "labels": labels,
            "fail": fail
        }


if __name__ == '__main__':
    import shutil
    from tqdm import tqdm
    from ensemble.rd.data.dataset import split_train_val

    data_root = r"D:\DataSets\挑战杯_揭榜挂帅_CQ-08赛题_数据集"
    _, val_batch_files = split_train_val(data_root, 4)
    save_path = os.path.join(data_root, "val_set")
    point_dir = os.path.join(save_path, "点迹")
    track_dir = os.path.join(save_path, "航迹")
    raw_dir = os.path.join(save_path, "原始回波")
    os.makedirs(point_dir, exist_ok=True)
    os.makedirs(track_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    for batch_file in tqdm(val_batch_files):
        point_file = batch_file.point_file
        track_file = batch_file.track_file
        raw_file = batch_file.raw_file
        match_result = re.match(r"PointTracks_(\d+)_(\d+)_(\d+)\.txt", os.path.basename(point_file))
        batch_id = int(match_result.group(1))
        num_points = int(match_result.group(3))
        shutil.copy(point_file, os.path.join(point_dir, f"PointTracks_{batch_id}_{num_points}.txt"))
        shutil.copy(track_file, os.path.join(track_dir, f"Tracks_{batch_id}_{num_points}.txt"))
        shutil.copy(raw_file, os.path.join(raw_dir, f"{batch_id}.dat"))