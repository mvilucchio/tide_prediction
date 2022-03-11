import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os


class TimeSeries(Dataset):
    def __init__(self, x, y, transform_x=None, transform_y=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]
        self.transfrom_x = transform_x
        self.transfrom_y = transform_y

    def __getitem__(self, idx):
        out = []
        if self.transfrom_x:
            out.append(self.transform(self.x[idx]))
        else:
            out.append(self.x[idx])

        if self.transfrom_y:
            out.append(self.transform_y(self.y[idx]))
        else:
            out.append(self.y[idx])

        return out

    def __len__(self):
        return self.len


class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype("float").reshape(-1, 2)
        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


# dataloader
# train_loader = DataLoader(dataset, shuffle=True,batch_size=256)
