import pathlib

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
    from torch.utils import _pytree as tree
except ImportError as err:
    raise ImportError('This require pytorch.') from err


class TactileDataset(Dataset):

    def __init__(self, dataset_dir: str) -> None:
        """Assuming a dataset is small enough to fit in RAM."""
        # Each of num_workers>0 creates a copy of data in a such way.
        ds_dir = pathlib.Path(dataset_dir)
        meta = np.load(ds_dir / 'config.npz')
        items = [dict(np.load(item)) for item in (ds_dir / 'items/').iterdir()]
        self.meta = tree.tree_map(lambda x: x.item(), dict(meta))
        self.items = tree.tree_map(np.stack, items)

    def __len__(self) -> int:
        return len(tree.tree_flatten(self.items)[0])

    def __getitem__(self, idx) -> tree.PyTree:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = tree.tree_map(np.s_[idx], self.items)
        return tree.tree_map(torch.from_numpy, data)
