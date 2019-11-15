from dataset.SyntheticDataset import SyntheticData
from dataset.DialogueStateDataset import DialogueStateDataset
from dataset.SST2Dataset import SST2Dataset
import ipdb

def get_dataset(args):
    dataset = None
    if args.dataset == 'sentiments':
        dataset = SST2Dataset(data_dir=args.data_dir)
    elif args.dataset == 'dialogue_states':
        dataset = DialogueStateDataset(data_dir=args.data_dir)
    elif args.dataset == 'synthetic_2d':
        dataset = SyntheticData(num_classes=args.output_dim,
                                num_elements_per_class=100,
                                std=5/3.0,
                                seed=0)
    return dataset
