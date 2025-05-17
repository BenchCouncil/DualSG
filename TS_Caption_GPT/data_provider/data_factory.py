from data_provider.data_loader import TimeCaptionDataset
from torch.utils.data import DataLoader

data_dict = {
    'timecaption': TimeCaptionDataset
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        data_path=args.data_path,
        scale=args.scale  # 假设 args 中有 scale 参数来控制是否归一化
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True, persistent_workers=True
    )
    return data_set, data_loader
