from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(args, flag):
    # 'ETTm1': Dataset_ETT_minute
    Data = data_dict[args.data]
    # timeF:timeenc = 1
    timeenc = 0 if args.embed != 'timeF' else 1

    # step1:按照训练测试验证设置参数
    # 测试阶段
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        # 思考为什么测试阶段不同任务的batch不同？
        # 异常检测和分类任务,batch=32
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        # 长期预测,短期预测和填补任务,batch=1(可能是因为后面的要依赖前面的?但是对于异常检测和分类任务只需要对一个样本给出一个预测结果)
        else:
            batch_size = 1  # bsz=1 for evaluation
        # freq="h",表示小时级别的频率
        freq = args.freq

    # 训练阶段&验证阶段:batch统一为32
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    # step2:按照任务设置参数
    # 并且在这一步直接创建Dataset与Dataloader

    # 异常检测任务
    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    # 时序分类任务
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader

    # 短期预测,长期预测和填补任务
    else:
        # 对于m4数据集需要特殊处理drop_last = False
        if args.data == 'm4':
            drop_last = False
        # 对于m4以外的数据集
        # 构建dataset
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        # 构建dataloader
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
