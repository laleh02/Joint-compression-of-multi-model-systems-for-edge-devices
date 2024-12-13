from os import path as osp
import json
import torch
import random
import logging
import math
from tqdm import tqdm
import warnings
import copy
import time


from torch.optim import lr_scheduler
import mmcv
from mmcv.utils import collect_env
from mmcv import Config
from torch.utils.tensorboard import SummaryWriter
from .config_handler import dict2str, validate_yunet_train_config
from .experiments import make_exp_dirs, mkdir_and_rename, get_time_str, get_env_info, get_root_logger
from .data import create_dataloader, Restoration_Dataset, Detection_Dataset
from .losses import PSNRLoss
from .setup_env import setup_multi_processes, update_data_root
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

##170 degrees FOV, 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate_configs(config):
    """Validates config file with respect to obligatory fields"""

    seed = config.get('seed')
    if seed is None:
        config['seed'] = random.randint(1,10000)

    mandatory_configs = ['datasets', 'train']

    assert all(param in config for param in mandatory_configs)
    print(config['datasets']['train'])
    num_epochs = config['train'].get('num_epochs')
    if num_epochs is None:
        config['train']['num_epochs'] = math.ceil(config['train']['total_iters'] / config['datasets']['train']['dataset_size'])

    return config

def init_loggers(config):
    log_file = osp.join(config['path']['log'],
                        f"train_{config['experiment_name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(config))

    tb_logger = None
    if config['logger'].get('use_tb_logger') and 'debug' not in config['experiment_name']:
        # tb_logger = init_tb_logger(log_dir=f'./logs/{config['experiment_name']}') #mkdir logs @CLY
        tb_logger = SummaryWriter(log_dir=osp.join('logs', config['experiment_name']))
    return logger, tb_logger


def check_resume(config, resume_iter):
    """Check resume states and pretrain_network paths.

    Args:
        config (dict): configions.
        resume_iter (int): Resume iteration.
    """
    logger = get_root_logger()
    if config['path']['resume_state']:
        # get all the networks
        networks = [key for key in config.keys() if key.startswith('network_')]
        flag_pretrain = False
        for network in networks:
            if config['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True
        if flag_pretrain:
            logger.warning(
                'pretrain_network path will be ignored during resuming.')
        # set pretrained model paths
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            if config['path'].get('ignore_resume_networks') is None or (
                    basename not in config['path']['ignore_resume_networks']):
                config['path'][name] = osp.join(
                    config['path']['models'], f'net_{basename}_{resume_iter}.pth')
                logger.info(f"Set {name} to {config['path'][name]}")

def setup_optimizers(model, train_opt):
    optimizers = []
    optim_params = []

    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
        else:
            logger = get_root_logger()
            logger.warning(f'Params {k} will not be optimized.')
    optim_type = train_opt['optimizer'].pop('type')
    if optim_type == 'Adam':
        optimizer_g = torch.optim.Adam(optim_params, **train_opt['optimizer'])
    elif optim_type == 'AdamW':
        optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optimizer'])
    else:
        raise NotImplementedError(
            f'optimizer {optim_type} is not supported yet.')
    optimizers.append(optimizer_g)
    return optimizers

def setup_schedulers(train_opt, optimizers, train_loader):
    n_batch = len(train_loader)

    """Set up schedulers."""
    schedulers = []
    scheduler_type = train_opt['scheduler'].pop('type')
    if scheduler_type in ['MultiStepLR']:
        for optimizer in optimizers:
            schedulers.append(
                torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                **train_opt['scheduler']))
    elif scheduler_type == 'CosineAnnealingLR':
        print('..', 'cosineannealingLR')
        for optimizer in optimizers:
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_batch * train_opt['num_epochs']))
    elif scheduler_type == 'LinearLR':
        for optimizer in optimizers:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, train_opt['total_iter']))
    else:
        raise NotImplementedError(
            f'Scheduler {scheduler_type} is not implemented yet.')
    
    return schedulers

def resume_training(self, resume_state):
    """Reload the optimizers and schedulers for resumed training.

    Args:
        resume_state (dict): Resume state.
    """
    resume_optimizers = resume_state['optimizers']
    resume_schedulers = resume_state['schedulers']
    assert len(resume_optimizers) == len(
        self.optimizers), 'Wrong lengths of optimizers'
    assert len(resume_schedulers) == len(
        self.schedulers), 'Wrong lengths of schedulers'
    for i, o in enumerate(resume_optimizers):
        self.optimizers[i].load_state_dict(o)
    for i, s in enumerate(resume_schedulers):
        self.schedulers[i].load_state_dict(s)

def train(model, train_config, results_dir, device
          ):
    """
    model : 
    train_config : {
        name : model/experiment name. 
        task : 'Restoration' or 'Detection'. Defines losses and training data schema. 
        seed : 
        path : path for model weights if pretrained/resuming training. Can contain 'resume_state' or pretrain_network
        logger : config for which loggers are used ('use_tb_logger', 'use_wb_logger'). For the moment only TensorBoard is availble.
        total_iters : 
        dataset_size : 
        batch_size :
        debug : 
        train : contains 'scheduler' and 'optimizer', which are dicts with key 'type' (corresponding to the torch definition) and kwargs.
    }"""
    model.train()
    train_config = validate_configs(train_config)
    generator = torch.Generator()
    generator.manual_seed(train_config['seed'])

    #From BasicSR
    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(train_config['experiment_name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    # load resume states if necessary
    if train_config['path'].get('resume_state') and torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            train_config['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    elif train_config['path'].get('resume_state'):
        resume_state = torch.load(
            train_config['path']['resume_state'],
            )
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(train_config)
        if train_config['logger'].get('use_tb_logger') and 'debug' not in train_config:
            mkdir_and_rename(osp.join('tb_logger', train_config['experiment_name']))

    # initialize loggers
    logger, tb_logger = init_loggers(train_config)

    # initialize train and validation dataloaders
    if train_config.get("task") == "Restoration":
        train_dataset = Restoration_Dataset(train_config['datasets']['train'])
        #val_dataset = Detection_Dataset(train_config['datasets']['calibration'])

        criterion = PSNRLoss()
        train_loader = create_dataloader(train_dataset, train_config)
        #val_loader = create_dataloader(val_dataset, train_config)
        train_restorator(model, train_dataset, train_config, device, logger, train_loader, criterion)


    elif train_config.get("task") == "Detection":

        cfg = Config.fromfile(train_config['yunet_config'])

        # update data root according to MMDET_DATASETS
        update_data_root(cfg)

        if train_config.get('cfg_options',None) is not None:
            cfg.merge_from_dict(train_config['cfg_options'])

        if train_config.get('auto_scale_lr',None) is not None:
            if 'auto_scale_lr' in cfg and \
                    'enable' in cfg.auto_scale_lr and \
                    'base_batch_size' in cfg.auto_scale_lr:
                cfg.auto_scale_lr.enable = True
            else:
                warnings.warn('Can not find "auto_scale_lr" or '
                            '"auto_scale_lr.enable" or '
                            '"auto_scale_lr.base_batch_size" in your'
                            ' configuration file. Please update all the '
                            'configuration files to mmdet >= 2.24.1.')

        # set multi-process settings
        setup_multi_processes(cfg)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # work_dir is determined by qat.py
        cfg.work_dir = results_dir
    
        if train_config.get('resume_from',None) is not None:
            cfg.resume_from = train_config['resume_from']
        cfg.auto_resume = train_config.get('auto_resume',None)
        if train_config.get('gpus',None) is not None:
            cfg.gpu_ids = range(1)
            warnings.warn('`--gpus` is deprecated because we only support '
                        'single GPU mode in non-distributed training. '
                        'Use `gpus=1` now.')
        if train_config.get('gpu_ids',None) is not None:
            cfg.gpu_ids = train_config['gpu_ids']
            warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                        'Because we only support single GPU mode in '
                        'non-distributed training. Use the first GPU '
                        'in `gpu_ids` now.')
        if train_config.get('gpus',None) is None and train_config.get('gpu_ids',None) is None:
            train_config['gpu_ids'] = [0]
            cfg.gpu_ids = train_config['gpu_ids']

        # init distributed env first, since logger depends on the dist info.

        distributed = False
        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(train_config['yunet_config'])))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
        meta['config'] = cfg.pretty_text
        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        cfg.device = device
        # set random seeds
        seed = init_random_seed(train_config['seed'], device=cfg.device)
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {train_config.get("deterministic",False)}')
        set_random_seed(seed, deterministic=train_config.get("deterministic",False))
        cfg.seed = seed
        meta['seed'] = seed
        meta['exp_name'] = osp.basename(train_config['experiment_name'])

        # model = build_detector(
        #     cfg.model,
        #     train_cfg=cfg.get('train_cfg'),
        #     test_cfg=cfg.get('test_cfg'))
        # model.init_weights()

        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))

        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=True,
            timestamp=timestamp,
            meta=meta)
    else:
        raise "Task not implemented : config['task'] in ['Restoration','Detection']"


    # #TODO: move resume quantization logic outside of training method in order to have a quantized checkpoint. 
    #In short, reimplement create_model to check if there is a checkpoint where stated, check it is a quantized model and load it.
    # # create model
    # if resume_state:  # resume training
    #     check_resume(train_config, resume_state['iter'])
    #     model = create_model(train_config,resume_state)
    #     resume_training(resume_state)  # handle optimizers and schedulers
    #     logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
    #                 f"iter: {resume_state['iter']}.")
    #     start_epoch = resume_state['epoch']
    #     current_iter = resume_state['iter']
    # else:
    #     model = create_model(train_config)
    #     start_epoch = 0
    #     current_iter = 0


def train_restorator(model, train_dataset, train_config, device, logger, train_loader, criterion):
    logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_dataset)}'
                f"\n\tTotal epochs: {train_config['train']['num_epochs']}; iters: {train_config['train']['total_iters']}.")

    optimizers = setup_optimizers(model, train_config['train'])
    if train_config['train'].get('scheduler') is not None:
        schedulers = setup_schedulers(train_config['train'], optimizers, train_loader)
    else:
        schedulers = None
    start_epoch = 0 #TODO look at logic of resuming state and adapt start_epoch
    current_iter = 0    
    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    
    # for epoch in range(start_epoch, total_epochs + 1):
    for epoch in range(start_epoch, train_config['train']['num_epochs']):
        for current_iter, data in tqdm(enumerate(train_loader),total=len(train_loader)):
            start = time.time()
            model.train()
            loss = train_batch(model, data, criterion, optimizers)

            if epoch % train_config['logger']['print_freq'] == 0:
                logger.info(
                        f'Epoch: {epoch}, '
                        f'Loss: {loss}, '
                        f'device: {device}, ' 
                        f'iter time: {str((time.time() - start))}, '
                        f'batch_size: {data["lq"].shape[0]}, '
                        f'learning rate: {schedulers[0].get_last_lr()}, ')
            [scheduler.step() for scheduler in schedulers]
        # end of iter

    # end of epoch

def train_batch(model, data, criterion, optimizers):
    [optimizer.zero_grad()  for optimizer in optimizers] 

    lq = data['lq'].to(device)
    gt = data['gt'].to(device)
    preds = model(lq)
    loss = criterion(preds,gt)
    loss.backward()
    [optimizer.step() for optimizer in optimizers]        

    return loss