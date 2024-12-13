# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

#TODO clean here, I want to use mmdet code without importint the library :> import mmdet


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = ":)"
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
