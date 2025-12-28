import os
from PIL import Image
import importlib
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import torch

import core.util as Util

class InfoLogger():
    """
    use logging to record log, only work on GPU 0 by judging global_rank
    """
    def __init__(self, opt):
        self.opt = opt
        self.rank = opt['global_rank']
        self.phase = opt['phase']

        self.setup_logger(None, opt['path']['experiments_root'], opt['phase'], level=logging.INFO, screen=False)
        self.logger = logging.getLogger(opt['phase'])
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):
        if self.rank != 0: # info only print on GPU 0.
            def wrapper(info, *args, **kwargs):
                pass
            return wrapper
        if name in self.infologger_ftns:
            print_info = getattr(self.logger, name, None)
            def wrapper(info, *args, **kwargs):
                print_info(info, *args, **kwargs)
            return wrapper
    
    @staticmethod
    def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
        """ set up logger """
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='a+')
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)
        if screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            l.addHandler(sh)

class VisualWriter():
    """ 
    use tensorboard to record visuals, support 'add_scalar', 'add_scalars', 'add_image', 'add_images', etc. funtion.
    Also integrated with save results function.
    """
    def __init__(self, opt, logger):
        log_dir = opt['path']['tb_logger']
        self.result_dir = opt['path']['results']
        enabled = opt['train']['tensorboard']
        self.rank = opt['global_rank']

        self.writer = None
        self.selected_module = ""

        if enabled and self.rank==0:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["tensorboardX", "torch.utils.tensorboard"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.epoch = 0
        self.iter = 0
        self.phase = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.custom_ftns = {'close'}
        self.timer = datetime.now()

    def set_iter(self, epoch, iter, phase='train'):
        self.phase = phase
        self.epoch = epoch
        self.iter = iter

    def save_images(self, results):
        result_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(result_path, exist_ok=True)
        result_path = os.path.join(result_path, str(self.epoch))
        os.makedirs(result_path, exist_ok=True)

        ''' get names and corresponding images from results[OrderedDict] '''
        try:
            names = results['name']
            outputs = results['result']
            
            # Check if we're dealing with 3D volumes (5D tensors: B, C, D, H, W)
            # or 2D images (4D tensors: B, C, H, W)
            is_3d = False
            if len(outputs) > 0:
                first_output = outputs[0]
                if isinstance(first_output, torch.Tensor) and first_output.dim() == 5:
                    is_3d = True
            
            if is_3d:
                # Save 3D volumes as NIfTI files
                import nibabel as nib
                for i in range(len(names)):
                    volume = outputs[i]
                    # volume shape: (C, D, H, W) or (1, D, H, W)
                    if volume.dim() == 5:
                        volume = volume[0]  # Remove batch dimension if present
                    if volume.dim() == 4:
                        volume = volume[0]  # Remove channel dimension: (D, H, W)
                    
                    # Convert to numpy and permute back to (X, Y, Z) = (width, height, depth)
                    volume_np = volume.numpy()
                    # Permute from (D, H, W) = (Z, Y, X) back to (X, Y, Z)
                    volume_np = volume_np.transpose(2, 1, 0)  # (Z, Y, X) -> (X, Y, Z)
                    
                    # Create NIfTI image
                    nii_img = nib.Nifti1Image(volume_np, affine=np.eye(4))
                    # Save with .nii.gz extension if not already present
                    save_name = names[i]
                    if not save_name.endswith('.nii.gz') and not save_name.endswith('.nii'):
                        save_name = save_name + '.nii.gz'
                    nib.save(nii_img, os.path.join(result_path, save_name))
            else:
                # Save 2D images as before
                outputs = Util.postprocess(outputs)
                for i in range(len(names)): 
                    Image.fromarray(outputs[i]).save(os.path.join(result_path, names[i]))
        except Exception as e:
            raise NotImplementedError(f'Error saving results: {str(e)}. You must specify the context of name and result in save_current_results functions of model.')

    def close(self):
        self.writer.close()
        print('Close the Tensorboard SummaryWriter.')

        
    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add phase(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(self.phase, tag)
                    add_data(tag, data, self.iter, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


class LogTracker:
    """
    record training numerical indicators.
    """
    def __init__(self, *keys, phase='train'):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return {'{}/{}'.format(self.phase, k):v for k, v in dict(self._data.average).items()}
