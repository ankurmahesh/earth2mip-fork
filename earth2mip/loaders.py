import torch
from typing import Protocol
from earth2mip import schema
from earth2mip.networks import swin, swin_residual
from types import SimpleNamespace
import numpy as np
from ruamel.yaml import YAML
from modulus.utils.zenith_angle import cos_zenith_angle

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ruamel.yaml import YAML
import json


class ParamsBase:
    """Convenience wrapper around a dictionary

    Allows referring to dictionary items as attributes, and tracking which
    attributes are modified.
    """

    def __init__(self):
        self._original_attrs = None
        self.params = {}
        self._original_attrs = list(self.__dict__)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return self.params.get(key, default)

    def to_dict(self):
        new_attrs = {key: val for key, val in vars(self).items() if key not in self._original_attrs}
        return {**self.params, **new_attrs}

    @staticmethod
    def from_json(path: str) -> "ParamsBase":
        with open(path) as f:
            c = json.load(f)
        params = ParamsBase()
        params.update_params(c)
        return params

    def update_params(self, config):
        for key, val in config.items():
            if val == "None":
                val = None
            self.params[key] = val
            self.__setattr__(key, val)


class YParams(ParamsBase):
    def __init__(self, yaml_filename, config_name, print_params=False):
        """Open parameters stored with ``config_name`` in the yaml file ``yaml_filename``"""
        super().__init__()
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        if print_params:
            print("------------------ Configuration ------------------")

        with open(yaml_filename) as _file:
            d = YAML().load(_file)[config_name]

        self.update_params(d)

        if print_params:
            for key, val in d.items():
                print(key, val)
            print("---------------------------------------------------")

    def log(self, logger):
        logger.info("------------------ Configuration ------------------")
        logger.info("Configuration file: " + str(self._yaml_filename))
        logger.info("Configuration name: " + str(self._config_name))
        for key, val in self.to_dict().items():
            logger.info(str(key) + " " + str(val))
        logger.info("---------------------------------------------------")


class SwinWrapper(torch.nn.Module):
    """Makes sure the parameter names are the same as the checkpoint"""

    def __init__(self, module):
        super().__init__()
        self.module = module
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)

    def forward(self, x):
        """x: (batch, history, channel, x, y)"""
        return self.module(x)

class LoaderProtocol(Protocol):
    def __call__(self, package, pretrained=True) -> None:
        return


def pickle(package, pretrained=True):
    """
    load a checkpoint into a model
    """
    assert pretrained
    p = package.get("weights.tar")
    return torch.load(p)


def torchscript(package, pretrained=True):
    """
    load a checkpoint into a model
    """
    p = package.get("scripted_model.pt")
    import json

    config = package.get("config.json")
    with open(config) as f:
        config = json.load(f)

    model = torch.jit.load(p)
    if config["nettype"] == 'swin':
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
    if config["add_zenith"]:
        import numpy as np

        from earth2mip.networks import CosZenWrapper

        lat = 90 - np.arange(721) * 0.25
        if package.metadata().grid == schema.Grid.grid_720x1440:
            lat = lat[:-1]
        lon = np.arange(1440) * 0.25
        model = CosZenWrapper(model, lon, lat)
    else:
        from earth2mip.networks import Wrapper
        model = Wrapper(model)


    return model


#def sfno(package, pretrained: bool = True) -> torch.nn.Module:
#    """Load SFNO model from checkpoints trained with era5_wind"""       
#    from earth2mip.networks import Wrapper
#    try:
#        path = package.get("config.json")                                   
#        params = ParamsBase.from_json(path)                                 
#    except:
#        path = package.get("hyperparams.yaml")
#        params = _load_params(path)
#    params.checkpointing = False
#    model = sfnonet.SphericalFourierNeuralOperatorNet(params)           
#                                                                        
#    if pretrained:                                                      
#        weights = package.get("weights.tar")                            
#        checkpoint = torch.load(weights)                                
#        load_me = Wrapper(model)                                        
#        state = checkpoint["model_state"]                               
#        state = _fix_state_dict_keys(state)
#        state = {"module.device_buffer": model.device_buffer, **state}  
#        load_me.load_state_dict(state)                                  
#                                                                        
#    if params.add_zenith:                                               
#        from earth2mip.networks import CosZenWrapper
#        nlat = params.img_shape_x                                       
#        nlon = params.img_shape_y                                       
#        lat = 90 - np.arange(nlat) * 0.25                               
#        lon = np.arange(nlon) * 0.25                                    
#        model = CosZenWrapper(model, lon, lat)                         
#                                                                        
#    return model   
def sfno(package, pretrained: bool = True) -> torch.nn.Module:          
    """Load SFNO model from checkpoints trained with era5_wind"""       
    from earth2mip.networks import Wrapper                                
    try:                                                                
        path = package.get("config.json")                               
        params = ParamsBase.from_json(path)                             
        params.checkpointing = False                                    
        model = sfnonet.SphericalFourierNeuralOperatorNet(params)                       
        #from fcn_mip.networks import fcn_dev                           
        #if params.add_zenith:                                          
        #    params.N_in_channels -= 1                                  
        #model = fcn_dev.sfno.SphericalFourierNeuralOperatorNet(params)                                                                        
    except:                                                             
        path = package.get("hyperparams.yaml")                          
        from ruamel.yaml import YAML                                    
        yaml = YAML()                                                   
        with open(path, 'r') as f:                                      
            hparams = yaml.load(f)                                      
        params = SimpleNamespace()                                      
        for k,v in hparams.items():                                     
            setattr(params, k, v)                                       
        #if params.add_zenith:                                          
        #    params.N_in_channels += 1                                  
        print("Drop Path Rate {}".format(params.drop_path_rate))        
        params.checkpointing = False
        from earth2mip.networks import fcn_dev                            
        #model = sfnonet.SphericalFourierNeuralOperatorNet(params)                                                                        
        model = fcn_dev.sfno.SphericalFourierNeuralOperatorNet(params)                  
    if pretrained:                                                      
        weights = package.get("weights.tar")                            
        checkpoint = torch.load(weights, map_location='cpu')                                
        load_me = Wrapper(model)                                        
        state = checkpoint["model_state"]                               
        state = _fix_state_dict_keys(state)                             
        state = {"module.device_buffer": model.device_buffer, **state}  
        load_me.load_state_dict(state)                                  
                                                                        
    if params.add_zenith:                                               
        from earth2mip.networks import CosZenWrapper                      
        nlat = params.img_shape_x                                       
        nlon = params.img_shape_y                                       
        lat = 90 - np.arange(nlat) * 0.25                               
        lon = np.arange(nlon) * 0.25                                    
        model = CosZenWrapper(model, lon, lat)                          
    model.eval()                                                        
                                                                        
    return model 

def _fix_state_dict_keys(state_dict, add_module=False):
    """Add or remove 'module.' from state_dict keys

    Parameters
    ----------
    state_dict : Dict
        Model state_dict
    add_module : bool, optional
        If True, will add 'module.' to keys, by default False

    Returns
    -------
    Dict
        Model state_dict with fixed keys
    """
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if add_module:
            new_key = "module." + key
        else:
            continue
            new_key = key.replace("model.", "")
        fixed_state_dict[new_key] = value
    return fixed_state_dict

def _load_params(fname, change_N_in_channels=True):
    yaml = YAML()                                                       
    with open(fname) as f:                                              
        hparams = yaml.load(f)                                          
    params = SimpleNamespace()                                          
    for k,v in hparams.items():                                         
        setattr(params, k, v)
    if params.add_zenith and change_N_in_channels:
        params.N_in_channels += 1
    return params

def _create_swin(package, architecture=None, constructor=swin.swinv2net):
    try:
        print("Using hyperparams.yaml from the model package")
        config_path = package.get('hyperparams.yaml')
    except:
        print("Using default hyperparams")
        config_path = pathlib.Path(__file__).parent / "swin" / "{}.yaml".format(architecture)
    
    params = _load_params(config_path)
    model = constructor(params, checkpoint_stages=True)
    model.eval()

    if params.add_zenith:
        from earth2mip.networks import CosZenWrapper
        lat = 90 - np.arange(721) * 0.25
        lat = lat[:-1]
        lon = np.arange(1440) * 0.25
        return CosZenWrapper(model, lon, lat)
    else:
        return Wrapper(model)
    
# def fix_state_dict(state_dict):
#     # Add 'model.' prefix to each key
#     return {f'model.{k}': v for k, v in state_dict.items()}

# # Load the state dict with modification
# model_state_dict = torch.load('path_to_saved_model.pth')
# fixed_state_dict = fix_state_dict(model_state_dict)
# model.load_state_dict(fixed_state_dict)

    #     # Check for the extra "model." prefix in the state dictionary keys
    # extra_prefix = 'model.'
    # prefix_found = any(key.startswith(extra_prefix) for key in model_state_dict.keys())

    # print(f"Extra '{extra_prefix}' prefix found in keys: {prefix_found}")

    # # If the extra prefix is found, modify the state dict to remove the prefix
    # if prefix_found:
    #     corrected_state_dict = {key[len(extra_prefix):]: value 
    #                             for key, value in model_state_dict.items()}
    #     model_state_dict = corrected_state_dict


def swin_loader(package, pretrained=True):
    model = _create_swin(package)
    path = package.get('weights.tar')
    checkpoint = torch.load(path)

    weights = checkpoint["model_state"]
    # Adjust the keys in the state dictionary
    new_weights = {}
    for key, value in weights.items():
        new_key = key.replace('module.', '')  # Remove 'module.' if present

        # Check for double 'model.' prefix and remove only one
        if new_key.startswith('model.model.'):
            new_key = new_key[len('model.'):]

        new_weights[new_key] = value
    model.load_state_dict(new_weights, strict=True)
    model.eval()

    return model

def old_swin_loader(package, pretrained=True):
    model = _create_swin(package)
    path = package.get('weights.tar')
    checkpoint = torch.load(path)

    weights = checkpoint["model_state"]
    # Adjust the keys in the state dictionary
    new_weights = {}
    for key, value in weights.items():
        new_key = key.replace('module.', 'model.')  # Remove 'module.' if present

        # Check for double 'model.' prefix and remove only one
        # if new_key.startswith('model.model.'):
        #     new_key = new_key[len('model.'):]

        new_weights[new_key] = value
    model.load_state_dict(new_weights, strict=True)
    model.eval()

    return model

def swin_residual_loader(package, pretrained=True):
    #model = _create_swin(package, constructor=swin_residual.swinv2net)
    model = swin_residual.swin_from_yaml(package.get("hyperparams.yaml"))
    lat = 90 - np.arange(721) * 0.25
    lat = lat[:-1]
    lon = np.arange(1440) * 0.25
    config_path = package.get('hyperparams.yaml')
    params = _load_params(config_path, change_N_in_channels=False)
    model = SwinResidualWrapper(model, lon, lat, params)
    path = package.get('weights.tar')
    checkpoint = torch.load(path)
    weights = checkpoint["model_state"]

    weights = {k.replace('module.', ''): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    model.eval()

    return model

import torch
import torch.nn as nn
import numpy as np

class PreProcessor(nn.Module):

    def __init__(self, params):
        super(PreProcessor, self).__init__()
        
        self.params = params
        imgx, imgy = params.img_size
        
        static_features = None
        if self.params.add_landmask:

            with torch.no_grad():
                lsm = torch.tensor(get_land_mask(params.landmask_path), dtype=torch.long)
                # one hot encode and move channels to front:
                lsm = torch.permute(torch.nn.functional.one_hot(lsm), (2, 0, 1)).to(torch.float32)
                lsm = torch.reshape(lsm, (1, lsm.shape[0], lsm.shape[1], lsm.shape[2]))[:,:,:imgx,:imgy]

                if static_features is None:
                    static_features = lsm
                else:
                    static_features = torch.cat([static_features, lsm], dim=1)


        if self.params.add_orography:

            with torch.no_grad():
                oro = torch.tensor(get_orography(params.orography_path), dtype=torch.float32)
                oro = torch.reshape(oro, (1, 1, oro.shape[0], oro.shape[1]))[:,:,:imgx,:imgy]

                # normalize
                eps = 1.0e-6
                oro = (oro - torch.mean(oro)) / (torch.std(oro) + eps)

                if static_features is None:
                    static_features = oro
                else:
                    static_features = torch.cat([static_features, oro], dim=1)
        self.do_add_static_features = static_features is not None
        if self.do_add_static_features:
            self.register_buffer("static_features", static_features, persistent=False)


    def forward(self, data):
        if self.params.add_zenith:
            # data has inp, tar, izen, tzen
            inp, tar, izen, tzen = data
            inp = torch.cat([inp, izen], dim=1)  # Concatenate input with zenith angle
        else:
            inp, tar = data

        if self.do_add_static_features:
            inp = torch.cat([inp, self.static_features], dim=1)

        return inp, tar

class SwinResidualWrapper(torch.nn.Module):                                      
    def __init__(self, model, lon, lat, params):                                   
        super().__init__()                                                 
        self.model = model                                                 
        self.lon = lon                                                  
        self.lat = lat                                                  
        self.preprocessor = PreProcessor(params)
                                                                        
    def forward(self, x, time):                                         
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)               
        cosz = cos_zenith_angle(time, lon_grid, lat_grid)               
        cosz = cosz.astype(np.float32)                                  
        z = torch.from_numpy(cosz).to(device=x.device)                  
        # assume no history                                             
        x = torch.cat([x, z[None, None]], dim=1)                        
        x = torch.cat([x, self.preprocessor.static_features], dim=1)
        return self.model(x)

import numpy as np
import torch
import datetime
from netCDF4 import Dataset as DS
import h5py


def get_orography(orography_path):
    """returns the surface geopotential for each grid point after normalizing it to be in the range [0, 1]"""

    with DS(orography_path, "r") as f:
        orography = f.variables["Z"][0, :, :]

        orography = (orography - orography.min()) / (orography.max() - orography.min())

    return orography


def get_land_mask(land_mask_path):
    """returns the land mask for each grid point. land sea mask is between 0 and 1"""

    with h5py.File(land_mask_path, "r") as f:
        lsm = f["LSM"][0, :, :]

    return lsm
