# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import io
import os
import pickle
from functools import lru_cache

import numpy as np
from _io import BufferedReader

MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30
import argparse
import numpy as np
import paddle
import pickle
from functools import lru_cache
from paddlenlp.utils.downloader import get_path_from_url
try:
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError(
        "OmegaConf is required to convert the LDM checkpoints. Please install it with `pip install OmegaConf`."
    )
from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler
)

import io
import os
import pickle
from functools import lru_cache

import numpy as np
from _io import BufferedReader

from .safetensor import safe_open
MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30


class TensorMeta:
    """
    metadata of tensor
    """

    def __init__(self, key: str, n_bytes: int, dtype: str):
        self.key = key
        self.nbytes = n_bytes
        self.dtype = dtype
        self.size = None

    def __repr__(self):
        return f"size: {self.size} key: {self.key}, nbytes: {self.nbytes}, dtype: {self.dtype}"


class SerializationError(Exception):
    """Exception for serialization"""

    pass


@lru_cache(maxsize=None)
def _storage_type_to_dtype_to_map():
    """convert storage type to numpy dtype"""
    return {
        "DoubleStorage": np.double,
        "FloatStorage": np.float32,
        "HalfStorage": np.half,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
        "ShortStorage": np.int16,
        "CharStorage": np.int8,
        "ByteStorage": np.uint8,
        "BoolStorage": np.bool8,
        "ComplexDoubleStorage": np.cdouble,
        "ComplexFloatStorage": np.cfloat,
    }


class StorageType:
    """Temp Class for Storage Type"""

    def __init__(self, name):
        self.dtype = _storage_type_to_dtype_to_map()[name]

    def __str__(self):
        return f"StorageType(dtype={self.dtype})"


def _element_size(dtype: str) -> int:
    """
    Returns the element size for a dtype, in bytes
    """
    if dtype in [np.float16, np.float32, np.float64]:
        return np.finfo(dtype).bits >> 3
    elif dtype == np.bool8:
        return 1
    else:
        return np.iinfo(dtype).bits >> 3


class UnpicklerWrapperStage(pickle.Unpickler):
    def find_class(self, mod_name, name):
        if type(name) is str and "Storage" in name:
            try:
                return StorageType(name)
            except KeyError:
                pass

        # pure torch tensor builder
        if mod_name == "torch._utils":
            return _rebuild_tensor_stage

        # pytorch_lightning tensor builder
        if "pytorch_lightning" in mod_name:
            return dumpy
        return super().find_class(mod_name, name)


def get_data_iostream(file: str, file_name="data.pkl"):
    FILENAME = f"archive/{file_name}".encode("latin")
    padding_size_plus_fbxx = 4 + 14
    data_iostream = []
    offset = MZ_ZIP_LOCAL_DIR_HEADER_SIZE + len(FILENAME) + padding_size_plus_fbxx
    with open(file, "rb") as r:
        r.seek(offset)
        for bytes_data in io.BytesIO(r.read()):
            if b".PK" in bytes_data:
                data_iostream.append(bytes_data.split(b".PK")[0])
                data_iostream.append(b".")
                break
            data_iostream.append(bytes_data)
    out = b"".join(data_iostream)
    return out, offset + len(out)


def _rebuild_tensor_stage(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    if isinstance(storage, TensorMeta):
        storage.size = size
    return storage


def dumpy(*args, **kwarsg):
    return None


def seek_by_string(file_handler: BufferedReader, string: str, file_size: int) -> int:
    """seek the index of file-handler with target words

    Args:
        file_handler (BufferedReader): file handler
        string (str): the specific string in the file
        file_size (int): size of file

    Returns:
        int: end index of target string
    """
    word_index = 0
    word_bytes = string.encode("latin")
    empty_byte = "".encode("latin")

    while word_index < len(string) and file_handler.tell() < file_size:
        content = file_handler.read(1)
        if content == empty_byte:
            break

        if word_bytes[word_index] == content[0]:
            word_index += 1
        else:
            word_index = 0

    if file_handler.tell() >= file_size - 1:
        raise SerializationError(f"can't find the find the target string<{string}> in the file")
    return file_handler.tell()


def read_prefix_key(file_handler: BufferedReader, file_size: int):
    """read the prefix key in model weight file, eg: archive/pytorch_model

    Args:
        file_handler (BufferedReader): file handler
        fiel_size (_type_): size of file

    Returns:
        _type_: _description_
    """
    end_index = seek_by_string(file_handler, "data.pkl", file_size)
    file_handler.seek(MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
    prefix_key = file_handler.read(end_index - MZ_ZIP_LOCAL_DIR_HEADER_SIZE - len("/data.pkl"))
    return prefix_key


def load_torch(path: str, **pickle_load_args):
    """
    load torch weight file with the following steps:

    1. load the structure of pytorch weight file
    2. read the tensor data and re-construct the state-dict

    Args:
        path: the path of pytorch weight file
        **pickle_load_args: args of pickle module

    Returns:

    """
    pickle_load_args.update({"encoding": "utf-8"})

    # 1. load the structure of pytorch weight file
    def persistent_load_stage1(saved_id):
        assert isinstance(saved_id, tuple)
        data = saved_id[1:]
        storage_type, key, _, numel = data
        dtype = storage_type.dtype
        n_bytes = numel * _element_size(dtype)
        return TensorMeta(key, n_bytes, dtype)

    data_iostream, pre_offset = get_data_iostream(path, file_name="data.pkl")
    # 1. read the structure of storage
    unpickler_stage1 = UnpicklerWrapperStage(io.BytesIO(data_iostream), **pickle_load_args)
    unpickler_stage1.persistent_load = persistent_load_stage1
    result_stage1 = unpickler_stage1.load()

    # 2. get the metadata of weight file
    metadata = {}

    def extract_maybe_dict(result):
        if isinstance(result, dict):
            for k, v in result.items():
                extract_maybe_dict(v)
        elif isinstance(result, (list, tuple)):
            for res in result:
                extract_maybe_dict(res)
        elif isinstance(result, TensorMeta):
            metadata[result.key] = result

    extract_maybe_dict(result_stage1)
    metadata = list(metadata.values())
    metadata = sorted(metadata, key=lambda x: x.key)
    # 3. parse the tensor of pytorch weight file
    stage1_key_to_tensor = {}
    content_size = os.stat(path).st_size
    with open(path, "rb") as file_handler:
        file_handler.seek(pre_offset)
        for tensor_meta in metadata:
            key = tensor_meta.key            
            seek_by_string(file_handler, f"data/{key}", content_size)
            seek_by_string(file_handler, "FB", content_size)

            padding_offset = np.frombuffer(file_handler.read(2)[:1], dtype=np.uint8)[0]
            file_handler.seek(padding_offset, 1)

            # save the tensor info in result to re-use memory
            stage1_key_to_tensor[key] = np.frombuffer(
                file_handler.read(tensor_meta.nbytes), dtype=tensor_meta.dtype
            ).reshape(tensor_meta.size)

    def persistent_load_stage2(saved_id):
        assert isinstance(saved_id, tuple)
        key = saved_id[2]
        return stage1_key_to_tensor[key]

    # 4. read the structure of storage
    unpickler_stage2 = UnpicklerWrapperStage(io.BytesIO(data_iostream), **pickle_load_args)
    unpickler_stage2.persistent_load = persistent_load_stage2
    result_stage2 = unpickler_stage2.load()

    return result_stage2


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item
        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "query.weight")
        new_item = new_item.replace("q.bias", "query.bias")

        new_item = new_item.replace("k.weight", "key.weight")
        new_item = new_item.replace("k.bias", "key.bias")

        new_item = new_item.replace("v.weight", "value.weight")
        new_item = new_item.replace("v.bias", "value.bias")

        new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
        new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            
            query, key, value = np.split(old_tensor, 3, axis=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def create_unet_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    unet_params = original_config.model.params.unet_config.params
    vae_params = original_config.model.params.first_stage_config.params.ddconfig

    block_out_channels = [unet_params.model_channels * mult for mult in unet_params.channel_mult]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params.attention_resolutions else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params.attention_resolutions else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    vae_scale_factor = 2 ** (len(vae_params.ch_mult) - 1)

    head_dim = unet_params.num_heads if "num_heads" in unet_params else None
    use_linear_projection = (
        unet_params.use_linear_in_transformer if "use_linear_in_transformer" in unet_params else False
    )
    if use_linear_projection:
        # stable diffusion 2-base-512 and 2-768
        if head_dim is None:
            head_dim = [5, 10, 20, 20]

    config = dict(
        sample_size=image_size // vae_scale_factor,
        in_channels=unet_params.in_channels,
        out_channels=unet_params.out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=unet_params.num_res_blocks,
        cross_attention_dim=unet_params.context_dim,
        attention_head_dim=head_dim,
        use_linear_projection=use_linear_projection,
    )

    return config


def create_vae_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config.model.params.first_stage_config.params.ddconfig
    _ = original_config.model.params.first_stage_config.params.embed_dim

    block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = dict(
        sample_size=image_size,
        in_channels=vae_params.in_channels,
        out_channels=vae_params.out_ch,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=vae_params.z_channels,
        layers_per_block=vae_params.num_res_blocks,
    )
    return config


def create_diffusers_schedular(original_config):
    schedular = DDIMScheduler(
        num_train_timesteps=original_config.model.params.timesteps,
        beta_start=original_config.model.params.linear_start,
        beta_end=original_config.model.params.linear_end,
        beta_schedule="scaled_linear",
    )
    return schedular

def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"] if len(checkpoint["state_dict"]) > 25 else checkpoint
    # extract state_dict for UNet
    unet_state_dict = {}
    keys = list(checkpoint.keys())

    unet_key = "model.diffusion_model."

    # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
    if sum(k.startswith("model_ema") for k in keys) > 100:
        print(f"Checkpoint {path} has both EMA and non-EMA weights.")
        if extract_ema:
            print(
                "我们将提取EMA版的UNET权重。如果你想使用非EMA版权重进行微调的话，请确保将『是否提取ema权重』选项设置为 『否』！"
            )
            for key in keys:
                if key.startswith("model.diffusion_model"):
                    flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
        else:
            print(
                "我们将提取非EMA版的UNET权重。如果你想使用EMA版权重的话，请确保将『是否提取ema权重』选项设置为 『是』"
            )

    if extract_ema and len(unet_state_dict) == 0:
        print("由于我们在CKPT中未找到EMA权重，因此我们将不会『提取ema权重』！")

    # 如果没有找到ema的权重，
    if len(unet_state_dict) == 0:
        for key in keys:
            if "model_ema" in key: continue
            if key.startswith(unet_key):
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)
            
    if len(unet_state_dict) == 0:
        return None
    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }
    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }
    
    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]


        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            if ["conv.weight", "conv.bias"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []
            elif ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]
    return new_checkpoint


def convert_ldm_vae_checkpoint(checkpoint, config, only_vae=False):
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"] if len(checkpoint["state_dict"]) > 25 else checkpoint
    # extract state dict for VAE
    vae_state_dict = {}
    vae_key = "first_stage_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    if only_vae:
        vae_state_dict = checkpoint
    
    if len(vae_state_dict) == 0:
        return None
    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


def convert_diffusers_vae_unet_to_ppdiffusers(vae_or_unet, diffusers_vae_unet_checkpoint, dtype="float32"):
    need_transpose = []
    for k, v in vae_or_unet.named_sublayers(include_self=True):
        if isinstance(v, paddle.nn.Linear):
            need_transpose.append(k + ".weight")
    new_vae_or_unet = {}
    for k, v in diffusers_vae_unet_checkpoint.items():
        if k not in need_transpose:
            new_vae_or_unet[k] = v.astype(dtype)
        else:
            new_vae_or_unet[k] = v.T.astype(dtype)
    return new_vae_or_unet


def check_keys(model, state_dict):
    cls_name = model.__class__.__name__
    missing_keys = []
    mismatched_keys = []
    for k, v in model.state_dict().items():
        if k not in state_dict.keys():
            missing_keys.append(k)
        else:            
            if list(v.shape) != list(state_dict[k].shape):
                mismatched_keys.append(k)
    if len(missing_keys):
        missing_keys_str = ", ".join(missing_keys)
        print(f"{cls_name} Found missing_keys {missing_keys_str}!")
    if len(mismatched_keys):
        mismatched_keys_str = ", ".join(mismatched_keys)
        print(f"{cls_name} Found mismatched_keys {mismatched_keys_str}!")


def convert_hf_clip_to_ppnlp_clip(checkpoint, dtype="float32"):
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"] if len(checkpoint["state_dict"]) > 25 else checkpoint
    clip = {}
    for key in checkpoint.keys():
        if key.startswith("cond_stage_model.transformer"):
            newkey = key[len("cond_stage_model.transformer.") :]
            if not newkey.startswith("text_model."):
                newkey = "text_model." + newkey
            clip[newkey] = checkpoint[key]

    if len(clip) == 0:
        return None, None
    
    new_model_state = {}
    transformers2ppnlp = {
        ".encoder.": ".transformer.",
        ".layer_norm": ".norm",
        ".mlp.": ".",
        ".fc1.": ".linear1.",
        ".fc2.": ".linear2.",
        ".final_layer_norm.": ".ln_final.",
        ".embeddings.": ".",
        ".position_embedding.": ".positional_embedding.",
        ".patch_embedding.": ".conv1.",
        "visual_projection.weight": "vision_projection",
        "text_projection.weight": "text_projection",
        ".pre_layrnorm.": ".ln_pre.",
        ".post_layernorm.": ".ln_post.",
        ".vision_model.": ".",
    }
    ignore_value = ["position_ids"]
    donot_transpose = ["embeddings", "norm", "concept_embeds", "special_care_embeds"]
    for name, value in clip.items():
        # step1: ignore position_ids
        if any(i in name for i in ignore_value):
            continue
        # step2: transpose nn.Linear weight
        if value.ndim == 2 and not any(i in name for i in donot_transpose):
            value = value.T
        # step3: hf_name -> ppnlp_name mapping
        for hf_name, ppnlp_name in transformers2ppnlp.items():
            name = name.replace(hf_name, ppnlp_name)
        # step4: 0d tensor -> 1d tensor
        if name == "logit_scale":
            value = value.reshape((1,))

        new_model_state[name] = value.astype(dtype)

    new_config = {
        "max_text_length": new_model_state["text_model.positional_embedding.weight"].shape[0],
        "vocab_size": new_model_state["text_model.token_embedding.weight"].shape[0],
        "text_embed_dim": new_model_state["text_model.token_embedding.weight"].shape[1],
        "text_heads": 12,
        "text_layers": 12,
        "text_hidden_act": "quick_gelu",
        "projection_dim": 768,
        "initializer_range": 0.02,
        "initializer_factor": 1.0,
    }
    return new_model_state, new_config

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--vae_checkpoint_path", default=None, type=str, help="Path to the vae checkpoint to convert."
    )
    # !wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--num_in_channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="pndm",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']",
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
        help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        ),
    )
    parser.add_argument("--dump_path", default=None, type=str, help="Path to the output model.")
    args = parser.parse_known_args()[0]
    return args

def main(args): #主函数
    if args.checkpoint_path.strip() == "":
        print("ckpt模型文件位置不能为空！")
        return 
    if not os.path.exists(args.checkpoint_path):
        print(f"{args.checkpoint_path} ckpt 文件不存在，请检查是否存在！")
        return 

    if args.vae_checkpoint_path is not None and args.vae_checkpoint_path.strip() == "":
        args.vae_checkpoint_path = None

    if args.vae_checkpoint_path is not None:
        if not os.path.exists(args.vae_checkpoint_path):
            print(f"{args.vae_checkpoint_path} vae 文件不存在，我们将尝试使用ckpt文件的vae权重！")
            args.vae_checkpoint_path = None
    print("正在开始转换，请耐心等待！！！")
    image_size = 512
    checkpoint = {}
    if args.checkpoint_path.endswith("ckpt"):
        checkpoint = load_torch(args.checkpoint_path)
    else:
        tensor = safe_open(args.checkpoint_path)
        tensor.get_md_size()
        tensor.get_metadata()
        for key in tensor.keys():
            checkpoint[key] = tensor.get_tensor(key)
    checkpoint = checkpoint.get("state_dict", checkpoint)
    if args.original_config_file is None:
        get_path_from_url("https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/v1-inference.yaml", root_dir="./")

        args.original_config_file = "./v1-inference.yaml"

    original_config = OmegaConf.load(args.original_config_file)

    if args.num_in_channels is not None:
        original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = args.num_in_channels

    num_train_timesteps = original_config.model.params.timesteps
    beta_start = original_config.model.params.linear_start
    beta_end = original_config.model.params.linear_end

    scheduler = DDIMScheduler(
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        beta_start=beta_start,
        num_train_timesteps=num_train_timesteps,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
    )
    # make sure scheduler works correctly with DDIM
    scheduler.register_to_config(clip_sample=False)

    if args.scheduler_type == "pndm":
        config = dict(scheduler.config)
        config["skip_prk_steps"] = True
        scheduler = PNDMScheduler.from_config(config)
    elif args.scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    elif args.scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
    elif args.scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
    elif args.scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
    elif args.scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    elif args.scheduler_type == "ddim":
        scheduler = scheduler
    else:
        raise ValueError(f"Scheduler of type {args.scheduler_type} doesn't exist!")

    print("1. 开始转换Unet！")
    # 1. Convert the UNet2DConditionModel model.
    diffusers_unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    diffusers_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, diffusers_unet_config, path=args.checkpoint_path, extract_ema=args.extract_ema
    )
    if diffusers_unet_checkpoint is not None:
        unet = UNet2DConditionModel.from_config(diffusers_unet_config)
        ppdiffusers_unet_checkpoint = convert_diffusers_vae_unet_to_ppdiffusers(unet, diffusers_unet_checkpoint)
        check_keys(unet, ppdiffusers_unet_checkpoint)
        unet.load_dict(ppdiffusers_unet_checkpoint)
        print(">>> Unet转换成功！")
    else:
        unet = None
        print("在CKPT中，未发现Unet权重，请确认是否存在！")

    print("2. 开始转换Vae！")
    # 2. Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    if args.vae_checkpoint_path is not None:
        vae_checkpoint = load_torch(args.vae_checkpoint_path)
        print(f"发现 {args.vae_checkpoint_path}，我们将转换该文件的vae权重！")
        only_vae = True
    else:
        vae_checkpoint = checkpoint
        only_vae = False
    diffusers_vae_checkpoint = convert_ldm_vae_checkpoint(vae_checkpoint, vae_config,  only_vae=only_vae)
    if diffusers_vae_checkpoint is not None:
        vae = AutoencoderKL.from_config(vae_config)
        ppdiffusers_vae_checkpoint = convert_diffusers_vae_unet_to_ppdiffusers(vae, diffusers_vae_checkpoint)
        check_keys(vae, ppdiffusers_vae_checkpoint)
        vae.load_dict(ppdiffusers_vae_checkpoint)
        print(">>> VAE转换成功！")
    else:
        vae = None
        print("在CKPT中，未发现Vae权重，请确认是否存在！")
        
    print("3. 开始转换text_encoder！")
    # 3. Convert the text_encoder model.
    text_model_state_dict, text_config = convert_hf_clip_to_ppnlp_clip(checkpoint, dtype="float32")
    if text_model_state_dict is not None:
        text_model = CLIPTextModel(**text_config)
        text_model.eval()
        check_keys(text_model, text_model_state_dict)
        text_model.load_dict(text_model_state_dict)
        print(">>> text_encoder转换成功！")
    else:
        text_model = None
        print("在CKPT中，未发现TextModel权重，请确认是否存在！")

    print("4. 开始转换CLIPTokenizer！")
    # 4. Convert the tokenizer.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", pad_token="!", model_max_length=77)
    print(">>> CLIPTokenizer 转换成功！")
    
    if text_model is not None and vae is not None and unet is not None:
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipe.save_pretrained(args.dump_path)
        print(">>> 所有权重转换完成啦，请前往"+str(args.dump_path)+"查看转换好的模型！")
    else:
        if vae is not None:
            vae.save_pretrained(os.path.join(args.dump_path, "vae"))
        if text_model is not None:
            text_model.save_pretrained(os.path.join(args.dump_path, "text_encoder"))
        if unet is not None:
            unet.save_pretrained(os.path.join(args.dump_path, "unet"))
        scheduler.save_pretrained(os.path.join(args.dump_path, "scheduler"))
        tokenizer.save_pretrained(os.path.join(args.dump_path, "tokenizer"))
        print(">>> 部分权重转换完成啦，请前往"+str(args.dump_path)+"查看转换好的部分模型！")
        
