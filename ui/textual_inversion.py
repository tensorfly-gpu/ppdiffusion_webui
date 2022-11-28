import paddle
from PIL import Image

def image_grid(imgs, rows=2, cols=2):
    imgs = imgs.astype("uint8")
    imgs = [Image.fromarray(img) for img in imgs]
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
    
@paddle.no_grad()
def log_image(
                input_ids=None,
                text_encoder=None,
                tokenizer=None,
                unet=None,
                vae=None,
                eval_scheduler=None,
                height=512,
                width=512,
                guidance_scale=7.5,
                **kwargs):
    text_encoder.eval()
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
        )
    # only log 8 image
    # if input_ids.shape[0] == 1:
    #     input_ids = input_ids.tile([4, 1])

    text_embeddings = text_encoder(input_ids)[0]
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        batch_size, max_length = input_ids.shape
        uncond_input = tokenizer([""] * batch_size,
                                        padding="max_length",
                                        truncation=True,
                                        max_length=max_length,
                                        return_tensors="pd")
        uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
        text_embeddings = paddle.concat(
            [uncond_embeddings, text_embeddings], axis=0)

    latents = paddle.randn((input_ids.shape[0], unet.in_channels,
                            height // 8, width // 8))
    # ddim donot use this
    latents = latents * eval_scheduler.init_noise_sigma

    for t in eval_scheduler.timesteps:
        # expand the latents if we are doing classifier free guidance
        latent_model_input = paddle.concat(
            [latents] * 2) if do_classifier_free_guidance else latents
        # ddim donot use this
        latent_model_input = eval_scheduler.scale_model_input(
            latent_model_input, t)

        # predict the noise residual
        noise_pred = unet(latent_model_input,
                                t,
                                encoder_hidden_states=text_embeddings).sample
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = eval_scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]) * 255.
    text_encoder.train()
    return image.numpy().round()

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import argparse
import itertools
import math
import os
import random
import numpy as np
import paddle
import glob
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler

from paddlenlp.utils.log import logger
from paddlenlp.trainer import set_seed
from ppdiffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from ppdiffusers.optimization import get_scheduler
from ppdiffusers.modeling_utils import unwrap_model

import PIL
from PIL import Image
from paddlenlp.utils.tools import compare_version

if compare_version(PIL.__version__, "9.1.0") >= 0:
    Resampling = PIL.Image.Resampling
else:
    Resampling = PIL.Image
from paddle.vision.transforms import RandomHorizontalFlip
from paddle.optimizer import AdamW
from tqdm.auto import tqdm
from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer


def get_writer(args):
    if args.writer_type == "visualdl":
        from visualdl import LogWriter
        writer = LogWriter(logdir=args.logging_dir)
    elif args.writer_type == "tensorboard":
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(logdir=args.logging_dir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer


def save_progress(text_encoder, placeholder_token_id, args, global_step=-1):
    learned_embeds = unwrap_model(
        text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {
        args.placeholder_token: learned_embeds.detach().cpu()
    }

    # remove \/"*:?<>| in filename
    name = args.placeholder_token
    name = name.translate({
        92: 95,
        47: 95,
        42: 95,
        34: 95,
        58: 95,
        63: 95,
        60: 95,
        62: 95,
        124: 95
    })
    path = os.path.join(args.output_dir, "step-"+str(global_step))
    os.makedirs(path, exist_ok=True)
    paddle.save(learned_embeds_dict,
                os.path.join(args.output_dir, "step-"+str(global_step), f"{name}.pdparams"))
    print(
        f"Global_step: {global_step} 程序没有卡住，目前正在生成评估图片，请耐心等待！训练好的权重和评估图片将会自动保存到 {path} 目录下。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10,
        help="Save learned_embeds.pdparams every X updates steps.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model or model identifier from local models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--train_data_dir",
                        type=str,
                        default=None,
                        required=False,
                        help="A folder containing the training data.")
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=False,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument("--initializer_token",
                        type=str,
                        default=None,
                        required=False,
                        help="A token to use as initializer word.")
    parser.add_argument("--learnable_property",
                        type=str,
                        default="object",
                        help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats",
                        type=int,
                        default=100,
                        help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help=
        ("The height for input images, all the images in the train/validation dataset will be resized to this"
         " height"),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help=
        ("The width for input images, all the images in the train/validation dataset will be resized to this"
         " width"),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=
        ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
         ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--adam_beta1",
                        type=float,
                        default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2",
                        type=float,
                        default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay",
                        type=float,
                        default=1e-2,
                        help="Weight decay to use.")
    parser.add_argument("--adam_epsilon",
                        type=float,
                        default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) or [VisualDL](https://www.paddlepaddle.org.cn/paddle/visualdl) log directory. Will default to"
         "*output_dir/logs"),
    )
    parser.add_argument("--writer_type",
                        type=str,
                        default="visualdl",
                        choices=["tensorboard", "visualdl"],
                        help="Log writer type.")

    args = parser.parse_args(args=[])
    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]



class TextualInversionDataset(Dataset):

    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        height=512,
        width=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.height = height
        self.width = width
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        ext = ['png', 'jpg', 'jpeg', 'bmp']
        self.image_paths = []
        for e in ext:
            self.image_paths.extend(glob.glob(os.path.join(data_root,
                                                           '*.' + e)))

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": Resampling.BILINEAR,
            "bilinear": Resampling.BILINEAR,
            "bicubic": Resampling.BICUBIC,
            "lanczos": Resampling.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = RandomHorizontalFlip(prob=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids


        example["input_ids_eval"] = self.tokenizer(
            placeholder_string,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2:(h + crop) // 2,
                      (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.width, self.height),
                             resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32).transpose([2, 0, 1])

        example["pixel_values"] = image
        return example


def freeze_params(params):
    for param in params:
        param.stop_gradient = True


def main(args):
    rank = paddle.distributed.get_rank()
    num_processes = paddle.distributed.get_world_size()
    if num_processes > 1:
        paddle.distributed.init_parallel_env()

    # If passed along, set the training seed now.
    if args.seed is not None:
        seed = args.seed + rank
        set_seed(seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.model_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(args.model_name, "tokenizer"))

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        # raise ValueError(
        #     f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
        #     " `placeholder_token` that is not already in the tokenizer.")
        raise ValueError(f"单词 {args.placeholder_token} 原本就已经存在了哦. 请用一个新的词汇.")

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token,
                                 add_special_tokens=False)["input_ids"]
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        # raise ValueError("The initializer token must be a single token.")
        raise ValueError(
            f"用来初始化的 ‘最接近的单词’ 只能是一个简单词, {args.initializer_token} 不可以哟.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(
        args.placeholder_token)

    # Load models and create wrapper for stable diffusion
    if args.text_encoder is None:
        text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(args.model_name, "text_encoder"))
    else:
        text_encoder = args.text_encoder

    if args.vae is None:
        vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
    else:
        vae = args.vae

    if args.unet is None:
        unet = UNet2DConditionModel.from_pretrained(args.model_name,
                                                    subfolder="unet")
    else:
        unet = args.unet

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    eval_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012,  beta_schedule='scaled_linear', skip_prk_steps=True)
    eval_scheduler.set_timesteps(50)
    # Initialise the newly added placeholder token with the embeddings of the initializer token
    with paddle.no_grad():
        token_embeds = text_encoder.get_input_embeddings()
        token_embeds.weight[placeholder_token_id] = token_embeds.weight[
            initializer_token_id]

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.transformer.parameters(),
        text_encoder.text_model.ln_final.parameters(),
        text_encoder.text_model.positional_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    if args.scale_lr:
        args.learning_rate = (args.learning_rate *
                              args.gradient_accumulation_steps *
                              args.train_batch_size * num_processes)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps *
        args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps *
        args.gradient_accumulation_steps,
    )

    if num_processes > 1:
        text_encoder = paddle.DataParallel(text_encoder)

    # Initialize the optimizer
    optimizer = AdamW(learning_rate=lr_scheduler,
                      parameters=unwrap_model(
                          text_encoder).get_input_embeddings().parameters(),
                      beta1=args.adam_beta1,
                      beta2=args.adam_beta2,
                      weight_decay=args.adam_weight_decay,
                      epsilon=args.adam_epsilon,
                      grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm)
                      if args.max_grad_norm is not None else None)

    noise_scheduler = DDPMScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    num_train_timesteps=1000)

    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        height=args.height,
        width=args.width,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )

    def collate_fn(examples):
        input_ids = [example["input_ids"] for example in examples]
        input_ids_eval = [example["input_ids_eval"] for example in examples]
        pixel_values = paddle.to_tensor(
            [example["pixel_values"] for example in examples], dtype="float32")
        input_ids = tokenizer.pad({
            "input_ids": input_ids
        },
                                  padding="max_length", max_length=tokenizer.model_max_length,
                                  return_tensors="pd").input_ids
        input_ids_eval = tokenizer.pad({
            "input_ids": input_ids_eval
        },
                                  padding="max_length", max_length=tokenizer.model_max_length,
                                  return_tensors="pd").input_ids
        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "input_ids_eval": input_ids_eval,
        }
        return batch

    train_sampler = DistributedBatchSampler(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True) if num_processes > 1 else BatchSampler(
            train_dataset, batch_size=args.train_batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  collate_fn=collate_fn)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)

    if rank == 0:
        # logger.info('-----------  Configuration Arguments -----------')
        # for arg, value in sorted(vars(args).items()):
        #     logger.info('%s: %s' % (arg, value))
        # logger.info('------------------------------------------------')
        writer = get_writer(args)

    # Train!
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num Epochs = {args.num_train_epochs}")
    # logger.info(
    #     f"  Instantaneous batch size per device = {args.train_batch_size}")
    # logger.info(
    #     f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    # )
    # logger.info(
    #     f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=rank > 0)
    progress_bar.set_description("Train Steps")
    global_step = 0

    text_encoder_embedding_clone = unwrap_model(
        text_encoder).get_input_embeddings().weight.clone()

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()
    text_encoder.train()
    try:
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                with paddle.no_grad():
                    # Convert images to latent space
                    latents = vae.encode(
                        batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = paddle.randn(latents.shape)
                    batch_size = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = paddle.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (batch_size, )).astype("int64")

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(
                        latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states).sample

                loss = F.mse_loss(noise_pred, noise,
                                  reduction="none").mean([1, 2, 3]).mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                with paddle.no_grad():
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = (paddle.arange(
                        len(tokenizer)) == placeholder_token_id
                                           ).astype("float32").unsqueeze(-1)
                    unwrap_model(text_encoder).get_input_embeddings(
                    ).weight.grad = unwrap_model(
                        text_encoder).get_input_embeddings(
                        ).weight.grad * index_grads_to_zero

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    with paddle.no_grad():
                        unwrap_model(text_encoder).get_input_embeddings(
                        ).weight[:-1] = text_encoder_embedding_clone[:-1]

                    lr_scheduler.step()
                    optimizer.clear_grad()
                    progress_bar.update(1)
                    global_step += 1
                    logs = {
                        "epoch":
                        str(epoch).zfill(4),
                        "step_loss":
                        round(loss.item() * args.gradient_accumulation_steps,
                              10),
                        "lr":
                        lr_scheduler.get_lr()
                    }
                    progress_bar.set_postfix(**logs)
                    if rank == 0:
                        for name, val in logs.items():
                            if name == "epoch": continue
                            writer.add_scalar(f"train/{name}",
                                              val,
                                              step=global_step)
                        if global_step % args.save_steps == 0:
                            save_progress(text_encoder, placeholder_token_id,
                                          args, global_step)
                            all_images = []
                            loop = 4 if batch["input_ids_eval"].shape[0] == 1 else 1
                            for i in range(loop):
                                img = log_image(batch["input_ids_eval"], 
                                                tokenizer=tokenizer,
                                                vae = unwrap_model(vae),
                                                eval_scheduler=eval_scheduler,
                                                text_encoder=unwrap_model(text_encoder), 
                                                unet=unwrap_model(unet))
                                all_images.append(img)
                            if len(all_images) > 1:
                                all_images = np.concatenate(all_images, axis=0)
                            else:
                                all_images = all_images[0]
                            writer.add_image("images", all_images,
                                                    step=global_step,
                                                    dataformats="NHWC")
                            name = args.placeholder_token
                            name = name.translate({
                                92: 95,
                                47: 95,
                                42: 95,
                                34: 95,
                                58: 95,
                                63: 95,
                                60: 95,
                                62: 95,
                                124: 95
                            })
                            image_grid(all_images).save(os.path.join(args.output_dir, "step-"+str(global_step), f"{name}.jpg"))


                if global_step >= args.max_train_steps:
                    break

        if rank == 0:
            writer.close()
            # # Create the pipeline using using the trained modules and save it.
            # pipeline = StableDiffusionPipeline.from_pretrained(
            #     args.model_name,
            #     text_encoder=unwrap_model(text_encoder),
            #     safety_checker=None,
            #     tokenizer=tokenizer,
            # )
            # pipeline.save_pretrained(args.output_dir)
            # Also save the newly trained embeddings
            save_progress(text_encoder, placeholder_token_id, args, global_step)
            print(f'训练完毕, 可以用新词 {args.placeholder_token} 去生成图片了.')

            import gc
            gc.collect()
            paddle.device.cuda.empty_cache()
    except:
        save_progress(text_encoder, placeholder_token_id, args, global_step)
        import gc
        gc.collect()
        del text_encoder
        del optimizer
        del vae
        del unet
        del text_encoder_embedding_clone
        paddle.device.cuda.empty_cache()
