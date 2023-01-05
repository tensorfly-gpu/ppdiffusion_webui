config = {
    "txt2img": {
        "prompt": 'extremely detailed CG unity 8k wallpaper,black long hair,cute face,1 adult girl,happy, green skirt dress, flower pattern in dress,solo,green gown,art of light novel,in field',
        "negative_prompt": 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
        "width": 512,
        "height": 512,
        # "seed": -1,
        # "num_return_images": 1,
        # "num_inference_steps": 50,
        # "guidance_scale": 7.5,
        # "fp16": 'float16',
        # "superres_model_name": '无',
        # "max_embeddings_multiples": '3',
        # "enable_parsing": '圆括号 () 加强权重',
        # "sampler": 'default',
        "model_name": 'MoososCap/NOVEL-MODEL',
        "output_dir": 'outputs/txt2img',
    },
    "img2img": {
        "prompt": 'red dress',
        "negative_prompt": 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
        "width": -1,
        "height": -1,
        "num_return_images": 1,
        "strength": 0.8,
        "model_name": 'MoososCap/NOVEL-MODEL',
        "image_path": 'resources/cat2.jpg',
        "mask_path": 'resources/mask8.jpg',
        "output_dir": 'outputs/img2img',
    },
    "superres": {
        "image_path": 'resources/image_Kurisu.png',
        "superres_model_name": 'falsr_a',
        "output_dir": 'outputs/highres',
    },
    "train_text_inversion": {
        "learnable_property": 'object',
        "placeholder_token": '<Alice>',
        "initializer_token": 'girl',
        "repeats": '100',
        "train_data_dir": 'resources/Alices',
        "output_dir": 'outputs/textual_inversion',
        "height": 512,
        "width": 512,
        "learning_rate": 5e-4,
        "max_train_steps": 1000,
        "save_steps": 200,
        "model_name": "MoososCap/NOVEL-MODEL",
    },
    "text_inversion": {
        "width": 512,
        "height": 512,
        "prompt": '<Alice> at the lake',
        "negative_prompt": '',
        "output_dir": 'outputs/text_inversion_txt2img',
    },
    "dreambooth": {                    #Dreambooth配置段
        "pretrained_model_name_or_path": "MoososCap/NOVEL-MODEL",
        "instance_data_dir": 'resources/Alices',
        "instance_prompt": 'a photo of Alices',
        "class_data_dir": 'resources/Girls',
        "class_prompt": 'a photo of girls',
        "num_class_images": 100,
        "prior_loss_weight": 1.0,
        "with_prior_preservation": True,
        #"num_train_epochs": 1,
        "max_train_steps": 1000,
        "save_steps": 1000,
        "train_text_encoder": False,
        "height": 512,
        "width": 512,
        "learning_rate": 5e-6,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 500,
        "center_crop": True,
        "output_dir": 'outputs/dreambooth',
    },
    "convert": {
        "model_root": 'models/',
    },
}

try:
    from user_config import config as _config
    for k in _config:
        if k in config:
            config[k].update(_config[k])
        else:
            config[k] = _config[k]
        
except:
    pass
    
    