config = {
    "txt2img": {
        "width": 512,
        "height": 512,
        "output_dir": 'outputs/txt2img',
        "prompt": 'extremely detailed CG unity 8k wallpaper,black long hair,cute face,1 adult girl,happy, green skirt dress, flower pattern in dress,solo,green gown,art of light novel,in field',
        "negative_prompt": 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
    },

    "img2img": {
        "width": -1,
        "height": -1,
        "num_return_images": 1,
        "output_dir": 'outputs/img2img',
        "prompt": 'red dress',
        "negative_prompt": 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
        "model_name": 'MoososCap/NOVEL-MODEL',
        "image_path": 'resources/cat2.jpg',
        "mask_path": 'resources/mask8.jpg',
        "upload_image_path": 'resources/upload.png',
        "upload_mask_path": 'resources/upload-mask.png',
    },
    
    # 未实现
    "inpaint": {},
    "superres": {},
    "train_text_inversion": {},
    "text_inversion": {},
}

try:
    from user_config import config as _config
    for k in _config:
        if k in config:
            config[k].update(_config[k])
        else:
            config[k] = _config[k]
        
finally:
    pass
    
    