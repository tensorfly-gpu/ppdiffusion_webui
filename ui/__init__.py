from .env import DEBUG_UI

from .config import config
if DEBUG_UI:
    
    print('==================================================')
    print('调试环境')
    print('==================================================')
    from .StableDiffusionUI_txt2img import StableDiffusionUI_txt2img
    from .StableDiffusionUI_img2img import StableDiffusionUI_img2img
    
    gui_txt2img = StableDiffusionUI_txt2img(**config['txt2img'])
    gui_img2img = StableDiffusionUI_img2img(**config['img2img'])
    gui_inpaint = gui_img2img
    
else:
    
    from .ui import (
        StableDiffusionUI_text_inversion,
        
        pipeline_superres,
        pipeline,
    )
    from .StableDiffusionUI_txt2img import StableDiffusionUI_txt2img
    from .StableDiffusionUI_img2img import StableDiffusionUI_img2img
    from .SuperResolutionUI import SuperResolutionUI


    gui_txt2img = StableDiffusionUI_txt2img(
        **config['txt2img']
    )
    gui_img2img = StableDiffusionUI_img2img(
        **config['img2img']
    )
    gui_superres = SuperResolutionUI(
        pipeline = pipeline_superres,
        **config['superres']
    )
    gui_train_text_inversion = StableDiffusionUI_text_inversion(
        **config['train_text_inversion']
    )
    gui_text_inversion = StableDiffusionUI_txt2img(
        **config['text_inversion']
    )
    gui_inpaint = gui_img2img