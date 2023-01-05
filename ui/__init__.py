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
        pipeline_superres,
        pipeline,
    )
    from .StableDiffusionUI_text_inversion import StableDiffusionUI_text_inversion
    from .StableDiffusionUI_dreambooth import StableDiffusionUI_dreambooth
    from .StableDiffusionUI_convert import StableDiffusionUI_convert
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
    gui_dreambooth = StableDiffusionUI_dreambooth( #dreamboothUI
        **config['dreambooth']
    )
    gui_convert = StableDiffusionUI_convert(
        **config['convert']
    )
    gui_inpaint = gui_img2img