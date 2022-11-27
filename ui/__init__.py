
from .ui import (
    StableDiffusionUI_text_inversion,
    StableDiffusionUI_text_inversion_prediction,
    
    pipeline_superres,
    pipeline,
)
from .StableDiffusionUI_txt2img import StableDiffusionUI_txt2img
from .StableDiffusionUI_img2img import StableDiffusionUI_img2img
from .StableDiffusionUI_inpaint import StableDiffusionUI_inpaint
from .SuperResolutionUI import SuperResolutionUI


gui_txt2img = StableDiffusionUI_txt2img()
gui_text_inversion = StableDiffusionUI_text_inversion_prediction()
gui_img2img = StableDiffusionUI_img2img()
gui_inpaint = StableDiffusionUI_inpaint()
gui_superres = SuperResolutionUI(pipeline = pipeline_superres)
gui_train_text_inversion = StableDiffusionUI_text_inversion()