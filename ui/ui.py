# Code credits to 凉心半浅良心人
# Has modified

import os
os.environ['PPNLP_HOME'] = "./model_weights"
import time
from IPython.display import clear_output, display
from .png_info_helper import serialize_to_pnginfo, imageinfo_to_pnginfo
from .views import createView

from .env import DEBUG_UI

if not DEBUG_UI:

    from .utils import diffusers_auto_update
    diffusers_auto_update()

    #from tqdm.auto import tqdm
    import paddle
    from .utils import StableDiffusionFriendlyPipeline, SuperResolutionPipeline, diffusers_auto_update
    from .utils import compute_gpu_memory, empty_cache
    from .utils import save_image_info
    from .convert import parse_args as convert_parse_args
    from .convert import main as convert_parse_main

    #_ENABLE_ENHANCE = False

    if paddle.device.get_device() != 'cpu':
        # settings for super-resolution, currently not supporting multi-gpus
        # see docs at https://github.com/PaddlePaddle/PaddleHub/tree/develop/modules/image/Image_editing/super_resolution/falsr_a
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    pipeline_superres = SuperResolutionPipeline()
    pipeline = StableDiffusionFriendlyPipeline(superres_pipeline = pipeline_superres)
else:
    pipeline_superres = None
    pipeline = None

####################################################################
#
#                     Graphics User Interface
#
####################################################################
# Code to turn kwargs into Jupyter widgets
import ipywidgets as widgets
from ipywidgets import Layout,HBox,VBox,Box
from collections import OrderedDict


# Allows long widget descriptions
style  = {'description_width': 'initial'}
# Force widget width to max
layout = widgets.Layout(width='100%')

def get_widget_extractor(widget_dict):
    # allows accessing after setting, this is to reduce the diff against the argparse code
    class WidgetDict(OrderedDict):
        def __getattr__(self, val):
            x = self.get(val)
            return x.value if x is not None else None
    return WidgetDict(widget_dict)


class StableDiffusionUI():
    def __init__(self, pipeline = pipeline):
        self.widget_opt = OrderedDict()
        self.pipeline = pipeline
        self.gui = None
        self.run_button = None
        self.run_button_out = widgets.Output()
        self.task = 'txt2img'

    def on_run_button_click(self, b):
        with self.run_button_out:
            clear_output()
            self.pipeline.run(
                get_widget_extractor(self.widget_opt), 
                task = self.task,
                on_image_generated = self.on_image_generated
            )
    
    def on_image_generated(self, image, options,  count = 0, total = 1, image_info = None):
        
        # 超分
        # --------------------------------------------------
        if self.task == 'superres':
            cur_time = time.time()
            os.makedirs(options.output_dir, exist_ok = True)
            image_path = os.path.join(
                options.output_dir,
                time.strftime(f'%Y-%m-%d_%H-%M-%S_Highres.png')
            )
            image.save(
                image_path,
                quality=100,
                pnginfo = imageinfo_to_pnginfo(image_info) if image_info is not None else None
            )
            clear_output()
            display(widgets.Image.from_file(image_path))
            return
        
        # 图生图/文生图
        # --------------------------------------------------
        image_path = save_image_info(image, options.output_dir, image_info)
        if count % 5 == 0:
            clear_output()
        
        try:
            # 使显示的图片包含嵌入信息
            display(widgets.Image.from_file(image_path))
        except:
            display(image)
        
        if 'seed' in image.argument['seed']:
            print('Seed = ', image.argument['seed'], 
                '    (%d / %d ... %.2f%%)'%(count + 1, total, (count + 1.) / total * 100))

#####################################
#M0DE1 C0NVERT
##############################
class StableDiffusionConvertUI():
    def __init__(self, pipeline = pipeline):
        self.widget_opt = OrderedDict()
        self.gui = None
        self.run_button = None
        self.run_button_out = widgets.Output()
        self.pipeline = pipeline

        # function pointers
        self.parse_args = None
        self.main = None

    def run(self, opt):
        args = self.parse_args()
        for k, v in opt.items():
            setattr(args, k, v.value)

        if args.checkpoint_path is None:
            raise ValueError("你必须给出一个可用的ckpt模型路径")
        self.main(args)
        empty_cache()
        
    def on_run_button_click(self, b):
        with self.run_button_out:
            clear_output()
            self.run(get_widget_extractor(self.widget_opt))

class StableDiffusionUI_convert(StableDiffusionConvertUI):
    def __init__(self, **kwargs):
        super().__init__()
        self.parse_args = convert_parse_args #配置加载
        self.main = convert_parse_main
        args = {  #注意无效Key错误
            "checkpoint_path": '',
            'dump_path': 'outputs/convert'
        }
        args.update(kwargs)
        
        layoutCol12 = Layout(
            flex = "12 12 90%",
            margin = "0.5em",
            max_width = "100%",
            align_items = "center"
        )
        styleDescription = {
            'description_width': "10rem"
        }

        widget_opt = self.widget_opt
        widget_opt['checkpoint_path'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='ckpt模型文件位置',
            description_tooltip='你要转换的模型位置',
            value=" ",
            disabled=False
        )
        widget_opt['dump_path'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='输出目录',
            description_tooltip='转换模型输出的地方',
            value="outputs/convert",
            disabled=False
        )

        for key in widget_opt:
            if (key in args) and (args[key] != widget_opt[key].value):
                widget_opt[key].value = args[key]

               
        self.run_button = widgets.Button(
            description='开始转换',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='点击运行（配置将自动更新）',
            icon='check'
        )
        self.run_button.on_click(self.on_run_button_click)
        
        self.gui = Box([
                Box([
                    widget_opt['checkpoint_path'],
                    widget_opt['dump_path'],
                    
                ], layout = Layout(
                    display = "flex",
                    flex_flow = "row wrap", #HBox会覆写此属性
                    align_items = "center",
                    max_width = '100%',
                )),
                self.run_button, 
                self.run_button_out
            ], layout = Layout(display="block",margin="0 45px 0 0")
        )

    def on_run_button_click(self, b):
        with self.run_button_out:
            self.run_button.disabled = True
            clear_output()
            try:
                print('开始处理...')
                self.run(get_widget_extractor(self.widget_opt))
            finally:
                self.run_button.disabled = False
