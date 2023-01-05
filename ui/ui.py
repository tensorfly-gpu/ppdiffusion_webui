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
    from .utils import collect_local_ckpts
    from .convert import parse_args as convert_parse_args
    from .convert import main as convert_parse_main
    from .views import createView

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
        # self.parse_args = None
        # self.main = None

    def run(self, opt):
        args = convert_parse_args()
        for k, v in opt.items():
            setattr(args, k, v.value)

        if args.checkpoint_path is None:
            raise ValueError("你必须给出一个可用的ckpt模型路径")
        convert_parse_main(args)
        empty_cache()
        
    # def on_run_button_click(self, b):
        # with self.run_button_out:
            # clear_output()
            # self.run(get_widget_extractor(self.widget_opt))

class StableDiffusionUI_convert(StableDiffusionConvertUI):
    def __init__(self, **kwargs):
        super().__init__()
        args = {  #注意无效Key错误
            # "checkpoint_path": '',
            "model_root": 'models/',
            "scheduler_type": 'pndm',
            "extract_ema": False,
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
        widget_opt['extract_ema'] = widgets.Dropdown(
            layout=layoutCol12, style=styleDescription,
            description='提取Ema权重',
            description_tooltip = """Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights
EMA weights usually yield higher quality images for inference.
Non-EMA weights are usually better to continue fine-tuning.""",
            options=[
                ('开启',  True),
                ('关闭',  False),
            ],
        )
        widget_opt['scheduler_type'] = widgets.Dropdown(
            layout=layoutCol12, style=styleDescription,
            description='采样器类型',
            options=['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm'],
        )
        widget_opt['checkpoint_path'] = widgets.Combobox(
            layout=layoutCol12, style=styleDescription,
            description='ckpt模型文件位置',
            description_tooltip='你要转换的模型位置',
            options=collect_local_ckpts(),
        )
        widget_opt['model_root'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='输出目录',
            description_tooltip='转换模型输出的地方',
        )
        INVALID_PATH_TEXT = '（未输入模型或模型文件不存在）'
        widget_opt['dump_path'] =  widgets.HTML(
            layout=layoutCol12, style=styleDescription,
            description='输出到：',
            value = INVALID_PATH_TEXT,
        )
        
        for key in widget_opt:
            if (key in args) and (args[key] != widget_opt[key].value):
                widget_opt[key].value = args[key]
        
        # 事件处理绑定
        def on_select_model(change):
            filepath = change.new
            if not os.path.isfile(filepath):
                self.run_button.disabled = True
                widget_opt['dump_path'].value = INVALID_PATH_TEXT
            else:
                self.run_button.disabled = False
                widget_opt['dump_path'].value = os.path.join(
                    widget_opt['model_root'].value,
                    os.path.basename(filepath).rpartition('.')[0]
                )
        widget_opt['checkpoint_path'].observe(on_select_model, names='value')
        
        self.run_button = createView('run_button',
            description = '开始转换',
            tooltip = '点击开始转换模型',
            disabled = True,
        )
        self.run_button.on_click(self.on_run_button_click)
        
        self.gui = createView("box_gui", 
            children = [
                createView("box_main", 
                [
                    widget_opt['checkpoint_path'],
                    widget_opt['scheduler_type'],
                    widget_opt['extract_ema'],
                    widget_opt['model_root'],
                    widget_opt['dump_path'],
                ]),
                self.run_button, 
                self.run_button_out
            ]
        )

    def on_run_button_click(self, b):
        with self.run_button_out:
            clear_output()
            assert os.path.isfile(self.widget_opt['dump_path'].value), '未输入模型或模型文件不存在'
            try:
                print('开始处理...')
                self.run_button.disabled = True
                self.run(get_widget_extractor(self.widget_opt))
            finally:
                self.run_button.disabled = False
