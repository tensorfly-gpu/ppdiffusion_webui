import os
import ipywidgets as widgets
from ipywidgets import Layout
from collections import OrderedDict
from IPython.display import clear_output
from .ui import pipeline, get_widget_extractor
from .utils import empty_cache, collect_local_ckpts
from .convert import parse_args as convert_parse_args
from .convert import main as convert_parse_main
from .views import createView
from .model_collection import model_collection

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
        
        model_collection.record_model_name(args.dump_path)
        model_collection.refresh_locals()
        
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
            "vae_checkpoint_path": '',
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
        widget_opt['vae_checkpoint_path'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='vae文件位置',
            description_tooltip='你要转换的vae模型位置',
            value=" ",
            disabled=False
        )
        widget_opt['model_root'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='输出目录',
            description_tooltip='转换模型输出的地方',
        )
        INVALID_PATH_TEXT = '（未输入ckpt模型或文件不存在）'
        widget_opt['dump_path'] =  widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='输出模型',
            description_tooltip='转换后的模型。（整个文件夹即为模型）',
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
                    widget_opt['vae_checkpoint_path'],
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
            assert os.path.isfile(self.widget_opt['checkpoint_path'].value), '未输入ckpt模型或文件不存在'
            try:
                print('开始处理...')
                self.run_button.disabled = True
                self.run(get_widget_extractor(self.widget_opt))
            finally:
                self.run_button.disabled = False
