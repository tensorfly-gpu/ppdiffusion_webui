import os
from .ui import pipeline, get_widget_extractor
from .textual_inversion import parse_args as textual_inversion_parse_args
from .textual_inversion import main as textual_inversion_main

from IPython.display import clear_output, display
import ipywidgets as widgets
from ipywidgets import Layout,HBox,VBox,Box
from . import views

####################################################################
#
#                            Training
#
####################################################################
     
class StableDiffusionTrainUI():
    def __init__(self, pipeline = pipeline):
        self.widget_opt = OrderedDict()
        self.gui = None
        self.run_button = None
        self.run_button_out = widgets.Output()
        self.pipeline = pipeline

        # function pointers
        #self.parse_args = None #Test
        self.main = None

    def run(self, opt):
        args = self.parse_args()
        for k, v in opt.items():
            setattr(args, k, v.value)

        self.pipeline.from_pretrained(model_name=opt.model_name)
        
        # todo junnyu
        args.pretrained_model_name_or_path = opt.model_name
        if args.language == "en":
            if "chinese-en" in args.pretrained_model_name_or_path.lower():
                args.language = "zh_en"
            elif "chinese" in args.pretrained_model_name_or_path.lower():
                args.language = "zh"
        if args.image_logging_prompt is None:
            args.image_logging_prompt = args.placeholder_token
        ## done

        args.text_encoder = self.pipeline.pipe.text_encoder
        args.unet         = self.pipeline.pipe.unet
        args.vae          = self.pipeline.pipe.vae

        if compute_gpu_memory() <= 17. or args.height==768 or args.width==768:
            args.train_batch_size = 1
            args.gradient_accumulation_steps = 4
        else:
            args.train_batch_size = 4
            args.gradient_accumulation_steps = 1

        if args.pretrained_model_name_or_path in ["stabilityai/stable-diffusion-2", "stabilityai/stable-diffusion-2-base", "BAAI/AltDiffusion", "BAAI/AltDiffusion-m9"]:
            args.train_batch_size = 1
            args.gradient_accumulation_steps = 4

        if args.train_data_dir is None:
            raise ValueError("You must specify a train data directory.")
        
        # remove \/"*:?<>| in filename
        name = args.placeholder_token
        name = name.translate({92: 95, 47: 95, 42: 95, 34: 95, 58: 95, 63: 95, 60: 95, 62: 95, 124: 95})
        args.logging_dir  = os.path.join(args.output_dir, 'logs', name)

        self.main(args)
        empty_cache()
        
    def on_run_button_click(self, b):
        with self.run_button_out:
            clear_output()
            self.run(get_widget_extractor(self.widget_opt))
            
class StableDiffusionUI_text_inversion(StableDiffusionTrainUI):
    def __init__(self, **kwargs):
        super().__init__()
        self.parse_args = textual_inversion_parse_args
        self.main = textual_inversion_main
        
        #默认参数覆盖次序：
        #user_config.py > config.py > 当前args > 实例化
        args = {  #注意无效Key错误
            "learnable_property": 'object',
            "placeholder_token": '<Alice>',
            "initializer_token": 'girl',
            "repeats": '100',
            "train_data_dir": 'resources/Alices',
            "output_dir": 'outputs/textual_inversion',
            "height": 512,
            "width": 512,
            "learning_rate": 5e-6,
            "max_train_steps": 1000,
            "save_steps": 200,
            "model_name": "MoososCap/NOVEL-MODEL",
        }
        args.update(kwargs)
        
        layoutCol12 = Layout(
            flex = "12 12 90%",
            margin = "0.375rem",
            max_width = "calc(100% - 0.75rem)",
            align_items = "center"
        )
        styleDescription = {
            'description_width': "10rem"
        }
        
        widget_opt = self.widget_opt
        widget_opt['learnable_property'] = widgets.Dropdown(
            layout=layoutCol12, style=styleDescription,
            description='训练目标',
            description_tooltip='训练目标是什么？风格还是实体？',
            value="object",
            options=[
                ('风格（style）',  "style"),
                ('实体（object）', "object"),
            ],
            orientation='horizontal',
            disabled=False
        )
        widget_opt['placeholder_token'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='用来表示该内容的新词',
            description_tooltip='用来表示该内容的新词，建议用<>封闭',
            value="<Alice>",
            disabled=False
        )
        widget_opt['initializer_token'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='该内容最接近的单词是',
            description_tooltip='该内容最接近的单词是？若无则用*表示',
            value="girl",
            disabled=False
        )
        widget_opt['repeats'] = widgets.IntText(
            layout=layoutCol12, style=styleDescription,
            description='图片重复次数',
            description_tooltip='训练图片需要重复多少遍',
            value="100",
            disabled=False
        )
        widget_opt['train_data_dir'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='训练图片的文件夹路径',
            value="resources/Alices",
            disabled=False
        )
        widget_opt['output_dir'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='训练结果的保存路径',
            value="outputs/textual_inversion",
            disabled=False
        )
        widget_opt['height'] = widgets.IntSlider(
            layout=layoutCol12, style=styleDescription,
            description='训练图片的高度',
            description_tooltip='训练图片的高度。越大尺寸，消耗的显存也越多。',
            value=512,
            min=64,
            max=1024,
            step=64,
            disabled=False
        )
        widget_opt['width'] = widgets.IntSlider(
            layout=layoutCol12, style=styleDescription,
            description='训练图片的宽度',
            description_tooltip='训练图片的宽度。越大尺寸，消耗的显存也越多。',
            value=512,
            min=64,
            max=1024,
            step=64,
            disabled=False
        )
        widget_opt['learning_rate'] = widgets.FloatText(
            layout=layoutCol12, style=styleDescription,
            description='训练学习率',
            description_tooltip='训练学习率',
            value=5e-4,
            step=1e-4,
            disabled=False
        )
        widget_opt['max_train_steps'] = widgets.IntText(
            layout=layoutCol12, style=styleDescription,
            description='最大训练步数',
            description_tooltip='最大训练步数',
            value=1000,
            step=100,
            disabled=False
        )
        widget_opt['save_steps'] = widgets.IntText(
            layout=layoutCol12, style=styleDescription,
            description='每隔多少步保存模型',
            value=200,
            step=100,
            disabled=False
        )
        widget_opt['model_name'] = createView(
            'model_name',
            layout_name='col12', style=styleDescription,
            description='训练所使用模型的名称（清空输入框以显示更多模型）',
        )
        
        for key in widget_opt:
            if (key in args) and (args[key] != widget_opt[key].value):
                widget_opt[key].value = args[key]
        
        self.run_button = widgets.Button(
            description='开始训练',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='点击运行（配置将自动更新）',
            icon='check'
        )
        self.run_button.on_click(self.on_run_button_click)
        
        self.gui = Box([
                Box([
                    widget_opt['learnable_property'],
                    widget_opt['placeholder_token'],
                    widget_opt['initializer_token'],
                    widget_opt['train_data_dir'],
                    widget_opt['width'],
                    widget_opt['height'],
                    widget_opt['repeats'],
                    widget_opt['learning_rate'],
                    widget_opt['max_train_steps'],
                    widget_opt['save_steps'],
                    widget_opt['model_name'],
                    widget_opt['output_dir'],
                    
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

