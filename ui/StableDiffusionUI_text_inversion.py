import os
import ipywidgets as widgets
from collections import OrderedDict
from IPython.display import clear_output
from .ui import pipeline, get_widget_extractor
from .utils import compute_gpu_memory, empty_cache
from .textual_inversion import parse_args as textual_inversion_parse_args
from .textual_inversion import main as textual_inversion_main
from .views import createView, setLayout, SHARED_STYLE_SHEETS

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
        #self.main = None

    def run(self, opt):
        args = textual_inversion_parse_args()
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

        textual_inversion_main(args)
        empty_cache()
        
    def on_run_button_click(self, b):
        with self.run_button_out:
            clear_output()
            self.run(get_widget_extractor(self.widget_opt))
            
class StableDiffusionUI_text_inversion(StableDiffusionTrainUI):
    def __init__(self, **kwargs):
        super().__init__()
        
        CLASS_NAME = self.__class__.__name__ \
                + '_{:X}'.format(hash(self))[-4:]
        
        STYLE_SHEETS = '''

'''
        
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
        
        widget_opt = self.widget_opt
        widget_opt['learnable_property'] = widgets.Dropdown(
            description='训练目标',
            description_tooltip='训练目标是什么？风格还是实体？',
            options=[
                ('风格（style）',  "style"),
                ('实体（object）', "object"),
            ],
        )
        widget_opt['placeholder_token'] = widgets.Text(
            description='用来表示该内容的新词',
            description_tooltip='用来表示该内容的新词，建议用<>封闭',
        )
        widget_opt['initializer_token'] = widgets.Text(
            description='该内容最接近的单词是',
            description_tooltip='该内容最接近的单词是？若无则用*表示',
        )
        widget_opt['repeats'] = widgets.IntText(
            description='图片重复次数',
            description_tooltip='训练图片需要重复多少遍',
        )
        widget_opt['train_data_dir'] = widgets.Text(
            description='训练图片的文件夹路径',
        )
        widget_opt['output_dir'] = widgets.Text(
            description='训练结果的保存路径',
        )
        widget_opt['height'] = widgets.IntSlider(
            description='训练图片的高度',
            description_tooltip='训练图片的高度。越大尺寸，消耗的显存也越多。',
            value=512,
            min=64,
            max=1024,
            step=64,
        )
        widget_opt['width'] = widgets.IntSlider(
            description='训练图片的宽度',
            description_tooltip='训练图片的宽度。越大尺寸，消耗的显存也越多。',
            value=512,
            min=64,
            max=1024,
            step=64,
        )
        widget_opt['learning_rate'] = widgets.FloatText(
            description='训练学习率',
            description_tooltip='训练学习率',
            value=5e-4,
            step=1e-4,
        )
        widget_opt['max_train_steps'] = widgets.IntText(
            description='最大训练步数',
            description_tooltip='最大训练步数',
            value=1000,
            step=100,
        )
        widget_opt['save_steps'] = widgets.IntText(
            description='每隔多少步保存模型',
            value=200,
            step=100,
        )
        widget_opt['model_name'] = createView(
            'model_name',
            layout_name='col12',
            description='训练所使用模型的名称（清空输入框以显示更多模型）',
        )
        
        _col12 = (
            # 'learnable_property',
            # 'initializer_token',
            # 'placeholder_token',
            # 'train_data_dir',
            'output_dir',
            'model_name',
        )
        for key in widget_opt:
            if key in _col12:
                setLayout(('col12','widget-wrap'), widget_opt[key])
            else:
                setLayout(('col06','widget-wrap'), widget_opt[key])
            if (key in args) and (args[key] != widget_opt[key].value):
                widget_opt[key].value = args[key]
        
        # 按钮
        self.run_button = createView('train_button')
        self.run_button.on_click(self.on_run_button_click)
        
        # 样式表
        STYLE_SHEETS = ('<style>' \
                + SHARED_STYLE_SHEETS \
                + STYLE_SHEETS \
                + '</style>'
            ).replace('{root}', '.' + CLASS_NAME)
        
        self.gui = createView("box_gui", 
            class_name = CLASS_NAME,
            children = [
                widgets.HTML(STYLE_SHEETS),
                createView("box_main", 
                [
                    widget_opt['learnable_property'],
                    widget_opt['train_data_dir'],
                    widget_opt['placeholder_token'],
                    widget_opt['initializer_token'],
                    widget_opt['width'],
                    widget_opt['height'],
                    widget_opt['repeats'],
                    widget_opt['learning_rate'],
                    widget_opt['max_train_steps'],
                    widget_opt['save_steps'],
                    widget_opt['model_name'],
                    widget_opt['output_dir'],
                ]),
                self.run_button, 
                self.run_button_out
            ],
        )

