import os
import ipywidgets as widgets
from collections import OrderedDict
from IPython.display import clear_output
from .ui import pipeline, get_widget_extractor
from .utils import empty_cache
from .dreambooth import parse_args as dreambooth_parse_args
from .dreambooth import main as dreambooth_main
from .views import createView, setLayout, SHARED_STYLE_SHEETS

####################################################################
#
#                            Training
#
####################################################################
        
#Dreambooth训练
class StableDiffusionDreamboothUI():
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
        args = dreambooth_parse_args()
        for k, v in opt.items():
            setattr(args, k, v.value)

        args.train_batch_size = 1
        args.gradient_accumulation_steps = 1
        
        if args.instance_data_dir is None:
            raise ValueError("You must specify a train data directory.")
        
        # remove \/"*:?<>| in filename
        name = args.instance_prompt
        name = name.translate({92: 95, 47: 95, 42: 95, 34: 95, 58: 95, 63: 95, 60: 95, 62: 95, 124: 95})
        args.logging_dir  = os.path.join(args.output_dir, 'logs', name)
        dreambooth_main(args)
        empty_cache()
        
    def on_run_button_click(self, b):
        with self.run_button_out:
            clear_output()
            self.run(get_widget_extractor(self.widget_opt))

class StableDiffusionUI_dreambooth(StableDiffusionDreamboothUI):
    def __init__(self, **kwargs):
        super().__init__()
        
        CLASS_NAME = self.__class__.__name__ \
                + '_{:X}'.format(hash(self))[-4:]
        
        STYLE_SHEETS = '''

'''        
        args = {  #注意无效Key错误
            "pretrained_model_name_or_path": "MoososCap/NOVEL-MODEL",# 预训练模型名称/路径
            "instance_data_dir": 'resources/Alices',
            "instance_prompt": 'a photo of Alices',
            "class_data_dir": 'resources/Girls',
            "class_prompt": 'a photo of girl',
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
        }
        args.update(kwargs)

        widget_opt = self.widget_opt
        widget_opt['pretrained_model_name_or_path'] = createView(
            'model_name',
            description='训练所使用模型的名称（清空输入框以显示更多模型）',
        )
        widget_opt['instance_data_dir'] = widgets.Text(
            description='实例（物体）的训练图片目录',
            description_tooltip='你要训练的特殊图片目录（人物，背景等）',
        )
        widget_opt['instance_prompt'] = widgets.Text(
            description='实例（物体）的提示词',
            description_tooltip='带有特定实例 物体的提示词描述文本例如『a photo of sks dog』其中dog代表实例物体。',
        )
        widget_opt['class_data_dir'] = widgets.Text(
            description='类别（Class）的训练图片目录',
            description_tooltip='类别 class 图片文件夹地址，这个文件夹里可以不放东西，会自动生成',
        )
        widget_opt['class_prompt'] = widgets.Text(
            description='类别（Class）的提示词',
            description_tooltip='该提示器要与实例物体是同一种类别 例如『a photo of dog』',
        )
        widget_opt['num_class_images'] = widgets.IntText(
            description='类别提示词对应的图片数',
            description_tooltip='如果文件夹里图片不够会自动补全',
        )
        widget_opt['with_prior_preservation'] = widgets.Dropdown(
            description='将生成的同类图片加入训练',
            description_tooltip='是否将生成的同类图片（先验知识）一同加入训练。当开启的时候 上面的设置才生效。',
            options=[
                ('开启',  True),
                ('关闭',  False),
            ],
        )
        widget_opt['prior_loss_weight'] = widgets.FloatText(
            description='先验loss占比权重',
            description_tooltip='不用改',
        )
        '''widget_opt['num_train_epochs'] = widgets.IntText(
            description='训练的轮数',
            description_tooltip='与最大训练步数互斥',
        )'''
        widget_opt['max_train_steps'] = widgets.IntText(
            description='最大训练步数',
            description_tooltip='当我们设置这个值后它会重新计算所需的轮数',
        )
        widget_opt['save_steps'] = widgets.IntText(
            description='模型保存步数',
            description_tooltip='达到这个数后会保存模型',
            value=1000,
        )
        widget_opt['train_text_encoder'] = widgets.Dropdown(
            description='同时训练文本编码器',
            description_tooltip='可以理解为是否同时训练textual_inversion',
            options=[
                ('开启',  True),
                ('关闭',  False),
            ],
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
            value=5e-6,
            step=1e-6,
        )
        widget_opt['lr_scheduler'] = widgets.Dropdown(
            description='学习率调度策略',
            description_tooltip='可以选不同的学习率调度策略',
            options=[
                ('linear',  "linear"),
                ('cosine',  "cosine"),
                ('cosine_with_restarts',  "cosine_with_restarts"),
                ('polynomial',  "polynomial"),
                ('constant',  "constant"),
                ('constant_with_warmup',  "constant_with_warmup"),
            ],
        )
        widget_opt['lr_warmup_steps'] = widgets.IntText(
            description='线性 warmup 的步数',
            description_tooltip='用于从 0 到 learning_rate 的线性 warmup 的步数。',
        )
        widget_opt['center_crop'] = widgets.Dropdown(
            description='自动裁剪图片时将人像居中',
            description_tooltip='自动裁剪图片时将人像居中',
            value=False,
            options=[
                ('开启',  True),
                ('关闭',  False),
            ],
        )
        widget_opt['output_dir'] = widgets.Text(
            description='输出目录',
            description_tooltip='训练好模型输出的地方',
        )

        _col04 = (
            "num_class_images",
            "prior_loss_weight",
            "with_prior_preservation",
            "num_train_epochs",
            "max_train_steps",
            "save_steps",
            "train_text_encoder",
            "learning_rate",
            "lr_scheduler",
            "lr_warmup_steps",
            "center_crop",
        )
        _col12 = (
            "output_dir",
            "pretrained_model_name_or_path",
        )
        
        for key in widget_opt:
            if key in _col04:
                setLayout(('col04','widget-wrap'), widget_opt[key])
            elif key in _col12:
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
                    widget_opt['instance_data_dir'],
                    widget_opt['instance_prompt'],
                    widget_opt['class_data_dir'],
                    widget_opt['class_prompt'],
                    widget_opt['width'],
                    widget_opt['height'],
                    widget_opt['num_class_images'],
                    widget_opt['prior_loss_weight'],
                    widget_opt['with_prior_preservation'],
                    #widget_opt['num_train_epochs'],
                    widget_opt['max_train_steps'],
                    widget_opt['save_steps'],
                    widget_opt['train_text_encoder'],
                    widget_opt['learning_rate'],
                    widget_opt['lr_scheduler'],
                    widget_opt['lr_warmup_steps'],
                    widget_opt['center_crop'],
                    widget_opt['pretrained_model_name_or_path'],
                    widget_opt['output_dir'],
                ]),
                self.run_button, 
                self.run_button_out
            ],
        )

