import os
from .ui import pipeline, get_widget_extractor
from .dreambooth import parse_args as dreambooth_parse_args
from .dreambooth import main as dreambooth_main

from IPython.display import clear_output, display
import ipywidgets as widgets
from ipywidgets import Layout,HBox,VBox,Box
from . import views

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
        self.parse_args = None
        self.main = None

    def run(self, opt):
        args = self.parse_args()
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
        self.main(args)
        empty_cache()
        
    def on_run_button_click(self, b):
        with self.run_button_out:
            clear_output()
            self.run(get_widget_extractor(self.widget_opt))

class StableDiffusionUI_dreambooth(StableDiffusionDreamboothUI):
    def __init__(self, **kwargs):
        super().__init__()
        self.parse_args = dreambooth_parse_args #配置加载
        self.main = dreambooth_main
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
            "learning_rate": 5e-4,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 500,
            "center_crop": True,
            "output_dir": 'outputs/dreambooth',
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
        widget_opt['pretrained_model_name_or_path'] = createView(
            'model_name',
            layout_name='col12', style=styleDescription,
            description='训练所使用模型的名称（清空输入框以显示更多模型）',
        )
        widget_opt['instance_data_dir'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='实例（物体）图片文件夹地址。',
            description_tooltip='你要训练的特殊图片目录（人物，背景等）',
            value="resources/Alices",
            disabled=False
        )
        widget_opt['instance_prompt'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='实例（物体）的提示词描述文本',
            description_tooltip='带有特定实例 物体的提示词描述文本例如『a photo of sks dog』其中dog代表实例物体。',
            value="a photo of Alices",
            disabled=False
        )
        widget_opt['class_data_dir'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='类别 class 图片文件夹地址',
            description_tooltip='类别 class 图片文件夹地址，这个文件夹里可以不放东西，会自动生成',
            value="resources/Girls",
            disabled=False
        )
        widget_opt['class_prompt'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='类别class提示词文本',
            description_tooltip='该提示器要与实例物体是同一种类别 例如『a photo of dog』',
            value="a photo of girl",
            disabled=False
        )
        widget_opt['num_class_images'] = widgets.IntText(
            layout=layoutCol12, style=styleDescription,
            description='类别class提示词对应图片数',
            description_tooltip='如果文件夹里图片不够会自动补全',
            value=100,
            disabled=False
        )
        widget_opt['with_prior_preservation'] = widgets.Dropdown(
            layout=layoutCol12, style=styleDescription,
            description='是否将生成的同类图片（先验知识）一同加入训练',
            description_tooltip='当开启的时候 上面的设置才生效。',
            value=True,
            options=[
                ('开启',  True),
                ('关闭',  False),
            ],
            disabled=False
        )
        widget_opt['prior_loss_weight'] = widgets.FloatText(
            layout=layoutCol12, style=styleDescription,
            description='先验loss占比权重',
            description_tooltip='不用改',
            value=1.0,
            disabled=False
        )
        '''widget_opt['num_train_epochs'] = widgets.IntText(
            layout=layoutCol12, style=styleDescription,
            description='训练的轮数',
            description_tooltip='与最大训练步数互斥',
            value=1,
            disabled=False
        )'''
        widget_opt['max_train_steps'] = widgets.IntText(
            layout=layoutCol12, style=styleDescription,
            description='最大训练步数',
            description_tooltip='当我们设置这个值后它会重新计算所需的轮数',
            value=1000,
            disabled=False
        )
        widget_opt['save_steps'] = widgets.IntText(
            layout=layoutCol12, style=styleDescription,
            description='模型保存步数',
            description_tooltip='达到这个数后会保存模型',
            value=1000,
            disabled=False
        )
        widget_opt['train_text_encoder'] = widgets.Dropdown(
            layout=layoutCol12, style=styleDescription,
            description='是否一同训练文本编码器的部分',
            description_tooltip='可以理解为是否同时训练textual_inversion',
            value=False,
            options=[
                ('开启',  True),
                ('关闭',  False),
            ],
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
            value=5e-6,
            step=1e-6,
            disabled=False
        )
        widget_opt['lr_scheduler'] = widgets.Dropdown(
            layout=layoutCol12, style=styleDescription,
            description='学习率调度策略',
            description_tooltip='可以选不同的学习率调度策略',
            value='constant',
            options=[
                ('linear',  "linear"),
                ('cosine',  "cosine"),
                ('cosine_with_restarts',  "cosine_with_restarts"),
                ('polynomial',  "polynomial"),
                ('constant',  "constant"),
                ('constant_with_warmup',  "constant_with_warmup"),
            ],
            disabled=False
        )
        widget_opt['lr_warmup_steps'] = widgets.IntText(
            layout=layoutCol12, style=styleDescription,
            description='线性 warmup 的步数',
            description_tooltip='用于从 0 到 learning_rate 的线性 warmup 的步数。',
            value=500,
            disabled=False
        )
        widget_opt['center_crop'] = widgets.Dropdown(
            layout=layoutCol12, style=styleDescription,
            description='自动裁剪图片时将人像居中',
            description_tooltip='自动裁剪图片时将人像居中',
            value=False,
            options=[
                ('开启',  True),
                ('关闭',  False),
            ],
            disabled=False
        )
        widget_opt['output_dir'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='输出目录',
            description_tooltip='训练好模型输出的地方',
            value="outputs/dreambooth",
            disabled=False
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
                    widget_opt['pretrained_model_name_or_path'],
                    widget_opt['instance_data_dir'],
                    widget_opt['instance_prompt'],
                    widget_opt['class_data_dir'],
                    widget_opt['class_prompt'],
                    widget_opt['num_class_images'],
                    widget_opt['prior_loss_weight'],
                    widget_opt['with_prior_preservation'],
                    #widget_opt['num_train_epochs'],
                    widget_opt['max_train_steps'],
                    widget_opt['save_steps'],
                    widget_opt['train_text_encoder'],
                    widget_opt['height'],
                    widget_opt['width'],
                    widget_opt['learning_rate'],
                    widget_opt['lr_scheduler'],
                    widget_opt['lr_warmup_steps'],
                    widget_opt['center_crop'],
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

