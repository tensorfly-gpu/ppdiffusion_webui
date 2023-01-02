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

    from .textual_inversion import parse_args as textual_inversion_parse_args
    from .textual_inversion import main as textual_inversion_main
    from .dreambooth import parse_args as dreambooth_parse_args
    from .dreambooth import main as dreambooth_main
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
        widget_opt['pretrained_model_name_or_path'] = widgets.Combobox(
            layout=layoutCol12, style=styleDescription,
            description='需要训练的模型名称',
            value="MoososCap/NOVEL-MODEL",
            options=model_name_list,
            ensure_option=False,
            disabled=False
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

