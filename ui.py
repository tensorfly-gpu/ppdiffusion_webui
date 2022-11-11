# Code credits to 凉心半浅良心人
# Has modified

import os
from IPython.display import clear_output

from utils import diffusers_auto_update
diffusers_auto_update(hint_kernel_restart = True)

from tqdm.auto import tqdm
import paddle

from textual_inversion import parse_args as textual_inversion_parse_args
from textual_inversion import main as textual_inversion_main
from utils import StableDiffusionFriendlyPipeline, SuperResolutionPipeline, diffusers_auto_update
from utils import compute_gpu_memory, empty_cache

_ENABLE_ENHANCE = False

if paddle.device.get_device() != 'cpu':
    # settings for super-resolution, currently not supporting multi-gpus
    # see docs at https://github.com/PaddlePaddle/PaddleHub/tree/develop/modules/image/Image_editing/super_resolution/falsr_a
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

pipeline_superres = SuperResolutionPipeline()
pipeline = StableDiffusionFriendlyPipeline(superres_pipeline = pipeline_superres)

####################################################################
#
#                     Graphics User Interface
#
####################################################################
# Code to turn kwargs into Jupyter widgets
import ipywidgets as widgets
from ipywidgets import Layout,TwoByTwoLayout,AppLayout, \
    HBox,VBox,Box
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
            self.pipeline.run(get_widget_extractor(self.widget_opt), task = self.task)
    

class StableDiffusionUI_txt2img(StableDiffusionUI):
    def __init__(self):
        super().__init__()
        layoutCol04 = Layout(
            flex = "4 4 30%",
            min_width = "160px",
            max_width = "100%",
            margin = "0.5em",
            align_items = "center"
        )
        layoutCol08 = Layout(
            flex = "8 8 60%",
            min_width = "320px",
            max_width = "100%",
            margin = "0.5em",
            align_items = "center"
        )
        styleDescription = {
            'description_width': "4rem"
        }
        DEFAULT_BADWORDS = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
        
        widget_opt = self.widget_opt
        # self.hboxes = {}
        
        widget_opt['prompt'] = widgets.Textarea(
            layout=Layout(
                flex = "1",
                min_height="12em",
                max_width = "100%",
                margin = "0.5em",
                align_items = 'stretch'
                ),
            style=styleDescription,
            description='正面描述' ,
            description_tooltip="仅支持(xxx)、(xxx:1.2)、[xxx]三种语法。设置括号格式可以对{}进行转换。",
            value="extremely detailed CG unity 8k wallpaper,black long hair,cute face,1 adult girl,happy, green skirt dress, flower pattern in dress,solo,green gown,art of light novel,in field",
            disabled=False
        )
        widget_opt['negative_prompt'] = widgets.Textarea(
            layout=Layout(
                flex = "1",
                #min_height="6em",
                max_width = "100%",
                margin = "0.5em",
                align_items = 'stretch'
                ),
            style=styleDescription,
            description='负面描述',
            description_tooltip="使生成图像的内容远离负面描述的文本",
            value=DEFAULT_BADWORDS,
            disabled=False
        )
        widget_opt['standard_size'] = widgets.Dropdown(
            layout=layoutCol04, style=styleDescription,
            description='图片尺寸',
            description_tooltip='生成图片的尺寸',
            value=5120512,
            options=[
                ('竖向（512x768）',            5120768),
                ('横向（768x512）',            7680512),
                ('正方形（640x640）',          6400640),
                ('大尺寸-竖向（512x1024）',    5121024),
                ('大尺寸-横向（1024x512）',   10240512),
                ('大尺寸-正方形（1024x1024）',10241024),
                ('小尺寸-竖向（384x640）',     3840640),
                ('小尺寸-横向（640x384）',     6400384),
                ('小尺寸-正方形（512x512）',   5120512),
            ],
            disabled=False
        )
        widget_opt['width'] = widgets.BoundedIntText(
            layout=Layout(
                flex = '1 0 2em'
            ),
            value=512,
            min=64,
            max=1024,
            step=64,
            disabled=False
        )
        widget_opt['height'] = widgets.BoundedIntText(
            layout=widget_opt['width'].layout,
            value=512,
            min=64,
            max=1024,
            step=64,
            disabled=False
        )
        
        widget_opt['num_return_images'] = widgets.IntText(
            layout=layoutCol04,
            style=styleDescription,
            description='生成数量',
            description_tooltip='生成图片的数量',
            value=1,
            disabled=False
        )
        widget_opt['enable_parsing'] = widgets.Dropdown(
            layout=layoutCol04,
            style=styleDescription,
            description='括号格式',
            description_tooltip='增加权重所用括号的格式，可以将{}替换为()。选择“否”则不解析加权语法',
            value="圆括号 () 加强权重",
            options=["圆括号 () 加强权重","花括号 {} 加权权重", "否"],
            disabled=False
        )
        

        widget_opt['num_inference_steps'] = widgets.IntSlider(
            layout=layoutCol04,
            style=styleDescription,
            description='推理步数',
            description_tooltip='推理步数（Step）：生成图片的迭代次数，步数越多运算次数越多。',
            value=50,
            min=2,
            max=80,
            orientation='horizontal',
            readout=True,
            disabled=False
        )
        widget_opt['guidance_scale'] = widgets.BoundedFloatText(
            layout=layoutCol04,
            style=styleDescription,
            description= 'CFG',
            description_tooltip='引导度（CFG Scale）：控制图片与描述词之间的相关程度。',
            min=0,
            max=100,
            value=7.5,
            disabled=False
        )

        widget_opt['max_embeddings_multiples'] = widgets.Dropdown(
            layout=layoutCol04,
            style=styleDescription,
            description='上限倍数',
            description_tooltip='修改长度上限倍数，使模型能够输入更长更多的描述词。',
            value="3",
            options=["1","2","3","4","5"],
            disabled=False
        )
        widget_opt['fp16'] = widgets.Dropdown(
            layout=layoutCol04,
            style=styleDescription,
            description='算术精度',
            description_tooltip='模型推理使用的精度。选择float16可以加快模型的推理速度，但会牺牲部分的模型性能。',
            value="float32",
            options=["float32", "float16"],
            disabled=False
        )
        
        widget_opt['seed'] = widgets.IntText(
            layout=layoutCol04,
            style=styleDescription,
            description='随机种子',
            description_tooltip='-1表示随机生成。',
            value=-1,
            disabled=False
        )
        widget_opt['superres_model_name'] = widgets.Dropdown(
            layout=layoutCol04,
            style=styleDescription,
            description='放大图片',
            description_tooltip='指定放大图片尺寸所用的模型',
            value="无",
            options=["falsr_a", "falsr_b", "falsr_c", "无"],
            disabled=False
        )
        
        widget_opt['output_dir'] = widgets.Text(
            layout=layoutCol08, style=styleDescription,
            description='保存路径',
            description_tooltip='用于保存输出图片的路径',
            value="outputs/txt2img",
            disabled=False
        )

        widget_opt['sampler'] = widgets.Dropdown(
            layout=layoutCol04, style=styleDescription,
            description='采样器',
            value="DDIM",
            options=["PNDM", "DDIM", "LMS"],
            disabled=False
        )
        widget_opt['model_name'] = widgets.Dropdown(
            layout=layoutCol04, style=styleDescription,
            description='模型名称',
            description_tooltip='需要加载的模型名称',
            value="MoososCap/NOVEL-MODEL",
            options=["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5", "hakurei/waifu-diffusion", "hakurei/waifu-diffusion-v1-3", "naclbit/trinart_stable_diffusion_v2_60k", "naclbit/trinart_stable_diffusion_v2_95k", "naclbit/trinart_stable_diffusion_v2_115k", "MoososCap/NOVEL-MODEL", "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1", "ruisi/anything"],
            disabled=False
        )

        widget_opt['concepts_library_dir'] = widgets.Text(
            layout=layoutCol08, style=styleDescription,
            description='风格权重',
            description_tooltip='TextualInversion训练的、“风格”或“人物”的权重文件路径',
            value="outputs/textual_inversion",
            disabled=False
        )
        
        def on_standard_dize_change(change):
            widget_opt['width'].value = change.new // 10000
            widget_opt['height'].value = change.new % 10000
        widget_opt['standard_size'].observe(
                on_standard_dize_change, 
                names = 'value'
        )
        
        def validate_width_height(change):
            num = change.new % 64
            if change.new < 64:
                change.owner.value = 64
            elif num == 0:
                pass
            elif num < 32:
                change.owner.value = change.new - num
            else:
                change.owner.value = change.new - num + 64
        widget_opt['width'].observe(
                validate_width_height,
                names = 'value'
            )
        widget_opt['height'].observe(
                validate_width_height,
                names = 'value'
            )
            
        labelSize = widgets.Label(
                value='X',
                layout = Layout(
                    flex='0 0 auto',
                    padding='0 1em'
                )
            )
        
        btnGoodQuality = widgets.Button(
            description= '',
            tooltip='填充标准质量描述',
            disabled=False,
            # button_style='primary',
            icon='palette',
            layout = Layout(
                height = '1.8rem',
                width = '1.8rem',
                margin = '-9rem 0 0 2rem'
            )
        )
        btnBadwards = widgets.Button(
            description= '',
            tooltip='填充标准负面描述',
            disabled=False,
            # button_style='primary',
            icon='paper-plane',
            layout = Layout(
                height = '1.8rem',
                width = '1.8rem',
                margin = '-2rem 0px 0rem -1.8rem'
            )
        )
        def fill_good_quality(b):
            if not widget_opt['prompt'].value.startswith('high quality,masterpiece,'):
                widget_opt['prompt'].value = 'high quality,masterpiece,' + widget_opt['prompt'].value
        def fill_bad_words(b):
            widget_opt['negative_prompt'].value = DEFAULT_BADWORDS
            
        btnGoodQuality.on_click(fill_good_quality)
        btnBadwards.on_click(fill_bad_words)
        
        
        self.run_button = widgets.Button(
            description='点击生成图片！',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='点击运行（会自动更新配置）',
            icon='check'
        )
        
        self.run_button.on_click(self.on_run_button_click)
        
        if _ENABLE_ENHANCE:
            widget_opt['image_dir'] = widgets.Text(
                layout=layout, style=style,
                description='需要放大图片的文件夹地址。',
                value="outputs/highres",
                disabled=False
            )
            widget_opt['upscale_image_dir'] = widgets.Text(
                layout=layout, style=style,
                description='放大后的图片所要保存的文件夹地址。',
                value="upscale_outputs",
                disabled=False
            )
            widget_opt['upscale_model_name'] = widgets.Dropdown(
                layout=layout, style=style,
                description='放大图片所用的模型',
                value="RealESRGAN_x4plus",
                options=[
                        'RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealSR',
                        'RealESRGAN_x4', 'RealESRGAN_x2', 'RealESRGAN_x8'
                    ],
                disabled=False
            )

            enhance_button = widgets.Button(
                description='开始放大图片！',
                disabled=False,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Click to run (settings will update automatically)',
                icon='check'
            )
            enhance_button_out = widgets.Output()
            def on_enhance_button_click(b):
                with run_button_out:
                    clear_output()
                with enhance_button_out:
                    clear_output()
                    enhance_run(get_widget_extractor(widget_opt))
            enhance_button.on_click(on_enhance_button_click)
    

        self.gui = Box([
                HBox([widget_opt['prompt']]),
                HBox([widget_opt['negative_prompt']]),
                Box([
                        btnGoodQuality, btnBadwards
                    ], layout = Layout(
                        height = '0',
                        overflow = 'visible'
                    )),
                Box([
                    widget_opt['standard_size'],
                    HBox([
                            widget_opt['width'],
                            labelSize,
                            widget_opt['height']
                        ], layout = layoutCol04
                    ),
                    widget_opt['superres_model_name'],
                    widget_opt['num_inference_steps'],
                    widget_opt['guidance_scale'],
                    widget_opt['sampler'],
                    widget_opt['num_return_images'],
                    widget_opt['seed'],
                    widget_opt['enable_parsing'],
                    widget_opt['max_embeddings_multiples'],
                    widget_opt['fp16'],
                    widget_opt['model_name'],
                    widget_opt['output_dir'],
                    widget_opt['concepts_library_dir']
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
    


class StableDiffusionUI_img2img(StableDiffusionUI):
    def __init__(self):
        super().__init__()

        widget_opt = self.widget_opt
        widget_opt['prompt'] = widgets.Textarea(
            layout=layout, style=style,
            description='prompt描述' + '&nbsp;' * 22,
            value="couple couple rings surface from (Starry Night of Van Gogh:1.1), couple couple rings in front of grey background, simple background, elegant style design, full display of fashion design",
            disabled=False
        )
        widget_opt['negative_prompt'] = widgets.Textarea(
            layout=layout, style=style,
            description='negative_prompt反面描述 <br />',
            value="",
            disabled=False
        )

        widget_opt['image_path'] = widgets.Text(
            layout=layout, style=style,
            description='需要转换的图片路径' + '&nbsp;'*12,
            value='resources/Ring.png',
            disabled=False
        )

        widget_opt['height'] = widgets.IntText(
            layout=layout, style=style,
            description='图片的高度, -1为自动判断' + '&nbsp;'*2,
            value=-1,
            disabled=False
        )
        
        widget_opt['width'] = widgets.IntText(
            layout=layout, style=style,
            description='图片的宽度, -1为自动判断' + '&nbsp;'*2,
            value=-1,
            disabled=False
        )

        widget_opt['num_return_images'] = widgets.BoundedIntText(
            layout=layout, style=style,
            description='生成图片数量' + '&nbsp;'*22,
            value=1,
            min=1,
            max=100,
            step=1,
            disabled=False
        )

        widget_opt['strength'] = widgets.BoundedFloatText(
            layout=layout, style=style,
            description='修改强度' + '&nbsp;'*29,
            value=0.8,
            min=0,
            max=1,
            step=0.1,
            disabled=False
        )

        widget_opt['num_inference_steps'] = widgets.IntText(
            layout=widgets.Layout(width='33%'), style=style,
            description='推理的步数' + '&nbsp'*24 ,
            value=50,
            disabled=False
        )

        widget_opt['guidance_scale'] = widgets.BoundedFloatText(
            layout=widgets.Layout(width='33%'), style=style,
            description= '&nbsp;'*22 + 'cfg',
            min=0,
            max=100,
            value=7.5,
            disabled=False
        )

        widget_opt['fp16'] = widgets.Dropdown(
            layout=widgets.Layout(width='33%'), style=style,
            description='&nbsp;'*15 + '精度' + '&nbsp;'*1,
            value="float32",
            options=["float32", "float16"],
            disabled=False
        )

        widget_opt['max_embeddings_multiples'] = widgets.Dropdown(
            layout=widgets.Layout(width='33%'), style=style,
            description='长度上限倍数' + '&nbsp;'*21,
            value="3",
            options=["1","2","3","4","5"],
            disabled=False
        )
        
        widget_opt['enable_parsing'] = widgets.Dropdown(
            layout=widgets.Layout(width='33%'), style=style,
            description='&nbsp;'*12 +'括号修改权重',
            value="圆括号 () 加强权重",
            options=["圆括号 () 加强权重","花括号 {} 加权权重", "否"],
            disabled=False
        )

        widget_opt['output_dir'] = widgets.Text(
            layout=layout, style=style,
            description='图片的保存路径' + '&nbsp;'*18,
            value="outputs/img2img",
            disabled=False
        )
        
        widget_opt['sampler'] = widgets.Dropdown(
            layout=widgets.Layout(width='50%'), style=style,
            description='采样器' + '&nbsp;'*30,
            value="DDIM",
            options=["PNDM", "DDIM", "LMS"],
            disabled=False
        )
        widget_opt['model_name'] = widgets.Dropdown(
            layout=layout, style=style,
            description='需要加载的模型名称',
            value="CompVis/stable-diffusion-v1-4",
            options=["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5", "hakurei/waifu-diffusion", "hakurei/waifu-diffusion-v1-3", "naclbit/trinart_stable_diffusion_v2_60k", "naclbit/trinart_stable_diffusion_v2_95k", "naclbit/trinart_stable_diffusion_v2_115k", "MoososCap/NOVEL-MODEL", "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1", "ruisi/anything"],
            disabled=False
        )
        widget_opt['superres_model_name'] = widgets.Dropdown(
            layout=widgets.Layout(width='50%'), style=style,
            description='&nbsp;'*15 + '超分模型',
            value="无",
            options=["falsr_a", "falsr_b", "falsr_c", "无"],
            disabled=False
        )

        widget_opt['seed'] = widgets.IntText(
            layout=layout, style=style,
            description='随机数种子(-1表示不设置随机种子)',
            value=-1,
            disabled=False
        )

        widget_opt['concepts_library_dir'] = widgets.Text(
            layout=layout, style=style,
            description='需要导入的"风格"或"人物"权重路径' + '&nbsp;'*4,
            value="outputs/textual_inversion",
            disabled=False
        )

        self.run_button = widgets.Button(
            description='点击生成图片！',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to run (settings will update automatically)',
            icon='check'
        )
        self.run_button.on_click(self.on_run_button_click)

        self.hboxes = {}
        self.hboxes['param1'] = widgets.HBox([
            widget_opt['num_inference_steps'],
            widget_opt['guidance_scale'],
            widget_opt['fp16']
        ])
        
        self.hboxes['tokenizer_settings'] = widgets.HBox([
            widget_opt['max_embeddings_multiples'],
            widget_opt['enable_parsing']
        ])

        self.hboxes['model'] = widgets.HBox([
            widget_opt['model_name'],
            widget_opt['sampler'],
            widget_opt['superres_model_name']
        ])
        

        self.gui = widgets.VBox([
            widget_opt['prompt'],
            widget_opt['negative_prompt'],
            widget_opt['image_path'],
            widget_opt['height'],
            widget_opt['width'],
            widget_opt['num_return_images'],
            widget_opt['strength'],
            self.hboxes['param1'],
            self.hboxes['tokenizer_settings'],
            widget_opt['output_dir'],
            widget_opt['seed'],
            self.hboxes['model'],
            widget_opt['concepts_library_dir'],
            self.run_button, 
            self.run_button_out
        ])

        self.task = 'img2img'


class SuperResolutionUI(StableDiffusionUI):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        widget_opt = self.widget_opt
        
        widget_opt['image_path'] = widgets.Text(
            layout=layout, style=style,
            description='需要超分的图片路径' ,
            value='resources/Ring.png',
            disabled=False
        )

        widget_opt['superres_model_name'] = widgets.Dropdown(
            layout=layout, style=style,
            description='超分模型的名字' + '&nbsp;'*6,
            value="falsr_a",
            options=["falsr_a", "falsr_b", "falsr_c"],
            disabled=False
        )

        widget_opt['output_dir'] = widgets.Text(
            layout=layout, style=style,
            description='图片的保存路径' + '&nbsp;'*6,
            value="outputs/highres",
            disabled=False
        )
        
        self.run_button = widgets.Button(
            description='点击超分图片！',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to run (settings will update automatically)',
            icon='check'
        )
        self.run_button.on_click(self.on_run_button_click)

        self.gui = widgets.VBox([widget_opt[k] for k in list(widget_opt.keys())]
                            +   [self.run_button, self.run_button_out])

        self.task = 'superres'



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
        self.parse_args = None
        self.main = None

    def run(self, opt):
        args = self.parse_args()
        for k, v in opt.items():
            setattr(args, k, v.value)

        # TODO junnyu 是否切换权重
        # runwayml/stable-diffusion-v1-5
        self.pipeline.from_pretrained(model_name=opt.model_name)
        self.pipeline.to('float32')
        args.text_encoder = self.pipeline.pipe.text_encoder
        args.unet         = self.pipeline.pipe.unet
        args.vae          = self.pipeline.pipe.vae

        if compute_gpu_memory() <= 17.:
            args.train_batch_size = 1
            args.gradient_accumulation_steps = 4
        else:
            args.train_batch_size = 4
            args.gradient_accumulation_steps = 1

        if args.train_data_dir is None:
            raise ValueError("You must specify a train data directory.")
        # args.logging_dir = os.path.join(args.output_dir, args.logging_dir)
        
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
    def __init__(self):
        super().__init__()
        self.parse_args = textual_inversion_parse_args
        self.main = textual_inversion_main

        widget_opt = self.widget_opt
        widget_opt['learnable_property'] = widgets.Dropdown(
            layout=layout, style=style,
            description='需要学习什么？风格或实体',
            value="object",
            options=["style", "object"],
            disabled=False
        )
        widget_opt['placeholder_token'] = widgets.Textarea(
            layout=layout, style=style,
            description='用来表示该内容的新词' + '&nbsp' * 7,
            value="<Alice>",
            disabled=False
        )
        widget_opt['initializer_token'] = widgets.Textarea(
            layout=layout, style=style,
            description='该内容最接近的单词是' + '&nbsp' * 7,
            value="girl",
            disabled=False
        )
        widget_opt['repeats'] = widgets.IntText(
            layout=layout, style=style,
            description='训练图片需要重复多少遍' + '&nbsp' * 3,
            value="100",
            disabled=False
        )
        widget_opt['train_data_dir'] = widgets.Text(
            layout=layout, style=style,
            description='训练图片的文件夹路径' + '&nbsp' * 7,
            value="resources/Alices",
            disabled=False
        )
        widget_opt['output_dir'] = widgets.Text(
            layout=layout, style=style,
            description='训练结果的保存路径' + '&nbsp' * 10,
            value="outputs/textual_inversion",
            disabled=False
        )
        widget_opt['height'] = widgets.IntText(
            layout=layout, style=style,
            description='训练图片的高度(像素), 64的倍数',
            value=512,
            step=64,
            disabled=False
        )
        widget_opt['width'] = widgets.IntText(
            layout=layout, style=style,
            description='训练图片的宽度(像素), 64的倍数',
            value=512,
            step=64,
            disabled=False
        )
        widget_opt['learning_rate'] = widgets.FloatText(
            layout=layout, style=style,
            description='训练学习率' + '&nbsp' * 24,
            value=5e-4,
            disabled=False
        )
        widget_opt['max_train_steps'] = widgets.IntText(
            layout=layout, style=style,
            description='最大训练步数' + '&nbsp' * 21,
            value=500,
            step=100,
            disabled=False
        )
        widget_opt['model_name'] = widgets.Dropdown(
            layout=layout, style=style,
            description='需要训练的模型名称',
            value="hakurei/waifu-diffusion-v1-3",
            options=["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5", "hakurei/waifu-diffusion", "hakurei/waifu-diffusion-v1-3", "naclbit/trinart_stable_diffusion_v2_60k", "naclbit/trinart_stable_diffusion_v2_95k", "naclbit/trinart_stable_diffusion_v2_115k", "MoososCap/NOVEL-MODEL", "ruisi/anything"],
            disabled=False
        )
        
        self.run_button = widgets.Button(
            description='开始训练',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to run (settings will update automatically)',
            icon='check'
        )
        self.run_button.on_click(self.on_run_button_click)
        
        self.gui = widgets.VBox([widget_opt[k] for k in list(widget_opt.keys())]
                            +   [self.run_button, self.run_button_out])


class StableDiffusionUI_text_inversion_prediction(StableDiffusionUI):
    def __init__(self):
        super().__init__()
        widget_opt = self.widget_opt
        widget_opt['prompt'] = widgets.Textarea(
            layout=layout, style=style,
            description='prompt描述' + '&nbsp;' * 22,
            value="<Alice> at the lake",
            disabled=False
        )
        widget_opt['negative_prompt'] = widgets.Textarea(
            layout=layout, style=style,
            description='negative_prompt反面描述 <br />',
            value="",
            disabled=False
        )

        widget_opt['height'] = widgets.IntText(
            layout=layout, style=style,
            description='图片的高度(像素), 64的倍数',
            value=512,
            step=64,
            disabled=False
        )

        widget_opt['width'] = widgets.IntText(
            layout=layout, style=style,
            description='图片的宽度(像素), 64的倍数',
            value=512,
            step=64,
            disabled=False
        )

        widget_opt['num_return_images'] = widgets.BoundedIntText(
            layout=layout, style=style,
            description='生成图片数量' + '&nbsp;'*22,
            value=1,
            min=1,
            max=100,
            step=1,
            disabled=False
        )

        widget_opt['num_inference_steps'] = widgets.IntText(
            layout=widgets.Layout(width='33%'), style=style,
            description='推理的步数' + '&nbsp'*24 ,
            value=50,
            disabled=False
        )

        widget_opt['guidance_scale'] = widgets.BoundedFloatText(
            layout=widgets.Layout(width='33%'), style=style,
            description= '&nbsp;'*22 + 'cfg',
            min=0,
            max=100,
            value=7.5,
            disabled=False
        )

        widget_opt['fp16'] = widgets.Dropdown(
            layout=widgets.Layout(width='33%'), style=style,
            description='&nbsp;'*15 + '精度' + '&nbsp;'*1,
            value="float32",
            options=["float32", "float16"],
            disabled=False
        )

        widget_opt['max_embeddings_multiples'] = widgets.Dropdown(
            layout=widgets.Layout(width='33%'), style=style,
            description='长度上限倍数' + '&nbsp;'*21,
            value="3",
            options=["1","2","3","4","5"],
            disabled=False
        )
        
        widget_opt['enable_parsing'] = widgets.Dropdown(
            layout=widgets.Layout(width='33%'), style=style,
            description='&nbsp;'*12 +'括号修改权重',
            value="圆括号 () 加强权重",
            options=["圆括号 () 加强权重","花括号 {} 加权权重", "否"],
            disabled=False
        )

        widget_opt['output_dir'] = widgets.Text(
            layout=layout, style=style,
            description='图片的保存路径' + '&nbsp;'*18,
            value="outputs/text_inversion_txt2img",
            disabled=False
        )
        
        widget_opt['seed'] = widgets.IntText(
            layout=layout, style=style,
            description='随机数种子(-1表示不设置随机种子)',
            value=-1,
            disabled=False
        )

        widget_opt['sampler'] = widgets.Dropdown(
            layout=widgets.Layout(width='50%'), style=style,
            description='采样器' + '&nbsp;'*30,
            value="DDIM",
            options=["PNDM", "DDIM", "LMS"],
            disabled=False
        )
        widget_opt['model_name'] = widgets.Dropdown(
            layout=layout, style=style,
            description='需要加载的模型名称',
            value="hakurei/waifu-diffusion-v1-3",
            options=["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5", "hakurei/waifu-diffusion", "hakurei/waifu-diffusion-v1-3", "naclbit/trinart_stable_diffusion_v2_60k", "naclbit/trinart_stable_diffusion_v2_95k", "naclbit/trinart_stable_diffusion_v2_115k", "MoososCap/NOVEL-MODEL", "ruisi/anything"],
            disabled=False
        )
        widget_opt['superres_model_name'] = widgets.Dropdown(
            layout=widgets.Layout(width='50%'), style=style,
            description='&nbsp;'*15 + '超分模型',
            value="无",
            options=["falsr_a", "falsr_b", "falsr_c", "无"],
            disabled=False
        )

        widget_opt['concepts_library_dir'] = widgets.Text(
            layout=layout, style=style,
            description='需要导入的"风格"或"人物"权重路径' + '&nbsp;'*4,
            value="outputs/textual_inversion",
            disabled=False
        )

        self.run_button = widgets.Button(
            description='点击生成图片！',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to run (settings will update automatically)',
            icon='check'
        )
        
        self.run_button.on_click(self.on_run_button_click)
    
        self.hboxes = {}
        self.hboxes['param1'] = widgets.HBox([
            widget_opt['num_inference_steps'],
            widget_opt['guidance_scale'],
            widget_opt['fp16']
        ])
        self.hboxes['tokenizer_settings'] = widgets.HBox([
            widget_opt['max_embeddings_multiples'],
            widget_opt['enable_parsing']
        ])
        self.hboxes['model'] = widgets.HBox([
            widget_opt['model_name'],
            widget_opt['sampler'],
            widget_opt['superres_model_name']
        ])

        self.gui = widgets.VBox([
            widget_opt['prompt'],
            widget_opt['negative_prompt'],
            widget_opt['height'],
            widget_opt['width'],
            widget_opt['num_return_images'],
            self.hboxes['param1'],
            self.hboxes['tokenizer_settings'],
            widget_opt['output_dir'],
            widget_opt['seed'],
            self.hboxes['model'],
            widget_opt['concepts_library_dir'],
            self.run_button, 
            self.run_button_out
        ])
# instantiation
gui_txt2img = StableDiffusionUI_txt2img()
gui_text_inversion = StableDiffusionUI_text_inversion_prediction()
gui_img2img = StableDiffusionUI_img2img()
gui_superres = SuperResolutionUI(pipeline = pipeline_superres)
gui_train_text_inversion = StableDiffusionUI_text_inversion()