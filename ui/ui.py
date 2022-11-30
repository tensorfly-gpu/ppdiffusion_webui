# Code credits to 凉心半浅良心人
# Has modified

import os
os.environ['PPNLP_HOME'] = "./model_weights"
from IPython.display import clear_output


from .env import DEBUG_UI

if not DEBUG_UI:

    from .utils import diffusers_auto_update
    diffusers_auto_update(hint_kernel_restart = True)

    #from tqdm.auto import tqdm
    import paddle

    from .textual_inversion import parse_args as textual_inversion_parse_args
    from .textual_inversion import main as textual_inversion_main
    from .utils import StableDiffusionFriendlyPipeline, SuperResolutionPipeline, diffusers_auto_update
    from .utils import compute_gpu_memory, empty_cache
    from .utils import save_image_info

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
    
    def on_image_generated(self, image, options, count, total):
        image_path = save_image_info(image, options.output_dir)
        if count % 5 == 0:
            clear_output()
        
        try:
            with open(image_path,'rb') as file:
                data = file.read()
            display(widgets.Image(value = data))    # 使显示的图片包含嵌入信息
        except:
            display(image)
        
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

        if compute_gpu_memory() <= 17. or args.height==768 or args.width==768:
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
            disabled=True
        )
        widget_opt['width'] = widgets.IntSlider(
            layout=layoutCol12, style=styleDescription,
            description='训练图片的宽度',
            description_tooltip='训练图片的宽度。越大尺寸，消耗的显存也越多。',
            value=512,
            min=64,
            max=1024,
            step=64,
            disabled=True
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
        widget_opt['model_name'] = widgets.Combobox(
            layout=layoutCol12, style=styleDescription,
            description='需要训练的模型名称',
            value="hakurei/waifu-diffusion-v1-3",
            options=[
                'CompVis/stable-diffusion-v1-4', 
                'runwayml/stable-diffusion-v1-5', 
                'stabilityai/stable-diffusion-2', 
                'stabilityai/stable-diffusion-2-base', 
                'hakurei/waifu-diffusion', 
                'hakurei/waifu-diffusion-v1-3', 
                'naclbit/trinart_stable_diffusion_v2_60k', 
                'naclbit/trinart_stable_diffusion_v2_95k', 
                'naclbit/trinart_stable_diffusion_v2_115k', 
                'IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1', 
                'IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1', 
                'BAAI/AltDiffusion', 
                'BAAI/AltDiffusion-m9',
                'MoososCap/NOVEL-MODEL', 
                'ruisi/anything',
                'Linaqruf/anything-v3.0', 
            ],
            ensure_option=False,
            disabled=False
        )
        
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
            value='default',
            options=[
                'default', 
                'DPMSolver',
                'EulerDiscrete',
                'EulerAncestralDiscrete', 
                'PNDM', 
                'DDIM', 
                'LMSDiscrete',
            ], 
            disabled=False
        )
        widget_opt['model_name'] = widgets.Dropdown(
            layout=layout, style=style,
            description='需要加载的模型名称',
            value="hakurei/waifu-diffusion-v1-3",
            options=[
                'CompVis/stable-diffusion-v1-4', 
                'runwayml/stable-diffusion-v1-5', 
                'stabilityai/stable-diffusion-2', 
                'stabilityai/stable-diffusion-2-base', 
                'hakurei/waifu-diffusion', 
                'hakurei/waifu-diffusion-v1-3', 
                'naclbit/trinart_stable_diffusion_v2_60k', 
                'naclbit/trinart_stable_diffusion_v2_95k', 
                'naclbit/trinart_stable_diffusion_v2_115k', 
                'IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1', 
                'IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1', 
                'BAAI/AltDiffusion', 
                'BAAI/AltDiffusion-m9',
                'MoososCap/NOVEL-MODEL', 
                'ruisi/anything',
                'Linaqruf/anything-v3.0', 
            ],
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
