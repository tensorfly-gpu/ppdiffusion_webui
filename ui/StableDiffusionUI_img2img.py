from .ui import StableDiffusionUI

import ipywidgets as widgets
from ipywidgets import Layout,HBox,VBox,Box
from . import views

class StableDiffusionUI_img2img(StableDiffusionUI):
    def __init__(self):
        super().__init__()
        self.task = 'img2img'
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
            'description_width': "4rem"
        }

        widget_opt = self.widget_opt
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
            value="couple couple rings surface from (Starry Night of Van Gogh:1.1), couple couple rings in front of grey background, simple background, elegant style design, full display of fashion design",
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
            value='',
            disabled=False
        )

        widget_opt['image_path'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='输入图片',
            description_tooltip='需要转换的图片的路径',
            value='resources/Ring.png',
            disabled=False
        )

        widget_opt['height'] = widgets.IntText(
            layout=layoutCol04, style=styleDescription,
            description='图片高度',
            description_tooltip='图片的高度, -1为自动判断',
            value=-1,
            disabled=False
        )
        
        widget_opt['width'] = widgets.IntText(
            layout=layoutCol04, style=styleDescription,
            description='图片宽度',
            description_tooltip='图片的宽度, -1为自动判断',
            value=-1,
            disabled=False
        )

        widget_opt['num_return_images'] = widgets.BoundedIntText(
            layout=layoutCol04, style=styleDescription,
            description='生成数量',
            description_tooltip='生成图片的数量',
            value=1,
            min=1,
            max=100,
            step=1,
            disabled=False
        )

        widget_opt['strength'] = widgets.FloatSlider(
            layout=layoutCol04, style=styleDescription,
            description='修改强度',
            description_tooltip='修改图片的强度',
            value=0.8,
            min=0,
            max=1,
            step=0.05,
            readout=True,
            readout_format='.1f',
            orientation='horizontal',
            disabled=False,
            continuous_update=False
        )

        widget_opt['num_inference_steps'] = widgets.BoundedIntText(
            layout=layoutCol04, style=styleDescription,
            description='推理步数',
            description_tooltip='推理步数（Step）：生成图片的迭代次数，步数越多运算次数越多。',
            value=50,
            min=2,
            max=250,
            # orientation='horizontal',
            # readout=True,
            disabled=False
        )

        widget_opt['guidance_scale'] = widgets.BoundedFloatText(
            layout=layoutCol04, style=styleDescription,
            description= 'CFG',
            description_tooltip='引导度（CFG Scale）：控制图片与描述词之间的相关程度。',
            min=0,
            max=100,
            value=7.5,
            disabled=False
        )

        widget_opt['fp16'] = widgets.Dropdown(
            layout=layoutCol04, style=styleDescription,
            description='算术精度',
            description_tooltip='模型推理使用的精度。选择float16可以加快模型的推理速度，但会牺牲部分的模型性能。',
            value="float32",
            options=["float32", "float16"],
            disabled=False
        )

        widget_opt['max_embeddings_multiples'] = widgets.Dropdown(
            layout=layoutCol04, style=styleDescription,
            description='上限倍数',
            description_tooltip='修改长度上限倍数，使模型能够输入更长更多的描述词。',
            value="3",
            options=["1","2","3","4","5"],
            disabled=False
        )
        
        widget_opt['enable_parsing'] = widgets.Dropdown(
            layout=layoutCol04, style=styleDescription,
            description='括号格式',
            description_tooltip='增加权重所用括号的格式，可以将{}替换为()。选择“否”则不解析加权语法',
            value="圆括号 () 加强权重",
            options=["圆括号 () 加强权重","花括号 {} 加权权重", "否"],
            disabled=False
        )
        
        widget_opt['output_dir'] = widgets.Text(
            layout=layoutCol08, style=styleDescription,
            description='保存路径',
            description_tooltip='用于保存输出图片的路径',
            value="outputs/img2img",
            disabled=False
        )
        
        widget_opt['sampler'] = widgets.Dropdown(
            layout=layoutCol04, style=styleDescription,
            description='采样器',
            value="DDIM",
            options=["PNDM", "DDIM", "LMS"],
            disabled=False
        )
            
        widget_opt['model_name'] = widgets.Combobox(
            layout=layoutCol08, style=styleDescription,
            description='模型名称',
            description_tooltip='需要加载的模型名称',
            value="CompVis/stable-diffusion-v1-4",
            options=["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5", "hakurei/waifu-diffusion", "hakurei/waifu-diffusion-v1-3", "naclbit/trinart_stable_diffusion_v2_60k", "naclbit/trinart_stable_diffusion_v2_95k", "naclbit/trinart_stable_diffusion_v2_115k", "MoososCap/NOVEL-MODEL", "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1", "ruisi/anything"],
            ensure_option=False,
            disabled=False
        )
        widget_opt['superres_model_name'] = widgets.Dropdown(
            layout=layoutCol04, style=styleDescription,
            description='图像放大',
            description_tooltip='指定放大图片尺寸所用的模型',
            value="无",
            options=["falsr_a", "falsr_b", "falsr_c", "无"],
            disabled=False
        )

        widget_opt['seed'] = widgets.IntText(
            layout=layoutCol04, style=styleDescription,
            description='随机种子',
            description_tooltip='-1表示随机生成。',
            value=-1,
            disabled=False
        )

        widget_opt['concepts_library_dir'] = widgets.Text(
            layout=layoutCol08, style=styleDescription,
            description='风格权重',
            description_tooltip='TextualInversion训练的、“风格”或“人物”的权重文件路径',
            value="outputs/textual_inversion",
            disabled=False
        )

        self.run_button = widgets.Button(
            description='点击生成图片！',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='点击运行（配置将自动更新）',
            icon='check'
        )
        self.run_button.on_click(self.on_run_button_click)

        self.gui = Box([
                HBox([widget_opt['prompt']]),
                HBox([widget_opt['negative_prompt']]),
                Box([
                    widget_opt['image_path'],
                    widget_opt['height'],
                    widget_opt['width'],
                    widget_opt['superres_model_name'],
                    
                    widget_opt['strength'],
                    widget_opt['num_return_images'],
                    widget_opt['seed'],
                    
                    widget_opt['num_inference_steps'],
                    widget_opt['guidance_scale'],
                    widget_opt['sampler'],
                    
                    widget_opt['enable_parsing'],
                    widget_opt['max_embeddings_multiples'],
                    widget_opt['fp16'],
                    
                    widget_opt['output_dir'],
                    widget_opt['model_name'],
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
        
