from datetime import datetime
import shutil
from .ui import StableDiffusionUI

import ipywidgets as widgets
from ipywidgets import Layout,HBox,VBox,Box
from . import views

class StableDiffusionUI_txt2img(StableDiffusionUI):
    def __init__(self):
        super().__init__()
        
        styleDescription = {
            'description_width': "4rem"
        }
        DEFAULT_BADWORDS = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
        STYLE_SHEETS = '''
<style>
@media (max-width:576px) {
    .StableDiffusionUI_txt2img {
        margin-right: 0 !important;
    }
    .StableDiffusionUI_txt2img .widget-text,
    .StableDiffusionUI_txt2img .widget-dropdown,
    .StableDiffusionUI_txt2img .widget-textarea {
        flex-wrap: wrap !important;
        height: auto;
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    .StableDiffusionUI_txt2img .widget-text > label,
    .StableDiffusionUI_txt2img .widget-dropdown > label,
    .StableDiffusionUI_txt2img .widget-textarea > label {
        width: 100% !important;
        text-align: left !important;
        font-size: small !important;
    }
    .StableDiffusionUI_txt2img .prompt > textarea {
        min-height:10em;
        margin-left:2rem!important;
    }
    .StableDiffusionUI_txt2img .negative_prompt > textarea {
        margin-left:2rem!important;
    }
    .StableDiffusionUI_txt2img .standard_size,
    .StableDiffusionUI_txt2img .superres_model_name {
        order: -1;
    }
    .StableDiffusionUI_txt2img .box_width_height {
        flex: 8 8 60% !important;
    }
    .StableDiffusionUI_txt2img .box_wrap_quikbtns {
        margin-left: 0 !important;
    }
    .StableDiffusionUI_txt2img .box_wrap_quikbtns>button {
        padding: 0 !important;
    }
    .StableDiffusionUI_txt2img button.run_button, 
    .StableDiffusionUI_txt2img button.collect_button {
        width: 45% !important;
    }
    
}
</style>
'''
        widget_opt = self.widget_opt
        
        widget_opt['prompt'] = views.createView(
            'prompt',
            value="extremely detailed CG unity 8k wallpaper,black long hair,cute face,1 adult girl,happy, green skirt dress, flower pattern in dress,solo,green gown,art of light novel,in field",
        )
        widget_opt['negative_prompt'] = views.createView(
            'negative_prompt',
            value=DEFAULT_BADWORDS,
        )
        widget_opt['standard_size'] = views.createView(
            'standard_size',
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
        
        widget_opt['num_return_images'] = views.createView(
            'num_return_images',
        )
        widget_opt['enable_parsing'] = views.createView(
            'enable_parsing',
        )
        widget_opt['num_inference_steps'] = views.createView(
            'num_inference_steps',
        )
        widget_opt['guidance_scale'] = views.createView(
            'guidance_scale',
        )

        widget_opt['max_embeddings_multiples'] = views.createView(
            'max_embeddings_multiples',
        )
        widget_opt['fp16'] = views.createView(
            'fp16',
        )
        
        widget_opt['seed'] = views.createView(
            'seed',
        )
        widget_opt['superres_model_name'] = views.createView(
            'superres_model_name',
        )
        
        widget_opt['output_dir'] = views.createView(
            'output_dir',
            value="outputs/txt2img",
        )

        widget_opt['sampler'] = views.createView(
            'sampler',
        )
        widget_opt['model_name'] = views.createView(
            'model_name',
        )
        widget_opt['concepts_library_dir'] = views.createView(
            'concepts_library_dir',
        )
        
        def on_standard_size_change(change):
            widget_opt['width'].value = change.new // 10000
            widget_opt['height'].value = change.new % 10000
        widget_opt['standard_size'].observe(
                on_standard_size_change, 
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
            icon='palette',
            layout = Layout(
                position = 'absolute',
                height = '1.8rem',
                width = '1.8rem',
                margin = '-11rem 0 0 0'
            )
        )
        btnBadwards = widgets.Button(
            description= '',
            tooltip='填充标准负面描述',
            disabled=False,
            icon='paper-plane',
            layout = Layout(
                position = 'absolute',
                height = '1.8rem',
                width = '1.8rem',
                margin = '-2rem 0px 0rem -1.8rem'
            )
        )
        def fill_good_quality(b):
            if not widget_opt['prompt'].value.startswith('masterpiece,best quality,'):
                widget_opt['prompt'].value = 'masterpiece,best quality,' + widget_opt['prompt'].value
        def fill_bad_words(b):
            widget_opt['negative_prompt'].value = DEFAULT_BADWORDS
            
        btnGoodQuality.on_click(fill_good_quality)
        btnBadwards.on_click(fill_bad_words)
        
        def on_seed_change(change):
            if change.new != -1:
                widget_opt['num_return_images'].value = 1
        def on_num_return_images(change):
            if change.new != 1:
                widget_opt['seed'].value = -1
        widget_opt['seed'].observe(on_seed_change, names='value')
        widget_opt['num_return_images'].observe(on_num_return_images, names='value')
        
        
        self.run_button = widgets.Button(
            # description='点击生成图片！',
            description='生成图片！',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='点击运行（配置将自动更新）',
            icon='check'
        )
        self.run_button.add_class('run_button')
        
        self.run_button.on_click(self.on_run_button_click)
        
        collect_button = widgets.Button(
            description='收藏图片',
            disabled=True,
            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='将图片转移到Favorates文件夹中',
            icon='star-o'
        )
        collect_button.add_class('collect_button')
        self._output_collections = []
        def collect_images(b):
            dir = datetime.now().strftime('./Favorates/txt2img-%m%d/') 
            os.makedirs(dir, exist_ok=True)
            
            for file in self._output_collections:
                if os.path.isfile(file):
                    shutil.move(file, dir)
                file = file[:-4] + '.txt'
                if os.path.isfile(file):
                    shutil.move(file, dir)
            self._output_collections.clear()
            
        collect_button.on_click(collect_images)
        
        
        box_width_height = HBox([
                                    widget_opt['width'],
                                    labelSize,
                                    widget_opt['height']
                                ],
                            )
        views.setLayout('col04', box_width_height)
        box_width_height.add_class('box_width_height')
        box_wrap_quikbtns = Box([
                                btnGoodQuality,btnBadwards,
                            ], layout = Layout(
                                margin = '0 1rem',
                                height = '0',
                                overflow = 'visible'
                            ));
        box_wrap_quikbtns.add_class('box_wrap_quikbtns')
        self.gui = Box([
                widgets.HTML(STYLE_SHEETS),
                HBox([widget_opt['prompt']]),
                HBox([widget_opt['negative_prompt']]),
                box_wrap_quikbtns,
                Box([
                    widget_opt['standard_size'],
                    box_width_height,
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
                self.run_button, collect_button,
                self.run_button_out
            ], layout = Layout(display="block",margin="0 45px 0 0")
        )
        self.gui.add_class('StableDiffusionUI_txt2img')
    
    def on_run_button_click(self, b):
        options = {}
        for k in self.widget_opt:
            options[k] = self.widget_opt[k].value
            
        self._output_collections.clear()
        self.run_button.disabled = True
        
        try:
            with self.run_button_out:
                clear_output()
                self.pipeline.run(
                    options, 
                    task = self.task,
                    on_image_generated = self.on_image_generated
                )
        finally:
            self.run_button.disabled = False
           
           
    def on_image_generated(self, image, options, count, total):
        image_path = save_image_info(image, options.output_dir)
        self._output_collections.append(image_path)
        
        if count % 5 == 0:
            clear_output()
        
        try:
            with open(image_path,'rb') as file:
                data = file.read()
            display(widgets.Image(value = data))    # 使显示的图片包含嵌入信息
        except:
            display(image)
        
        print('Seed = ', image.argument['seed'], 
            '    (%d / %d ... %.2f%%)'%(i + 1, total, (count + 1.) / total * 100))

