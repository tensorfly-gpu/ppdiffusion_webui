from datetime import datetime
import os
import shutil
from .ui import StableDiffusionUI
from .utils import save_image_info

from IPython.display import clear_output, display
import ipywidgets as widgets
from ipywidgets import Layout,HBox,VBox,Box
from . import views

class StableDiffusionUI_txt2img(StableDiffusionUI):
    def __init__(self, **kwargs):
        super().__init__()      #暂且不处理pipline
        
        CLASS_NAME = self.__class__.__name__ \
                + '_{:X}'.format(hash(self))[-4:]
        
        STYLE_SHEETS = '''
@media (max-width:576px) {
    {root} .standard_size,
    {root} .superres_model_name {
        order: -1;
    }
    
    {root} button.run_button, 
    {root} button.collect_button {
        width: 45% !important;
    }
}
'''
        
        #默认参数
        args = {}
        args.update(kwargs)
        widget_opt = self.widget_opt

        
        # 提示词部分
        view_prompts = views.createPromptsView(
            value = args['prompt'],
            negative_value = args['negative_prompt'],
        )
        
        widget_opt['prompt'] = view_prompts['prompt']
        widget_opt['negative_prompt'] = view_prompts['negative_prompt']
        
        
        # 图片尺寸部分
        view_width_height = views.createWidthHeightView(
            width_value = args['width'], 
            height_value = args['width'], 
            step64 = True,
        )
        widget_opt['width'] = view_width_height['width']
        widget_opt['height'] = view_width_height['height']
        
        for key in (
             'standard_size',
             'num_return_images',
             'enable_parsing',
             'num_inference_steps',
             'guidance_scale',
             'max_embeddings_multiples',
             'fp16',
             'seed',
             'superres_model_name',
             'output_dir',
             'sampler',
             'model_name',
             'concepts_library_dir'
            ):
            widget_opt[key] = views.createView(key)
            if key in args:
                widget_opt[key].value = args[key]
        
        
        # 事件处理绑定
        def on_standard_size_change(change):
            widget_opt['width'].value = change.new // 10000
            widget_opt['height'].value = change.new % 10000
        widget_opt['standard_size'].observe(
                on_standard_size_change, 
                names = 'value'
        )
        def on_seed_change(change):
            if change.new != -1:
                widget_opt['num_return_images'].value = 1
        def on_num_return_images(change):
            if change.new != 1:
                widget_opt['seed'].value = -1
        widget_opt['seed'].observe(on_seed_change, names='value')
        widget_opt['num_return_images'].observe(on_num_return_images, names='value')
        
        # 按钮x2
        self.run_button = views.createView('run_button')
        self.collect_button = views.createView('collect_button')
        
        self._output_collections = []
        self.run_button.on_click(self.on_run_button_click)
        self.collect_button.on_click(self.on_collect_button_click)
        
        # 样式表
        STYLE_SHEETS = ('<style>' \
                + views.SHARED_STYLE_SHEETS \
                + STYLE_SHEETS \
                + view_prompts.style_sheets \
                + view_width_height.style_sheets \
                + '</style>'
            ).replace('{root}', '.' + CLASS_NAME)
        
        #
        self.gui = views.createView("box_gui", 
            class_name = CLASS_NAME,
            children = [
                widgets.HTML(STYLE_SHEETS),
                view_prompts.container,
                views.createView("box_main", 
                [
                    widget_opt['standard_size'],
                    view_width_height.container,
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
                ]),
                HBox(
                    (self.run_button,self.collect_button,),
                    layout = Layout(
                        justify_content = 'space-around',
                        max_width = '100%',
                    )
                ),
                self.run_button_out
            ], 
        )
    
    def on_collect_button_click(self, b):
        with self.run_button_out:
            dir = datetime.now().strftime(f'Favorates/{self.task}-%m%d/') 
            info = f'收藏图片到 ' + dir
            dir = './' + dir
            os.makedirs(dir, exist_ok=True)
            
            for file in self._output_collections:
                if os.path.isfile(file):
                    shutil.move(file, dir)
                    print(info + os.path.basename(file))
                file = file[:-4] + '.txt'
                if os.path.isfile(file):
                    shutil.move(file, dir)
            self._output_collections.clear()
            self.collect_button.disabled = True

    def on_run_button_click(self, b):
        with self.run_button_out:
            self._output_collections.clear()
            self.collect_button.disabled = True
            self.run_button.disabled = True
            try:
                super().on_run_button_click(b)
            finally:
                self.run_button.disabled = False
                self.collect_button.disabled = len(self._output_collections) < 1

    def on_image_generated(self, image, options, count, total):
        image_path = save_image_info(image, options.output_dir)
        self._output_collections.append(image_path)
        
        if count % 5 == 0:
            clear_output()
        
        try:
            # 使显示的图片包含嵌入信息
            display(widgets.Image.from_file(image_path))
        except:
            display(image)
        
        print('Seed = ', image.argument['seed'], 
            '    (%d / %d ... %.2f%%)'%(count + 1, total, (count + 1.) / total * 100))

