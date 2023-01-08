import os
import shutil
import time
from .ui import StableDiffusionUI
from .utils import save_image_info

from IPython.display import clear_output, display
import ipywidgets as widgets
from ipywidgets import Layout,HBox,VBox,Box
from . import views

MAX_DISPLAY_COUNT = 5

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
        
        #默认参数覆盖次序：
        #user_config.py > config.py > 当前args > views.py
        args = {  #注意无效Key错误
            "prompt": '',
            "negative_prompt": '',
            "width": 512,
            "height": 512,
        }
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
            height_value = args['height'], 
            step64 = True,
        )
        widget_opt['width'] = view_width_height['width']
        widget_opt['height'] = view_width_height['height']
        
        widget_opt['model_name'] = views.createModelNameView(
            self.pipeline,
            args['model_name'] if 'model_name' in args else None
            )
        
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
        self._imgview_collection = OutputImageViewCollection(
            output_display = self.run_button_out,
            task_name = self.task,
            set_seed_callback = self.set_seed,
            length = MAX_DISPLAY_COUNT
        )
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
            dir = time.strftime(f'Favorates/{self.task}-%m%d/') 
            info = '收藏图片到 ' + dir
            dir = './' + dir
            os.makedirs(dir, exist_ok=True)
            
            for view in self._imgview_collection.get_all_visible():
                image_path = view.image_path
                if image_path in self._output_collections: self._output_collections.remove(image_path)
                view.collect()
            
            for image_path in self._output_collections:
                if os.path.isfile(image_path):
                    shutil.move(image_path, dir)
                    print(info + os.path.basename(image_path))
                txt_file = image_path.rpartition('.')[0] + '.txt'
                if os.path.isfile(txt_file):
                    shutil.move(txt_file, dir)
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

    def set_seed(self,seed):
        self.widget_opt['seed'].value = seed
    
    def on_image_generated(self, image, options, count = 0, total = 1, image_info = None):
        image_path = save_image_info(image, options.output_dir)
        self._output_collections.append(image_path)
        
        index = count % MAX_DISPLAY_COUNT
        if index == 0:
            clear_output()
            self._imgview_collection.reset_all()
        
        view = self._imgview_collection[index]
        seed = image.argument['seed'] if 'seed' in image.argument else -1
        view.set_value(
            image_path, seed, count, total
        )
        view.render()
    
    def test_output(self, image_path):
        seed = 793344
        count = 1
        total = 2

        if count % 5 == 0:
            clear_output()

        with self.run_button_out:
            view = OutputImageView(self.run_button_out, self.task, self.set_seed)
            view.set_value(
                image_path, seed, count, total
            )
            view.render()
    

class OutputImageView():
    
    def __init__(self, output_display, task_name, set_seed_callback):
        label = widgets.Label(
            '#1/1',
            layout = Layout(
                padding = '0 0.5rem'
            )
        )
        btn_collect = widgets.Button(
            icon = 'star-o',
            button_style = 'info',
            tooltip = '将图片转移到Favorates文件夹中',
            layout = Layout(
                width= 'auto',
                height = 'auto',
            )
        )
        btn_delete = widgets.Button(
            icon = 'trash-alt',
            button_style = 'danger',
            tooltip = '将图片转移到TrashBin文件夹中',
            layout = Layout(
                width= 'auto',
                height = 'auto',
                margin = '0',
            )
        )
        btn_seed = widgets.Button(
            description = '-1',
            tooltip = '设置随机种子',
            layout = Layout(
                width= 'auto',
                height = 'auto',
                margin = '0',
            )
        )
        
        self.box_toolbar = HBox(
            (
                label, btn_seed, 
                widgets.Label(layout = Layout(flex = '10 10 auto')),
                btn_collect, btn_delete,
            ),
            layout = Layout(
                width= '100%',
                max_width= '100%',
                align_items = 'center',
                align_centent = 'center',
                justify_content = 'center',
            )
        )
        self.container = views.FlexBox(
            layout = Layout(
                width = '100%',
                border = 'var(--jp-border-width) solid var(--jp-border-color1)',
            )
        )
        
        
        def _on_btn_seed_click(b):
            with output_display:
                set_seed_callback(seed = self.seed_value)
        btn_collect.on_click(self._on_btn_click)
        btn_delete.on_click(self._on_btn_click)
        btn_seed.on_click(_on_btn_seed_click)
        
        
        self.output_display = output_display
        self.set_seed_callback = set_seed_callback
        self.label = label
        self.btn_collect = btn_collect
        self.btn_delete = btn_delete
        self.btn_seed = btn_seed
        self.state = 'standby'
        self.image_path = ''
        self.seed_value = -1
        self.task_name = task_name
        self.visible = False

    def render(self):
        self.visible = True
        display(self.container)
    
    def set_value(self, image_path, seed, count = 0, total = 1):
        self.seed_value = seed
        self.btn_seed.description = str(self.seed_value)
        self.label.value = f'#{count+1}/{total}'
        self.state = 'standby'
        self.btn_delete.disabled = False
        self.btn_collect.disabled = False
        if self.image_path != image_path:
            self.image_path = image_path
            if not os.path.isfile(image_path):
                self.container.children = (
                    self.box_toolbar,
                    widgets.HTML('未选中图片或无效的图片'),
                    )
            else:
                self.container.children = (
                    self.box_toolbar,
                    widgets.Image.from_file(image_path),
                    )
    
    def move_image(self, opt_type = 'collect'):
        # 检查可用
        if opt_type == 'collect':
            new_state = 'collected'
            if self.state == new_state: return
            dir = time.strftime(f'Favorates/{self.task_name}-%m%d/') 
            info = '收藏图片到 ' + dir
        elif opt_type == 'delete':
            new_state = 'deleted'
            if self.state == new_state: return
            dir = time.strftime(f'TrashBin/{self.task_name}-%m%d/') 
            info = '移除图片到 ' + dir
        else:
            raise Exception()
            
        image_path = self.image_path
        image_name = os.path.basename(image_path)
        if not os.path.isfile(image_path):
            return print('未找到文件：', image_path)
        
        # 创建文件夹
        dir = './' + dir
        os.makedirs(dir, exist_ok=True)
        
        # 转移文件
        shutil.move(image_path, dir)
        print(info + image_name)
        txt_file = image_path.rpartition('.')[0] + '.txt'
        if os.path.isfile(txt_file):
            shutil.move(txt_file, dir)
        
        self.image_path = os.path.join(dir,image_name)
        self.state == new_state
        if new_state == 'collected':
            self.btn_delete.disabled = False
            self.btn_collect.disabled = True
        elif new_state == 'deleted':
            self.btn_delete.disabled = True
            self.btn_collect.disabled = False
    
    def collect(self):
        self.move_image('collect')
    
    def _on_btn_click(self, b):
        with self.output_display:
            if b == self.btn_collect:
                self.move_image('collect')
            elif b == self.btn_delete:
                self.move_image('delete')
            else:
                raise Exception()
                
            
            
class OutputImageViewCollection(list):
    def __init__(self, output_display, task_name, set_seed_callback, length = 5):
        self.output_display = output_display
        self.task_name = task_name
        self.set_seed_callback = set_seed_callback
        i = 0
        while i < length:
            super().append(None)
            i += 1
    
    def __getitem__(self, index):
        val = super().__getitem__(index)
        
        if val is not None: return val
        
        val = OutputImageView(self.output_display, self.task_name, self.set_seed_callback)
        super().__setitem__(index, val)
            
        return val
    
    def reset_all(self):
        for val in self:
            if val is not None:
                val.visible = False
    
    def get_all_visible(self):
        for val in self:
            if val is not None and val.visible:
                yield val
        