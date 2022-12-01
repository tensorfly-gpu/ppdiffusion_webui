from datetime import datetime
import os
import shutil
from .ui import StableDiffusionUI
from .utils import save_image_info

from IPython.display import clear_output, display
import ipywidgets as widgets
from ipywidgets import Layout,HBox,VBox,Box
            # HBox/VBox应当仅用于【单行/单列】内容
            # Box应当始终假定display不明
from . import views
from .views import Div, Tab, Bunch

class StableDiffusionUI_img2img(StableDiffusionUI):
    def __init__(self, **kwargs):
        super().__init__()      #暂且不处理pipline
        CLASS_NAME = self.__class__.__name__ \
                + '_{:X}'.format(hash(self))[-4:]
        
        STYLE_SHEETS = '''
@media (max-width:576px) {
    {root} .column_left,
    {root} .column_right {
        min-width: 240px !important;    #解决X轴超出边框
    }
    {root} .column_right .panel01 {
        max_height: 500px !important;    #解决Y轴超出边框
    }
    
    {root} .p-TabPanel-tabContents.widget-tab-contents {
        padding: 0.5rem 0.1rem !important;
    }
    
    {root} button.run_button, 
    {root} button.collect_button {
        _width: 45% !important;
    }
    {root} .seed {
        min-width = '6rem';
    }
}

/* 强度拖动条 */
{root} .strength.widget-slider > .widget-readout {
    border: var(--jp-border-width) solid var(--jp-border-color1) !important;
}
@media (max-width:576px) {
    {root} .strength.widget-slider {
        flex: 4 4 30% !important;
        min-width: 6rem !important;
    }
    
    {root} .strength.widget-slider > .slider-container {
        display: none !important;  /* 手机无法拖动 */
    }
}
'''
        
        self.task = 'img2img'
        
        #默认参数
        args = {}
        args.update(kwargs)
        args['num_return_images'] = 1 #不支持批量生成
        # widget_opt = self.widget_opt
        
        #生成主要控件
        self._generateControls(args)
        
        #生成左侧
        self._renderColumnLeft(args)
        
        #生成右侧
        self._renderColumnRight(args)
        
        
        # 样式表
        STYLE_SHEETS = ('<style>' \
                + views.SHARED_STYLE_SHEETS \
                + STYLE_SHEETS \
                + self.view_prompts.style_sheets \
                + self.view_width_height.style_sheets \
                + '</style>'
            ).replace('{root}', '.' + CLASS_NAME)
        html_css = widgets.HTML(STYLE_SHEETS)
        html_css.layout.display = 'none'
        
        
        box_gui = Box([
                html_css,
                self._column_left,
                self._column_right,
            ],
            layout = Layout(
                display = "flex",
                flex_flow = "row wrap", #HBox会覆写此属性
                max_width = '100%',
            ),
        )
        box_gui.add_class(CLASS_NAME)
        
        clear_output()
        self.gui = box_gui
    
    # 生成主要控件
    def _generateControls(self, args):
        widget_opt = self.widget_opt
        
        # 提示词部分
        view_prompts = views.createPromptsView(
            value = args['prompt'],
            negative_value = args['negative_prompt'],
        )
        widget_opt['prompt'] = view_prompts['prompt']
        widget_opt['negative_prompt'] = view_prompts['negative_prompt']
        self.view_prompts = view_prompts
        
        # 图片尺寸部分
        view_width_height = views.createWidthHeightView(
            width_value = args['width'], 
            height_value = args['width'], 
        )
        widget_opt['width'] = view_width_height['width']
        widget_opt['height'] = view_width_height['height']
        self.view_width_height = view_width_height
        
        # 强度
        widget_opt['strength'] = widgets.FloatSlider(
            style={
                    'description_width': "4rem"
                },
            description='修改强度',
            description_tooltip='修改图片的强度',
            value=0.8,
            min=0.01,
            max=0.99,
            step=0.01,
            readout=True,
            readout_format='.2f',
            orientation='horizontal',
            disabled=False,
            continuous_update=False
        )
        widget_opt['strength'].add_class('strength')
        views.setLayout('col08', widget_opt['strength'])
        
        
        for key in (
             'num_return_images',
             'num_inference_steps',
             'guidance_scale',
             'seed',
             'output_dir',
             'sampler',
             'model_name',
             'concepts_library_dir',
            ):
            widget_opt[key] = views.createView(key)
            if key in args:
                widget_opt[key].value = args[key]
        for key in (
             'enable_parsing',
             'max_embeddings_multiples',
             'superres_model_name',
             'fp16',
            ):
            widget_opt[key] = views.createView(
                key,
                layout_name = 'col06',
            )
            if key in args:
                widget_opt[key].value = args[key]
        widget_opt['seed'].layout.min_width = '8rem'
        
        # 按钮x2
        self.run_button = views.createView('run_button')
        self.collect_button = views.createView('collect_button')
        self._output_collections = []
        self.collect_button.on_click(self.on_collect_button_click)
        self.run_button.on_click(self.on_run_button_click)
        
        # 事件处理绑定
        def on_seed_change(change):
            if change.new != -1:
                widget_opt['num_return_images'].value = 1
        def on_num_return_images(change):
            if change.new != 1:
                widget_opt['seed'].value = -1
        widget_opt['seed'].observe(on_seed_change, names='value')
        widget_opt['num_return_images'].observe(on_num_return_images, names='value')
        
    
    # 构建视图
    def _renderColumnRight(self, args):
        widget_opt = self.widget_opt
        
        views.setLayout('col12', self.view_prompts.container)
        widget_opt['sampler'].layout.min_width = '10rem'

        _panel_layout = Layout(
            display = 'flex',
            flex_flow = 'row wrap', #HBox会覆写此属性
            max_width = '100%',
            
            min_height = '360px',
            max_height = '455px',
            # height = '455px',
            align_items = 'center',
            align_content = 'center',
        )
        panel01 = Box(
            layout = _panel_layout,
            children = (
                self.view_prompts.container,
                widget_opt['strength'],
                widget_opt['seed'],
                widget_opt['num_inference_steps'],
                widget_opt['guidance_scale'],
                widget_opt['sampler'],
                widget_opt['model_name'],
            ),
        )
        panel02 = Box(
            layout = _panel_layout,
            children = (
                self.view_width_height.container,
                widget_opt['superres_model_name'],
                widget_opt['fp16'],
                
                widget_opt['enable_parsing'],
                widget_opt['max_embeddings_multiples'],
                
                widget_opt['output_dir'],
                widget_opt['concepts_library_dir'],
            ),
        )
        panel03 = Box(
            layout = _panel_layout,
            children = (
                self.run_button_out,
            ),
        )
        panel01.add_class('panel01')
        
        tab_right = Tab(
            titles = ('参数','其他','输出'),
            children = (
                panel01,
                panel02,
                panel03,
            ),
            layout = Layout(
                # flex = '1 1 360px',
                margin = '0',
            )
        )
        
        
        column_right = Div(
            children = [
                tab_right,
                HBox(
                    (self.run_button, self.collect_button,),
                    layout = Layout(
                        justify_content = 'space-around',
                        align_centent = 'center',
                        height = '45px',
                    )
                ),
            ],
            layout = Layout(
                flex = '1 1 300px',
                min_width = '300px',
                margin = '0.5rem 0',
            )
        )
        column_right.add_class('column_right')
         
        self._column_right = column_right
        self._tab_right = tab_right
     
    def _renderColumnLeft(self, args):
        widget_opt = self.widget_opt
        
        #--------------------------------------------------
        # ImportPanel
        view_upload = _createUploadView(
            label = '输入图片', 
            tooltip = '选择一张图片开始图生图',
            default_path = args['image_path'], 
            upload_path = args['upload_image_path'],
            text = '选择一张图片作为图生图的原始图片。你可以选择云端的文件，或者上传图片。',
        )
        view_upload_mask = _createUploadView(
            label = '蒙版图片',
            tooltip = '选择一张图片限定重绘范围',
            default_path = args['mask_path'],
            upload_path = args['upload_mask_path'],
            text = '选择一张图片作为蒙版，用白色区域限定图片重绘的范围。折叠此面板时不启用蒙版。',
        )
        widget_opt['image_path'] = view_upload.input
        widget_opt['mask_path'] = view_upload_mask.input
        self.uploador = view_upload.uploador
        
        btn_confirm = widgets.Button(
            description='导 入',
            disabled=False,
            button_style='success',
            layout=Layout(
                flex = '0 1 auto',
            ),
        )
        btn_reset = widgets.Button(
            description='重 置',
            disabled=False,
            button_style='warning',
            layout=Layout(
                flex = '0 1 auto',
            ),
        )
        btn_reset.add_class('btnV5')
        btn_reset.add_class('btn-small')
        btn_confirm.add_class('btnV5')
        btn_confirm.add_class('btn-small')
        
        accordion = widgets.Accordion([
                view_upload_mask.container,
            ],
                layout = Layout(
                    margin = '0.5rem 0',
                    max_width = '100%',
                )
            )
        accordion.set_title(0, '启用蒙版')
        accordion.selected_index = None
        
        panel_import = HBox(
            layout = Layout(
                height = '100%',
                max_width = '100%',
                max_height = '500px',
                min_height = '360px',
                align_items = "center",
                justify_content = 'center',
            ),
            children = [
                Div([
                    view_upload.container,
                    accordion,
                    HBox(
                        (btn_confirm, btn_reset),
                        layout = Layout(
                            justify_content = 'space-around',
                            max_width = '100%',
                        )
                    ),
                ]
                ),
            ],
        )
        
        #--------------------------------------------------
        # 其他Panel以及Tab
        view_image = createPanelImage(args['image_path'])
        view_image_mask = createPanelImage(args['mask_path'])
        view_image_output = createPanelImage()
        
        tab_left = Tab(
            titles = ('导入','原图','蒙版','输出'),
            children = (
                panel_import,
                view_image.container,
                view_image_mask.container,
                view_image_output.container,
            ),
            layout = Layout(
                flex = '1 1 300px',
                # max_width = '360px',
                min_width = '300px',
                margin = '0.5rem 0',
            )
        )
        tab_left.add_class('column_left')
        
        #--------------------------------------------------
        # 处理事件
        def whether_use_mask():
            return accordion.selected_index == 0
        def on_reset_button_click(b):
            with self.run_button_out:
                view_upload.reset()
                view_upload_mask.reset()
        def on_conform_button_click(b):
            with self.run_button_out:
                path = view_upload.confirm()
                if not view_image.set_file(path): raise IOError('未能读取文件：'+path)
                if whether_use_mask():
                    path = view_upload_mask.confirm()
                    if not view_image_mask.set_file(path): raise IOError('未能读取文件：'+path)
                    tab_left.selected_index = 2
                else:
                    tab_left.selected_index = 1
                view_image_output.set_file()
            return
        btn_reset.on_click(on_reset_button_click)
        btn_confirm.on_click(on_conform_button_click)
        self.is_inpaint_task = whether_use_mask
        
        self._column_left = tab_left
        self._tab_left = tab_left
        self._set_output_image = view_image_output.set_file #不是class所以不用self
        
        return
        
    def on_collect_button_click(self, b):
        with self.run_button_out:
            dir = datetime.now().strftime(f'Favorates/{self.task}-%m%d/') 
            info = '收藏图片到 ' + dir
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
            self.task = 'img2img' if not self.is_inpaint_task() else 'inpaint'
            self._tab_left.selected_index = 1
            self._tab_right.selected_index = 2
        try:
            super().on_run_button_click(b)
        finally:
            self.run_button.disabled = False
            self.collect_button.disabled = len(self._output_collections) < 1
 
    def on_image_generated(self, image, options, count, total):
        image_path = save_image_info(image, options.output_dir)
        self._output_collections.append(image_path)
        
        self._set_output_image(image_path)
        self._tab_left.selected_index = 3
        
        if count % 5 == 0:
            clear_output()
        print('> Seed = ' + str(image.argument["seed"]))
        print('> ' + image_path)
        print('    (%d / %d ... %.2f%%)'%(count + 1, total, (count + 1.) / total * 100))

            
def _createUploadView(
        label = '输入图片', 
        tooltip = '需要转换的图片的路径',
        default_path = 'resources/Ring.png', 
        upload_path = 'resources/upload.png',
        text = ''):
    
    input = widgets.Text(
        style={ 'description_width': "4rem" },
        description = label,
        description_tooltip = tooltip,
        value = default_path,
    )
    upload = widgets.FileUpload(
        accept = '.png,.jpg,.jpeg',
        description = '上传图片',
        layout = Layout(
            padding = '0.5rem',
            height = 'auto',
        )
    )
    description = widgets.HTML(
        text,
    )
    views.setLayout('col08', input)
    views.setLayout('col12', upload)
    views.setLayout('col12', description)
    input.layout.margin = '0'
    
    container = Box([
        description,
        input,
        upload,
    ], layout = Layout(
        display = 'flex',
        flex_flow = 'row wrap',
        max_width = '100%',
    ))
    # views.setLayout('col12', input_image_path)
    # input_image_path.add_class('image_path')
    
    def reset():
        input.value = default_path
        upload.value = ()
    
    def confirm():
        # 【注意】v8.0与7.5的value结构不同
        for name in upload.value:
            dict = upload.value[name]
            #检查文件类型
            path = upload_path
            if dict['metadata']['type'] == 'image/jpeg':
                path = upload_path.split('.')[0] + '.jpg'
            elif dict['metadata']['type'] == 'image/png':
                path = upload_path.split('.')[0] + '.png'
            print('保存上传到：'+path)
            with open(path, 'wb') as file:
                file.write(dict['content'])
            
            upload.value.clear()
            input.value = path
            break
        return input.value
        
    
    return Bunch({
        'container': container,
        'input': input,
        'reset': reset,
        'confirm': confirm,
        'uploador': upload,
    })

    
def createPanelImage(filename = None):
    layout = Layout(
        object_fit = 'contain',
        #object_position = 'center center',
        margin = '0 0 0 0',
        max_height = '500px',
    )
    
    _None_Image = widgets.HTML('无效的图片或未选中图片')
    
    container = HBox(
        layout = Layout(
            max_height = '500px',
            min_height = '360px',
            align_items = 'center',
            align_centent = 'center',
            justify_content = 'center',
        ),
    )
    
    def set_file(filename = None):
        if filename is None or not os.path.isfile(filename):
            container.children = (_None_Image,)
            return False
        else:
            img = widgets.Image.from_file(filename)
            img.layout = layout
            container.children = (img,)
            return True
    
    set_file(filename)
    
    return Bunch({
        'container': container,
        'set_file': set_file,
    })
    