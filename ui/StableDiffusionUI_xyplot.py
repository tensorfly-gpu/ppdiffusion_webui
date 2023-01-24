import os
import shutil
import time
import random
import traitlets
import ipywidgets as widgets
from PIL import Image
from enum import Enum
from logging import warning
from traitlets import directional_link, observe, validate, Int, Bool, Bunch
from ipywidgets import Layout,HBox,Box, Label
from IPython.display import clear_output, display
from .ui import StableDiffusionUI
from .utils import save_image_info
from .model_collection import model_collection
from .views import (
    createView,
    createPromptsView,
    createWidthHeightView,
    createModelNameView,
    setLayout,
    getOptionDescription,
    SHARED_STYLE_SHEETS,
    Accordion,
    Tab,
    freeze,
)
from .xyplot import draw_xyplot, generate_test_image


MAX_DISPLAY_COUNT = 50
UNSET_TEXT = '(None)'

KEY_SORTED = (
    'replacement',
    '$X',
    '$Y',
    'num_inference_steps',
    'guidance_scale',
    'sampler',
    'fp16',
    'seed',
    'enable_parsing',
    'max_embeddings_multiples',
    'model_name',
)


def append_unset_option(widget):
    widget.options = (UNSET_TEXT,) + widget.options
    widget.value = UNSET_TEXT
    widget.description_tooltip = f'设为 {UNSET_TEXT} 表示继承公共参数'

def create_dropdown(name):
    widget = createView(name,
       description_tooltip = f'设为 {UNSET_TEXT} 表示继承公共参数'
    )
    widget.options = (UNSET_TEXT,) + widget.options
    widget.value = UNSET_TEXT
    return widget
    
def get_key_description(keys):
    if isinstance(keys, str):
        return getOptionDescription(keys) or keys
    return str([(getOptionDescription(x) or x) for x in keys])
    
def log_xydata(data):
    for k in KEY_SORTED:
        if k in data:
            print(f'{get_key_description(k)} =', data[k])
    
def check_xyplot_datas(xDatas, yDatas, prompt):
    critical = False
    
    _xrep = False
    _yrep = False
    
    if (len(xDatas) * len(yDatas) == 1):
        warning('1行1列这怎么跑xy图？')
        critical = True
    
    # 检查行列参数完全相同
    for i in range(len(xDatas)):
        _xrep = _xrep or 'replacement' in xDatas[i]
        for j in range(i):
            if xDatas[i] == xDatas[j]:
                critical = True
                warning(f'{j+1}列与{i+1}列参数完全相同')
                log_xydata(xDatas[i])
    for i in range(len(yDatas)):
        _yrep = _yrep or 'replacement' in yDatas[i]
        for j in range(i):
            if yDatas[i] == yDatas[j]:
                critical = True
                warning(f'{j+1}行与{i+1}行参数完全相同')
                log_xydata(yDatas[i])
    
    # 检查$X$Y
    if _xrep and ('$X' not in prompt):
        print('描述词中不包含$X，$X将被追加到正面描述的最后')
    if _yrep and ('$Y' not in prompt):
        print('描述词中不包含$Y，$Y将被追加到正面描述的最后')
                
                
    for i in range(len(xDatas)):
        xdata = xDatas[i]
        xkeys = xdata.keys()
        print('==================================================')
        for j in range(len(yDatas)):
            if j > 0: print('--------------------------------------------------')
            print(f'# {i+1}列{j+1}行\n')
            ydata = yDatas[j]
            
            # 打印参数
            _data = xdata.copy()
            if 'replacement' in _data:
                _data['$X'] = _data.pop('replacement')
            
            _data.update(ydata)
            if 'replacement' in _data:
                _data['$Y'] = _data.pop('replacement')
            log_xydata(_data)
            
            # 检查参数覆盖
            _keys = xkeys & ydata.keys()
            _keys.discard('replacement')
            if 'model_name' in _keys:
                warning(f'模型名称应当只设置列（{i+1}）或者只设置行（{j+1}）')
                print('模型名称应当只设置列或者只设置行')
                critical = True
                _keys.discard('model_name')
            if _keys:
                print(f'> 行列同时设置了参数{get_key_description(_keys)}，行参数将覆盖列参数')
                

    return critical

class XYTabView(Box):
    index = Int(0)
    total = Int(1)
    
    def __init__(self, isX = True):
        super().__init__()
        self.widget_opt = {}
        self.index = 0
        self.total = 1
        self.data_list = [{}]
        self.placeholder = '$X' if isX else '$Y'
        
        widget_opt = self.widget_opt
        placeholder = self.placeholder
        widget_opt['replacement'] = createView('replacement',
            value = '',
            description = '替换'+placeholder,
            description_tooltip = f'用于替换公共描述词中的{placeholder}'
        )
        widget_opt['num_inference_steps'] = createView('num_inference_steps',
            value = 0,
            min = 0,
            description_tooltip = '设为 0 表示继承公共参数'
        )
        widget_opt['guidance_scale'] = createView('guidance_scale',
            value = 0,
            description_tooltip = '设为 0 表示继承公共参数'
        )
        widget_opt['seed'] = createView('seed',
            value = 0,
            description_tooltip = '设为 0或-1 表示继承公共参数'
        )
        widget_opt['fp16'] = create_dropdown('fp16')
        widget_opt['sampler'] = create_dropdown('sampler')
        widget_opt['enable_parsing'] = create_dropdown('enable_parsing')
        widget_opt['max_embeddings_multiples'] = create_dropdown('max_embeddings_multiples')
        
        widget_opt['model_name'] = create_dropdown('model_name')
        directional_link(
            (model_collection, 'models'), 
            (widget_opt['model_name'], 'options'),
            lambda x: (UNSET_TEXT,) + x,
        )
        
        w_pagination = self._renderPagination()
        btn_reset = widgets.Button(
            description='重 置',
            layout=Layout(
                flex = '0 1 auto',
            ),
        )
        btn_reset_all = widgets.Button(
            description='全部重置',
            button_style='warning',
            layout=Layout(
                flex = '0 1 auto',
            ),
        )
        btn_reset.add_class('btnV5')
        btn_reset.add_class('btn-sm')
        btn_reset_all.add_class('btnV5')
        btn_reset_all.add_class('btn-sm')
        btn_reset.on_click(lambda b: self.page_reset())
        btn_reset_all.on_click(lambda b: self.reset_all())
        
        self.layout = Layout(
            display = 'flex',
            flex_flow = 'row wrap', #HBox会覆写此属性
            max_width = '100%',
            
            # min_height = '360px',
            # max_height = '455px',
            # height = '455px',
            align_items = 'center',
            align_content = 'center',
        )
        self.children = (
            w_pagination,
            widget_opt['replacement'],
            
            widget_opt['num_inference_steps'],
            widget_opt['guidance_scale'],
            widget_opt['sampler'],
            
            widget_opt['fp16'],
            widget_opt['seed'],
            widget_opt['enable_parsing'],
            widget_opt['max_embeddings_multiples'],
            widget_opt['model_name'],
            
            HBox(
                (btn_reset, btn_reset_all),
                layout = Layout(
                    justify_content = 'space-around',
                    width = '100%',
                )
            ),
        )
        
    def _renderPagination(self):
        # 翻页部分
        w_index = widgets.BoundedIntText(
            # description = placeholder,
            min = 1,
            layout = Layout(
                flex = '0 1 3rem',
                width = '3rem',
            )
        )
        w_total = Label(
            layout = w_index.layout
        )
        directional_link((self, 'index'), (w_index, 'value'), lambda x: x+1)
        directional_link((self, 'total'), (w_index, 'max'), lambda x: x+1)
        directional_link((self, 'total'), (w_total, 'value'), lambda x: str(x))
        w_index.observe(
            lambda x: (x.new - 1 != self.index) and self.page_to(x.new - 1),
            names = 'value',
        )
        btn_up = widgets.Button(
            # description = '上',
            tooltip = '前一页',
            icon = 'chevron-left',
            layout = Layout(
                flex = '0 1 4rem',
                width = '4rem',
            ),
        )
        btn_down = widgets.Button(
            # description = '下',
            tooltip = '后一页',
            icon = 'chevron-right',
            layout = btn_up.layout
        )
        btn_up.on_click(lambda b: self.page_up())
        btn_down.on_click(lambda b: self.page_down())
        return HBox(
            (
                btn_up,
                Box(
                    (
                        Label('列' if self.placeholder == '$X' else '行'),
                        w_index,
                        Label('/'),
                        w_total,
                    ),
                    layout = Layout(
                        flex = '0 1 auto',
                    ),
                ),
                btn_down,
            ),
            layout = Layout(
                width = '100%',
                margin = '0 0 1.5rem 0',
                justify_content = 'space-around',
            )
        )
        
    def get_datas(self):
        # 保存
        data = self.data_list[self.index]
        self.serialize_to(data)
        
        return tuple(dict(x) for x in self.data_list)
        
    def serialize_to(self, data):
        data.clear()
        for key in (
            'num_inference_steps', 'guidance_scale', 'seed',
        ):
            val = self.widget_opt[key].value
            if val > 0:
                data[key] = val
        
        for key in (
            'fp16','sampler','enable_parsing','max_embeddings_multiples','model_name','replacement'
        ):
            val = self.widget_opt[key].value
            if (UNSET_TEXT != val) and ('' != val):
                data[key] = val
        
        return data
    
    def deserialize_from(self, data = None):
        data = {} if data is None else data
        
        for key in (
            'num_inference_steps', 'guidance_scale', 'seed',
        ):
            self.widget_opt[key].value = data[key] if key in data else 0
            
        for key in (
            'fp16','sampler','enable_parsing','max_embeddings_multiples','model_name'
        ):
            self.widget_opt[key].value = data[key] if key in data else UNSET_TEXT
        
        self.widget_opt['replacement'].value = data['replacement'] if 'replacement' in data else ''
        
    def page_conform(self):
        # 保存
        data = self.data_list[self.index]
        self.serialize_to(data)
        
    def page_reset(self):
        data = self.data_list[self.index]
        data.clear()
        self.deserialize_from(data)
        
    def reset_all(self):
        data = {}
        self.data_list.clear()
        self.data_list.append(data)
        self.deserialize_from(data)
        self.index = 0
        self.total = 1
        
    def page_down(self):
        return self.page_to(self.index + 1)
        
    def page_up(self):
        return self.page_to(self.index - 1)
        
    def page_to(self, index = 0):
        last = self.index
        total = self.total
        
        # 页码不变 => 拒绝
        if index == last or index < 0: return False
        
        # 保存
        data = self.data_list[self.index]
        self.serialize_to(data)
        
        # 当前页为空，且是最后一页
        if not data and (last == total - 1):
            # 向后翻 => 当前页不是第一页 => 拒绝
            if (index == total):
                if (last != 0): return False
                
            # 向前翻 => 删除空页
            elif (index < last):
                del(self.data_list[last])
                total = len(self.data_list)    #更新总页数
            
        # 翻页
        assert index <= total
        if index == total:
            self.data_list.append({})    #新页面
            total = len(self.data_list)    #更新总页数
            
        data = self.data_list[index]
        self.deserialize_from(data)
        
        # 最后应用self.index变化，因为会引发observe事件
        self.index = index
        self.total = total  # total应当在index之后，因为会引发UI上的index的变化
        return True
            
        
class StableDiffusionUI_xyplot(StableDiffusionUI):
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
        # widget_opt = self.widget_opt

        self._generateControls(args)
        self._generateTabpanel(args)
        
        # 样式表
        STYLE_SHEETS = ('<style>' \
                + SHARED_STYLE_SHEETS \
                + STYLE_SHEETS \
                + self.view_prompts.style_sheets \
                + self.view_width_height.style_sheets \
                + '</style>'
            ).replace('{root}', '.' + CLASS_NAME)
        
        #
        self.collection_view.visible = False
        self.gui = createView("box_gui", 
            class_name = CLASS_NAME,
            children = [
                widgets.HTML(STYLE_SHEETS),
                self._tab,
                self.collection_view,
                self.run_button_out,
            ], 
        )
        
    # 生成主要控件
    def _generateControls(self, args):
        widget_opt = self.widget_opt
        
        # 提示词部分
        view_prompts = createPromptsView(
            value = args['prompt'],
            negative_value = args['negative_prompt'],
        )
        widget_opt['prompt'] = view_prompts['prompt']
        widget_opt['negative_prompt'] = view_prompts['negative_prompt']
        self.view_prompts = view_prompts
        
        # 图片尺寸部分
        view_width_height = createWidthHeightView(
            width_value = args['width'], 
            height_value = args['height'], 
            step64 = True,
        )
        widget_opt['width'] = view_width_height['width']
        widget_opt['height'] = view_width_height['height']
        self.view_width_height = view_width_height
        
        widget_opt['model_name'] = createModelNameView(
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
            widget_opt[key] = createView(key)
            if key in args:
                widget_opt[key].value = args[key]
        
        
        # 事件处理绑定
        directional_link( 
            (widget_opt['standard_size'], 'value'),
            (widget_opt['width'], 'value'),
            lambda x: x // 10000
        )
        directional_link( 
            (widget_opt['standard_size'], 'value'),
            (widget_opt['height'], 'value'),
            lambda x: x  % 10000
        )
        def on_seed_change(change):
            if change.new != -1:
                widget_opt['num_return_images'].value = 1
        def on_num_return_images(change):
            if change.new != 1:
                widget_opt['seed'].value = -1
        widget_opt['seed'].observe(on_seed_change, names='value')
        widget_opt['num_return_images'].observe(on_num_return_images, names='value')
        
        # 按钮x4
        self.run_button = createView('run_button',
            description = '文生图',
            tooltip = '单击开始从文字生成图片'
        )
        self.run_button.on_click(self.on_run_button_click)
        
        self.collect_button = createView('collect_button')
        self.collect_button.on_click(self.on_collect_button_click)
        
        self.check_button = createView('check_button')
        self.check_button.on_click(self.on_check_button_click)
        
        self.run_plot = createView('run_plot')
        # self.run_plot.on_click(self.test_run_plot_click)
        self.run_plot.on_click(self.on_run_plot_click)
        
        self.collection_view = ImageCollectionView(
            output_display = self.run_button_out,
            callback_set_seed = self.set_seed
        )
        directional_link(
            (self.collection_view, 'total'),
            (self.collect_button, 'disabled'),
            lambda x: x < 1
        )
        
    def _generateTabpanel(self, args):
        widget_opt = self.widget_opt
        setLayout('col12', self.view_prompts.container)
        widget_opt['prompt'].layout.max_width = '100%'
        widget_opt['prompt'].layout.margin = '0.375rem 0'
        widget_opt['negative_prompt'].layout.max_width = '100%'
        widget_opt['negative_prompt'].layout.margin = '0.375rem 0'
        # TODO更新prompts的布局，使其全部嵌套在flex中，设置外壳margin而非prompt的margin
        
        panel01 = createView("box_main", 
        [
            self.view_prompts.container,
            widget_opt['standard_size'],
            self.view_width_height.container,
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
            widget_opt['concepts_library_dir'],
            HBox(
                (self.run_button,self.collect_button,),
                layout = Layout(
                    justify_content = 'space-around',
                    width = '100%',
                )
            ),
        ])
        
        self._tabX = XYTabView()
        self._tabY = XYTabView(False)
        accordion = Accordion(
                (self._tabX , self._tabY),
                titles = ('列参数','行参数'),
                layout = Layout(
                    margin = '0.5rem 0',
                    max_width = '100%',
                    width = '100%',
                    min_height = '360px',
                )
            )
        accordion.selected_index = 0
        panel02 = createView("box_main", 
        [
            accordion,
            HBox(
                (self.check_button,self.run_plot,),
                layout = Layout(
                    justify_content = 'space-around',
                    width = '100%',
                )
            ),
        ])
        
        self._tab = Tab(
            titles = ('文生图', 'XY图'),
            children = (
                panel01,
                panel02,
            ),
            layout = Layout(
                margin = '0',
            )
        )

    def on_check_button_click(self,b):
        with freeze(b), self.run_button_out:
            clear_output()
            check_xyplot_datas(
                self._tabX.get_datas(),
                self._tabY.get_datas(),
                self.widget_opt["prompt"].value + '\n' + self.widget_opt["negative_prompt"].value
            )
        
    def on_collect_button_click(self, b):
        with self.run_button_out:
            self.collection_view.collect_all()
            b.disabled = True

    def on_run_button_click(self, b):
        with freeze(b), self.run_button_out:
            self.collection_view.visible = False
            self.collection_view.clear()
            super().on_run_button_click(b)
            
    def on_run_plot_click(self, b):
        with freeze(b), self.run_button_out:
            clear_output()
            self.collection_view.visible = False
            self.collection_view.clear()
            x_datas = self._tabX.get_datas()
            y_datas = self._tabY.get_datas()
            
            # 检查参数
            fail = check_xyplot_datas(
                x_datas, y_datas,
                self.widget_opt["prompt"].value + '\n' + self.widget_opt["negative_prompt"].value
            )
            if (fail): return
            
            # 检查生成方向
            horizontal = not any(x for x in x_datas if 'model_name' in x)
            
            # 检查公共参数
            options = { x :self.widget_opt[x].value for x in self.widget_opt }
            if options['seed'] < 0: options['seed'] = random.randint(0, 2**32)
            options['num_return_images'] = 1
            options['superres_model_name'] = '无'
            
            # 生成文生图参数清单
            task_list = []
            x_total, y_total = len(x_datas), len(y_datas)
            for i in range(x_total * y_total):
                x, y = (i%x_total, i//x_total) if horizontal else (i//y_total, i%y_total)
                x, y = x_datas[x], y_datas[y]
                task_list.append(
                    self.merge_options(options,x,y)
                )
            
            # 开始生成图片
            for option in task_list:
                self.pipeline.run(
                    option, 
                    task = self.task,
                    on_image_generated = self.on_image_generated
                )
            
            # 生成完毕=>生成xy图
            images = [Image.open(x) for x in self.collection_view.all_images()]
            plot = draw_xyplot(images,x_datas,y_datas,horizontal)
            _filepath = os.path.join(
                options['output_dir'],
                time.strftime(f'XYPlot_{x_total}x{y_total}_%Y-%m-%d_%H-%M-%S.jpg'),
            )
            os.makedirs(options['output_dir'], exist_ok=True)
            plot.save(_filepath)
            self.collection_view.append_show_last(_filepath)
        
    @staticmethod
    def merge_options(base, srcX, srcY):
        result = Bunch(base)
        result.update(srcX)
        result.update(srcY)
        if 'replacement' in result: del result['replacement']
        
        prompt = result.prompt
        negative_prompt = result.negative_prompt
        _X = srcX['replacement'] if 'replacement' in srcX else ''
        _Y = srcY['replacement'] if 'replacement' in srcY else ''
        
        if _X and ('$X' not in prompt) and ('$X' not in negative_prompt):
            prompt += ' ,$X' if prompt else '$X'
        if _Y and ('$Y' not in prompt) and ('$Y' not in negative_prompt):
            prompt += ' ,$Y' if prompt else '$Y'
        
        prompt = prompt.replace('$X', _X).replace('$Y', _Y)
        negative_prompt = negative_prompt.replace('$X', _X).replace('$Y', _Y)
        
        result.prompt = prompt
        result.negative_prompt = negative_prompt
        return result
            
    def test_run_plot_click(self,b):
        with freeze(b), self.run_button_out:
            clear_output()
            self.collection_view.visible = False
            self.collection_view.clear()
            x_datas = self._tabX.get_datas()
            y_datas = self._tabY.get_datas()
            
            # 检查参数
            fail = check_xyplot_datas(
                x_datas, y_datas,
                self.widget_opt["prompt"].value + '\n' + self.widget_opt["negative_prompt"].value
            )
            if (fail): return
            
            # 检查生成方向
            horizontal = not any(x for x in x_datas if 'model_name' in x)
            
            # 检查公共参数
            options = { x :self.widget_opt[x].value for x in self.widget_opt }
            if options['seed'] < 0: options['seed'] = random.randint(0, 2**32)
            options['num_return_images'] = 1
            options['superres_model_name'] = '无'
            
            # 生成文生图参数清单
            task_list = []
            x_total, y_total = len(x_datas), len(y_datas)
            for i in range(x_total * y_total):
                x, y = (i%x_total, i//x_total) if horizontal else (i//y_total, i%y_total)
                x, y = x_datas[x], y_datas[y]
                task_list.append(
                    self.merge_options(options,x,y)
                )
            
            # 生成模拟图片
            images = []
            for option in task_list:
                time.sleep(2)
                img = generate_test_image(option)
                images.append(img)
                img.save('test_img.jpg')
                self.collection_view.append_show_last('test_img.jpg')
                # clear_output()
                time.sleep(2)
            
            # 生成完毕=>生成xy图
            # images = [Image.open(x) for x in self.collection_view.all_images()]
            plot = draw_xyplot(images,x_datas,y_datas,horizontal)
            _filepath = os.path.join(
                options['output_dir'],
                time.strftime(f'XYPlot_{x_total}x{y_total}_%Y-%m-%d_%H-%M-%S.jpg'),
            )
            os.makedirs(options['output_dir'], exist_ok=True)
            plot.save(_filepath)
            self.collection_view.append_show_last(_filepath)
            
        
    def set_seed(self,seed):
        self.widget_opt['seed'].value = seed
    
    def on_image_generated(self, image, options, count = 0, total = 1, image_info = None):
        image_path = save_image_info(image, options.output_dir)
        self.collection_view.append_show_last(image_path, image.argument)
        
        clear_output()

class ImageCollectionView(Box):

    class FileState(Enum):
        Unknown = 0
        Initial = 1
        Collected = 2
        Deleted = 4
        
    index = Int(-1)
    total = Int(0)
    seed = Int(-1)
    visible = Bool(True)
    image_state = traitlets.Enum((
        FileState.Unknown,
        FileState.Initial,
        FileState.Collected,
        FileState.Deleted
        ), default_value = FileState.Unknown)
    
    INVALID_VIEW = widgets.HTML('未选中图片或无效的图片')
    def __init__(self, output_display, callback_set_seed = None):
        super().__init__()
        self.data_list = []
        
        self.collect_dir = time.strftime('Favorites/images-%m%d/') 
        self.delete_dir = time.strftime('TrashBin/images-%m%d/') 
        
        # 翻页部分
        w_index = widgets.BoundedIntText(
            # description = placeholder,
            min = 0,
            layout = Layout(
                flex = '0 1 3rem',
                width = '3rem',
            )
        )
        w_total = Label(
            layout = w_index.layout
        )
        btn_up = widgets.Button(
            # description = '上',
            tooltip = '前一页',
            icon = 'chevron-left',
            layout = Layout(
                flex = '0 1 4rem',
                width = '4rem',
            ),
        )
        btn_down = widgets.Button(
            # description = '下',
            tooltip = '后一页',
            icon = 'chevron-right',
            layout = btn_up.layout
        )
        directional_link((self, 'index'), (w_index, 'value'), lambda x: x+1)
        directional_link((self, 'total'), (w_index, 'max'), lambda x: x)    #注意设置max会导致value的变化
        directional_link((self, 'total'), (w_total, 'value'), lambda x: str(x))
        def set_index(index):
            if self.total < 1 or index < 0: return
            index = min(index, self.total - 1)
            if self.index != index:
                self.index = index
        w_index.observe(lambda c: set_index(c.new-1),names = 'value')
        btn_up.on_click(lambda b: set_index(self.index-1))
        btn_down.on_click(lambda b: set_index(self.index+1))
        
        
        # 工具按钮
        btn_collect = widgets.Button(
            icon = 'star-o',
            button_style = 'info',
            tooltip = '将图片转移到Favorites文件夹中',
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
            description = '',
            tooltip = '设置随机种子',
            layout = Layout(
                width= 'auto',
                height = 'auto',
                margin = '0',
            )
        )
        directional_link(
            (self, 'seed'), 
            (btn_seed, 'description'),
            lambda x: '' if x < 0 else str(x)
        )
        directional_link(
            (self, 'seed'), 
            (btn_seed, 'disabled'),
            lambda x: x < 0
        )
        directional_link(
            (self, 'image_state'),
            (btn_collect, 'disabled'),
            lambda x: x is self.FileState.Collected or x is self.FileState.Unknown
        )
        directional_link(
            (self, 'image_state'),
            (btn_delete, 'disabled'),
            lambda x: x is self.FileState.Deleted or x is self.FileState.Unknown
        )
        def _on_btn_click(b):
            if self.total < 1 or self.index < 0: return;
            with output_display:
                self.image_state = self.move_image(
                    self.data_list[self.index],
                    self.FileState.Collected if b is btn_collect else self.FileState.Deleted
                )
        btn_collect.on_click(_on_btn_click)
        btn_delete.on_click(_on_btn_click)
        btn_seed.on_click(lambda b: (self.seed >= 0) and callback_set_seed and callback_set_seed(self.seed) )
        
        
        self.box_toolbar = HBox(
            (
                btn_up, w_index, Label('/'), w_total, btn_seed, btn_down,
                Label(layout = Layout(flex = '10 10 auto')),
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
        self.box_image =  HBox(
            (self.INVALID_VIEW,),
            layout = Layout(
                min_height = '360px',
                align_items = 'center',
                align_centent = 'center',
                justify_content = 'center',
            ),
        )
        
        self.layout.display = 'block'
        self.layout.width = '100%'
        self.layout.border = 'var(--jp-border-width) solid var(--jp-border-color1)'
        directional_link( 
            (self, 'visible'),
            (self, 'children'),
            lambda x: x and (
                self.box_toolbar,
                self.box_image,
            ) or tuple(),
        )
        
    @validate('index')
    def _validate_index(self, proposal):
        return min(
            proposal['value'], 
            self.total -1,
            len(self.data_list) -1,
        )
    
    @observe('index')
    def _on_index_change(self, change):
        index = change.new
        seed = -1
        state = self.FileState.Unknown
        if (index >= 0) and (self.total > 0):
            data = self.data_list[index]
            if 'seed' in data.image_argument:
                seed = data.image_argument['seed']
            state = data.state
        
        self.seed = int(seed)
        self.image_state = state
        return self.render()
        
    def last(self):
        self.index = self.total - 1
        
    def append(self, image_path, image_argument = None):
        data = Bunch(
            image_path = image_path,
            image_argument = image_argument or {},
            state = self.FileState.Initial,
        )
        self.data_list.append(data)
        self.total = len(self.data_list)
        
        if self.total == 1:
            self.index = 0
        elif (self.total) % MAX_DISPLAY_COUNT == 0: 
            self.clear_image_cache()
            
    def append_show_last(self, image_path, image_argument = None):
        self.append(image_path, image_argument)
        self.index = self.total - 1
        self.visible = True
            
    def clear(self):
        self.index = -1
        # 注意减少total时，始终先更新index后更新total
        self.data_list.clear()
        self.total = len(self.data_list)
        
    def clear_image_cache(self):
        for data in self.data_list[:- MAX_DISPLAY_COUNT]:
            if 'image' in data:
                del data['image']
        
    def render(self):
            
        # 更换图片
        image = self.INVALID_VIEW
        if (self.total > 0) and (self.index >= 0):
            try:
                data = self.data_list[self.index]
                image = data['image'] if 'image' in data else \
                    widgets.Image.from_file(data['image_path'])
                data['image'] = image
                image.layout.margin = '0'
            finally:
                pass
        
        # 更新视图
        self.box_image.children = ( image, )
        
    def move_image(self, data, dest_state):
        if dest_state is data.state or data.state is self.FileState.Unknown: 
            return data.state
        elif dest_state is self.FileState.Collected:
            dir = self.collect_dir
            info = '收藏图片到 ' + dir
        elif dest_state is self.FileState.Deleted:
            dir = self.delete_dir
            info = '移除图片到 ' + dir
        else:
            raise ValueError()
        
        image_path = data.image_path
        image_name = os.path.basename(image_path)
        if not os.path.isfile(image_path):
            print(f'未找到文件：{image_path}')
            warning(f'未找到文件：{image_path}')
            data.state = self.FileState.Unknown
            return data.state
        
        # 创建文件夹
        dir = './' + dir
        os.makedirs(dir, exist_ok=True)
        
        # 转移文件
        shutil.move(image_path, dir)
        print(info + image_name)
        txt_file = image_path.rpartition('.')[0] + '.txt'
        if os.path.isfile(txt_file):
            shutil.move(txt_file, dir)
            
        # 更新按钮状态
        data.state = dest_state
        data.image_path = os.path.join(dir,image_name)
        return dest_state
        
    def collect_all(self, data):
        if self.total < 1 or self.index < 0: return
        for d in self.data_list:
            self.move_image(d, self.FileState.Collected)
        
        self.image_state = self.data_list[self.index].state
        
    def all_images(self):
        return [x.image_path for x in self.data_list]
        
