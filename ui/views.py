import ipywidgets
from contextlib import contextmanager
from ipywidgets import (
    IntText,
    BoundedIntText,
    Layout,
    Button,
    Label,
    Box, HBox,
    # Box应当始终假定display不明
    # HBox/VBox应当仅用于【单行/单列】内容
)
from traitlets import Bunch, directional_link
from .model_collection import model_collection
from .model_collection import default_model_list as model_name_list


sampler_list = [
    "default",
    "DPMSolver",
    "EulerDiscrete",
    "EulerAncestralDiscrete",
    "PNDM",
    "DDIM",
    "LMSDiscrete",
    "HeunDiscrete",
    "KDPM2AncestralDiscrete",
    "KDPM2Discrete"
]
_DefaultLayout = {
    'col04': {
        'flex':  "4 4 30%",
        'min_width':  "6rem",    #480/ sm-576, md768, lg-992, xl-12000
        'max_width':  "calc(100% - 0.75rem)",
        'margin':  "0.375rem",
        'align_items':  "center"
    },
    'col06': {
        'flex':  "6 6 45%",
        'min_width':  "9rem",   #手机9rem会换行
        'max_width':  "calc(100% - 0.75rem)",
        'margin':  "0.375rem",
        'align_items':  "center"
    },
    'col08': {
        'flex':  "8 8 60%",
        'min_width':  "12rem",
        'max_width':  "calc(100% - 0.75rem)",
        'margin':  "0.375rem",
        'align_items':  "center"
    },
    'col12': {
        'flex':  "12 12 90%",
        'max_width':  "calc(100% - 0.75rem)",
        'margin':  "0.375rem",
        'align_items':  "center"
    },
    'btnV5': {}, #见css
    'widget-wrap': {}, #见css
}

# 为工具设置布局，并标记dom class
def setLayout(layout_names, widget):
    _lists = layout_names if isinstance(layout_names, list) or \
        isinstance(layout_names, tuple) \
        else (layout_names,)
    
    for name in _lists:
        if name not in _DefaultLayout: 
            raise Exception(f'未定义的layout名称：{name}')
        
        styles = _DefaultLayout[name];
        
        for key in styles:
            setattr(widget.layout, key, styles[key])
        
        widget.add_class(name)
    
_description_style = { 'description_width': "4rem" }
_Views = {
    # Textarea 
    "prompt": {
        "__type": 'Textarea',
        "class_name": 'prompt',
        "layout": {
            "flex": '1',
            "min_height": '10rem',
            "max_width": 'calc(100% - 0.75rem)',
            "margin": '0.375rem',
            "align_items": 'stretch'
        },
        "style": _description_style,
        "description": '正面描述' ,
        "description_tooltip": '仅支持(xxx)、(xxx:1.2)、[xxx]三种语法。设置括号格式可以对{}进行转换。',
    },
    "negative_prompt": {
        "__type": 'Textarea',
        "class_name": 'negative_prompt',
        "layout": {
            "flex": '1',
            "max_width": 'calc(100% - 0.75rem)',
            "margin": '0.375rem',
            "align_items": 'stretch'
        },
        "style": _description_style,
        "description": '负面描述',
        "description_tooltip": '使生成图像的内容远离负面描述的文本',
    },
    "replacement": {
        "__type": 'Textarea',
        "class_name": 'replacement',
        "layout_name": 'col12',
        "style": _description_style,
        "description": '描述词' ,
        "description_tooltip": '仅支持(xxx)、(xxx:1.2)、[xxx]三种语法。设置括号格式可以对{}进行转换。',
    },
    
    # Text
    "concepts_library_dir": {
        "__type": 'Text',
        "class_name": 'concepts_library_dir',
        "layout_name": 'col08',
        "style": _description_style,
        "description": '风格权重',
        "description_tooltip": 'TextualInversion训练的、“风格”或“人物”的权重文件路径',
        "value": 'outputs/textual_inversion',
    },
    "output_dir": {
        "__type": 'Text',
        "class_name": 'output_dir',
        "layout_name": 'col08',
        "style": _description_style,
        "description": '保存路径',
        "description_tooltip": '用于保存输出图片的路径',
        "value": 'outputs',
    },
    "seed": {
        "__type": 'IntText',
        "class_name": 'seed',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '随机种子',
        "description_tooltip": '-1表示随机生成。',
        "value": -1,
    },
    "num_inference_steps": {
        "__type": 'BoundedIntText',
        "class_name": 'num_inference_steps',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '推理步数',
        "description_tooltip": '推理步数（Step）：生成图片的迭代次数，步数越多运算次数越多。',
        "value": 50,
        "min": 2,
        "max": 500,
    },
    "num_return_images": {
        "__type": 'BoundedIntText',
        "class_name": 'num_return_images',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '生成数量',
        "description_tooltip": '生成图片的数量',
        "value": 1,
        "min": 1,
        "max": 100,
        "step": 1,
    },
    "guidance_scale": {
        "__type": 'BoundedFloatText',
        "class_name": 'guidance_scale',
        "layout_name": 'col04',
        "style": _description_style,
        "description": 'CFG',
        "description_tooltip": '引导度（CFG Scale）：控制图片与描述词之间的相关程度。',
        "min": 0,
        "max": 50,
        "value": 7.5,
    },
    
    # Dropdown 
    "enable_parsing": {
        "__type": 'Dropdown',
        "class_name": 'enable_parsing',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '括号格式',
        "description_tooltip": '增加权重所用括号的格式，可以将{}替换为()。选择“否”则不解析加权语法',
        "value": '圆括号 () 加强权重',
        "options": ['圆括号 () 加强权重','花括号 {} 加权权重', '否'],
    },
    "fp16": {
        "__type": 'Dropdown',
        "class_name": 'fp16',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '算术精度',
        "description_tooltip": '模型推理使用的精度。选择float16可以加快模型的推理速度，但会牺牲部分的模型性能。',
        "value": 'float32',
        "options": ['float32', 'float16'],
    },
    "max_embeddings_multiples": {
        "__type": 'Dropdown',
        "class_name": 'max_embeddings_multiples',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '描述上限',
        "description_tooltip": '修改描述词的上限倍数，使模型能够输入更长更多的描述词。',
        "value": '3',
        "options": ['1','2','3','4','5'],
    },
    "sampler": {
        "__type": 'Dropdown',
        "class_name": 'sampler',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '采样器',
        "value": 'default',
        "options": sampler_list, 
    },
    "standard_size": {
        "__type": 'Dropdown',
        "class_name": 'standard_size',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '图片尺寸',
        "description_tooltip": '生成图片的尺寸',
        "value": 5120512,
        "options": [
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
    },
    "superres_model_name": {
        "__type": 'Dropdown',
        "class_name": 'superres_model_name',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '图像放大',
        "description_tooltip": '指定放大图片尺寸所用的模型',
        "value": '无',
        "options": ['falsr_a', 'falsr_b', 'falsr_c', '无'],
    },
    
    # Combobox 
    "model_name": {
        "__type": 'Combobox',
        "class_name": 'model_name',
        "layout_name": 'col08',
        "style": _description_style,
        "description": '模型名称',
        "description_tooltip": '需要加载的模型名称（清空输入框以显示更多模型）',
        "value": 'MoososCap/NOVEL-MODEL',
        "options": model_name_list,
        "ensure_option": False,
    },
    
    # Button
    "run_button": {
        "__type": 'Button',
        "class_name": 'run_button',
        "layout_name": 'btnV5',
        "button_style": 'success', # 'success', 'info', 'warning', 'danger' or ''
        "description": '生成图片！',
        "tooltip": '单击开始生成图片',
        "icon": 'check',
    },
    "run_plot": {
        "__type": 'Button',
        "class_name": 'run_plot',
        "layout_name": 'btnV5',
        "button_style": 'success',
        "description": '生成XY图',
        "tooltip": '单击开始生成XY表格图',
        "icon": 'check',
    },
    "check_button": {
        "__type": 'Button',
        "class_name": 'check_button',
        "layout_name": 'btnV5',
        "button_style": 'warning',
        "description": '检查参数',
        "tooltip": '检查行列参数是否设置异常（注意警告信息）',
        "icon": 'tasks',
    },
    "collect_button": {
        "__type": 'Button',
        "class_name": 'collect_button',
        "layout_name": 'btnV5',
        "button_style": 'info', # 'success', 'info', 'warning', 'danger' or ''
        "description": '收藏图片',
        "tooltip": '将图片转移到Favorites文件夹中',
        "icon": 'star-o',
        "disabled": True,
    },
    "train_button": {
        "__type": 'Button',
        "class_name": 'run_button',
        "layout_name": 'btnV5',
        "button_style": 'success', # 'success', 'info', 'warning', 'danger' or ''
        "description": '开始训练！',
        "tooltip": '单击开始训练任务',
        "icon": 'check'
    },
    
    # Box
    "box_gui": {
        "__type": 'Box',
        "layout": {
            "display": 'block', #Box默认值为flex
            "margin": '0 45px 0 0',
        },
    },
    "box_main": {
        "__type": 'Box',
        "layout": {
            "display": 'flex',
            "flex_flow": 'row wrap', #HBox会覆写此属性
            "align_items": 'center',
            "max_width": '100%',
        },
    },
    
}

SHARED_STYLE_SHEETS = '''
@media (max-width:576px) {
    {root} {
        margin-right: 0 !important;
    }
    {root} .widget-text,
    {root} .widget-dropdown,
    {root} .widget-hslider,
    {root} .widget-textarea {
        flex-wrap: wrap !important;
        height: auto;
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    {root} .widget-text > label,
    {root} .widget-dropdown > label,
    {root} .widget-hslider > label,
    {root} .widget-textarea > label {
        width: 100% !important;
        text-align: left !important;
        font-size: small !important;
    } /* TODO: 合并为.widget-wrap-sm */
    {root} .col04,
    {root} .col06 {
        /*手机9rem会换行*/
        min-width: 6rem !important; 
    }
}
{root} {
    background-color: var(--jp-layout-color1);
}
{root} .widget-text > label,
{root} .widget-text > .widget-label {
    user-select: none;
}

/* 控件换行 */
{root} .widget-wrap {
    flex-wrap: wrap !important;
    height: auto;
}
{root} .widget-wrap > label {
    width: 100% !important;
    text-align: left !important;
}

/* bootcss v5 */
{root} button.btnV5.jupyter-button.widget-button
{
    height:auto;
    font-weight: 400;
    line-height: 1.5;
    text-align: center;
    vertical-align: middle;
    padding: .375rem .75rem;
    font-size: 1rem;
}
{root} button.btnV5.btn-sm.jupyter-button.widget-button
{
    padding: .25rem .5rem;
    font-size: .875rem;
}
{root} .jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {
    padding: .5rem 0;
    text-align: center;
    transition: color .15s ease-in-out,background-color .15s ease-in-out,border-color .15s ease-in-out;
}
'''

CUSTOM_OPTIONS = ('__type','class_name', 'layout_name')
def _mergeViewOptions(defaultOpt,kwargs):
    r = {}
    r.update(defaultOpt)
    
    for k in kwargs:
        r[k] = kwargs[k]
        if (k in defaultOpt) and (type(defaultOpt[k]) == 'dict') \
        and (type(kwargs[k]) == 'dict'):
            r[k] = {}
            r[k].update(defaultOpt[k])
            r[k].update(kwargs[k])
    
    #处理layout
    if ('layout' in r) and (type(r['layout']) == 'dict'):
        r.layout = Layout(**r['layout'])
    
    #提取非ipywidgets参数
    r2 = {}
    for k in CUSTOM_OPTIONS:
        if (k in r):
            r2[k] = r.pop(k)
    
    return (r, r2)

def getOptionDescription(name):
    return _Views[name]["description"] \
        if (name in _Views) \
        and ("description" in _Views[name]) \
        else ''
    
def createView(name, value = None, **kwargs):
    assert name in _Views, f'未定义的View名称 {name}'
    assert '__type' in _Views[name], f'View {name} 没有声明组件类型'
    
    # 合并参数
    args, options = _mergeViewOptions(_Views[name], kwargs)
   
    # 反射
    __type = _Views[name]['__type']
    assert hasattr(ipywidgets, __type), f'View {name} 声明的组件{__type}未被实现'
    ctor = getattr(ipywidgets, __type)
    
    if value is None:
        pass
    elif hasattr(ctor, 'value'):
        args['value'] = value
    elif hasattr(ctor, 'children'):
        args['children'] = value
    
    #实例化
    widget = ctor(**args)
    
    # 添加DOM class名
    if 'class_name' in options:
        widget.add_class(options['class_name'])
    # 设置预设布局
    if 'layout_name' in options:
        setLayout(options['layout_name'], widget)
    
    return widget

DEFAULT_BADWORDS = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
def createPromptsView(value = '', negative_value = ''):

    style_sheets = '''
@media (max-width:576px) {
    {root} .box_prompts .prompt > textarea {
        min-height:8rem;
        margin-left:2rem!important;
    }
    {root} .box_prompts .negative_prompt > textarea {
        margin-left:2rem!important;
    }
    {root} .box_prompts > .box_wrap_quikbtns {
        margin-left: 0 !important;
    }
    {root} .box_prompts > .box_wrap_quikbtns > button {
        padding: 0 !important;
    }
}
'''
    
    prompt = createView('prompt', value = value)
        
    negative_prompt = createView('negative_prompt', value = negative_value) 
    
    # 按钮
    btnGoodQuality = Button(
        description= '',
        tooltip='填充标准质量描述',
        icon='palette',
        layout = Layout(
            #不支持的属性？ position = 'absolute',
            height = '1.8rem',
            width = '1.8rem',
            margin = '-11rem 0 0 0'
        )
    )
    btnBadwards = Button(
        description= '',
        tooltip='填充标准负面描述',
        icon='paper-plane',
        layout = Layout(
            #不支持的属性？ position = 'absolute',
            height = '1.8rem',
            width = '1.8rem',
            margin = '-2rem 0px 0rem -1.8rem'
        )
    )
    def fill_good_quality(b):
        if not prompt.value.startswith('masterpiece,best quality,'):
            prompt.value = 'masterpiece,best quality,' + prompt.value
    def fill_bad_words(b):
        negative_prompt.value = DEFAULT_BADWORDS
        
    btnGoodQuality.on_click(fill_good_quality)
    btnBadwards.on_click(fill_bad_words)
        
    box_wrap_quikbtns = Box([
                            btnGoodQuality,btnBadwards,
                        ], layout = Layout(
                            margin = '0 1rem',
                            height = '0',
                            overflow = 'visible'
                        ));
    box_wrap_quikbtns.add_class('box_wrap_quikbtns')
    
    container = Box([
        HBox([prompt]),
        HBox([negative_prompt]),
        box_wrap_quikbtns,
    ])
    container.layout.display = 'block';
    container.add_class('box_prompts')
    
    return Bunch({
        'container': container,
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'style_sheets': style_sheets,
    })
    
def _create_WHView(width_value = 512, height_value = 512):
    style_sheets = '''
@media (max-width:576px) {
    {root} .box_width_height {
        flex: 8 8 60% !important;
    }
}
'''
    _layout = Layout(
        flex = '1 0 2rem',
        width = '2rem',
    )
    w_width = BoundedIntText(
        layout=_layout,
        value=width_value,
        min=64,
        max=1440,
        step=16,
    )
    w_height = BoundedIntText(
        layout=_layout,
        value=height_value,
        min=64,
        max=1440,
        step=16,
    )
    
    def validate(change):
        num = change.new % 8
        if change.new < 64:
            change.owner.value = 64
        elif num == 0:
            pass
        elif num < 4:
            change.owner.value = change.new - num
        else:
            change.owner.value = change.new - num + 8
    w_width.observe(validate,names = 'value')
    w_height.observe(validate,names = 'value')
        
    container = HBox([
        w_width,
        Label(
            value = 'X',
            layout = Layout(
                flex='0 0 auto',
                padding='0 0.75rem'
            ),
        ),
        w_height,
    ])
    setLayout('col04', container)
    container.add_class('box_width_height')
    
    return Bunch({
        'container': container,
        'width': w_width,
        'height': w_height,
        'style_sheets': style_sheets,
    })
    
def _create_WHView_for_img2img(width_value = -1, height_value = -1):
    style_sheets = '''
{root} .box_width_height > .widget-label:first-of-type {
    text-align: right !important;
}
@media (max-width:576px) {
    {root} .box_width_height {
        flex: 8 8 60% !important;
        
        flex-wrap: wrap !important;
        height: auto;
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    {root} .box_width_height > .widget-label:first-of-type {
        width: 100% !important;
        text-align: left !important;
        font-size: small !important;
    }
}
'''
    _layout = Layout(
        flex = '1 0 2rem',
        width = '2rem',
    )
    w_width = IntText(
        layout=_layout,
        value=width_value,
    )
    w_height = IntText(
        layout=_layout,
        value=height_value,
    )
    
    container = HBox([
        Label( 
            value = '图片尺寸',
            description_tooltip = '-1表示自动检测',
            style = _description_style,
            layout = Layout(
                flex='0 0 auto',
                width = '4rem',
                # margin = '0 4px 0 0',
                margin = '0 calc(var(--jp-widgets-inline-margin)*2) 0 0',
            )
        ),
        w_width,
        Label(
            value = 'X',
            layout = Layout(
                flex='0 0 auto',
                padding='0 0.75rem'
            ),
        ),
        w_height,
    ])
    setLayout('col08', container)
    container.add_class('box_width_height')
    
    return Bunch({
        'container': container,
        'width': w_width,
        'height': w_height,
        'style_sheets': style_sheets,
    })

def createWidthHeightView(width_value = 512, height_value = 512, step64 = False):
    if step64:
        return _create_WHView(width_value, height_value)
    else:
        return _create_WHView_for_img2img(width_value, height_value)
    
def createModelNameView(pipeline = None, value = None, **kwargs):
    widget = createView('model_name', value, **kwargs)
    if pipeline is not None:
        _val = widget.value
        directional_link( (pipeline, 'model'), (widget, 'value'))
        widget.value = _val
    model_collection.load_locals()
    directional_link( (model_collection, 'models'), (widget, 'options'))
    return widget
    
    
@contextmanager
def freeze(w):
    try:
        w.disabled = True
        yield
        # 注意错误会被隐藏
    finally:
        w.disabled = False

# --------------------------------------------------
    
def Accordion(children = None, **kwargs):
    titles = None if 'titles' not in kwargs else kwargs.pop('titles')
    if children is not None: kwargs['children'] = children
    tab = ipywidgets.Accordion(**kwargs)
    if titles is not None:
        for i in range(len(titles)):
            tab.set_title(i, titles[i])
    return tab
    
def Tab(children = None, **kwargs):
    titles = None if 'titles' not in kwargs else kwargs.pop('titles')
    if children is not None: kwargs['children'] = children
    tab = ipywidgets.Tab(**kwargs)
    if titles is not None:
        for i in range(len(titles)):
            tab.set_title(i, titles[i])
    return tab

def Div(children = None, **kwargs):
    if children is not None: kwargs['children'] = children
    box = Box(**kwargs)
    box.layout.display = 'block' # Box 默认flex
    return box

def FlexBox(children = None, **kwargs):
    if children is not None: kwargs['children'] = children
    
    layout = kwargs['layout'] if 'layout' in kwargs else Layout()
    layout.display = 'flex'
    layout.flex_flow = 'row wrap'# HBox覆写nowrap，Box默认nowrap
    layout.max_width = layout.max_width or '100%'
    layout.align_items = layout.align_items or 'center'
    layout.align_content = layout.align_content or 'center'
    layout.justify_content = layout.justify_content or 'center'
    kwargs['layout'] = layout
    box = Box(**kwargs)
    return box
    