
from ipywidgets import (
    Textarea,
    Text,
    IntText,
    BoundedIntText,
    BoundedFloatText,
    Dropdown,
    Combobox,
    Layout,
    Button,
    Label,
    
    Box, HBox, VBox,
)

_DefaultLayout = {
    'col04': {
        'flex':  "4 4 30%",
        'min_width':  "5rem",    #480/ sm-576, md768, lg-992, xl-12000
        'max_width':  "100%",
        'margin':  "0.5em",
        'align_items':  "center"
    },
    'col08': {
        'flex':  "8 8 60%",
        'min_width':  "10rem",
        'max_width':  "100%",
        'margin':  "0.5em",
        'align_items':  "center"
    },
    'xs-flexwrap': {}, #未实现
}

# 为工具设置布局，并标记dom class
def setLayout(layout_name, widget):
    _lists = layout_name if isinstance(layout_name, list) else [layout_name]
    
    for name in _lists:
        if layout_name not in _DefaultLayout: 
            raise Exception(f'未定义的layout名称：{layout_name}')
        
        styles = _DefaultLayout[layout_name];
        
        for key in styles:
            setattr(widget.layout, key, styles[key])
        
        widget.add_class(layout_name)
    
_description_style = { 'description_width': "4rem" }
_Views = {
    # Textarea 
    "prompt": {
        "__type": 'Textarea',
        "class_name": 'prompt',
        "layout": {
            "flex": '1',
            "min_height": '12em',
            "max_width": '100%',
            "margin": '0.5em',
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
            "max_width": '100%',
            "margin": '0.5em',
            "align_items": 'stretch'
        },
        "style": _description_style,
        "description": '负面描述',
        "description_tooltip": '使生成图像的内容远离负面描述的文本',
    },
    
    # Text
    "concepts_library_dir": {
        "__type": 'Text',
        "layout_name": 'col08',
        "style": _description_style,
        "description": '风格权重',
        "description_tooltip": 'TextualInversion训练的、“风格”或“人物”的权重文件路径',
        "value": 'outputs/textual_inversion',
    },
    "output_dir": {
        "__type": 'Text',
        "layout_name": 'col08',
        "style": _description_style,
        "description": '保存路径',
        "description_tooltip": '用于保存输出图片的路径',
        "value": 'outputs',
    },
    "seed": {
        "__type": 'IntText',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '随机种子',
        "description_tooltip": '-1表示随机生成。',
        "value": -1,
    },
    "num_inference_steps": {
        "__type": 'BoundedIntText',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '推理步数',
        "description_tooltip": '推理步数（Step）：生成图片的迭代次数，步数越多运算次数越多。',
        "value": 50,
        "min": 2,
        "max": 250,
    },
    "num_return_images": {
        "__type": 'BoundedIntText',
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
        "layout_name": 'col04',
        "style": _description_style,
        "description": 'CFG',
        "description_tooltip": '引导度（CFG Scale）：控制图片与描述词之间的相关程度。',
        "min": 0,
        "max": 100,
        "value": 7.5,
    },
    
    # Dropdown 
    "enable_parsing": {
        "__type": 'Dropdown',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '括号格式',
        "description_tooltip": '增加权重所用括号的格式，可以将{}替换为()。选择“否”则不解析加权语法',
        "value": '圆括号 () 加强权重',
        "options": ['圆括号 () 加强权重','花括号 {} 加权权重', '否'],
    },
    "fp16": {
        "__type": 'Dropdown',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '算术精度',
        "description_tooltip": '模型推理使用的精度。选择float16可以加快模型的推理速度，但会牺牲部分的模型性能。',
        "value": 'float32',
        "options": ['float32', 'float16'],
    },
    "max_embeddings_multiples": {
        "__type": 'Dropdown',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '描述上限',
        "description_tooltip": '修改描述词的上限倍数，使模型能够输入更长更多的描述词。',
        "value": '3',
        "options": ['1','2','3','4','5'],
    },
    "sampler": {
        "__type": 'Dropdown',
        "layout_name": 'col04',
        "style": _description_style,
        "description": '采样器',
        "value": 'DDIM',
        "options": ['PNDM', 'DDIM', 'LMS'],
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
        "layout_name": 'col08',
        "style": _description_style,
        "description": '模型名称',
        "description_tooltip": '需要加载的模型名称',
        "value": 'MoososCap/NOVEL-MODEL',
        "options": ['CompVis/stable-diffusion-v1-4', 'runwayml/stable-diffusion-v1-5', 'hakurei/waifu-diffusion', 'hakurei/waifu-diffusion-v1-3', 'naclbit/trinart_stable_diffusion_v2_60k', 'naclbit/trinart_stable_diffusion_v2_95k', 'naclbit/trinart_stable_diffusion_v2_115k', 'MoososCap/NOVEL-MODEL', 'IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1', 'IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1', 'ruisi/anything'],
        "ensure_option": False,
    },
    
    # Button
    "run_button": {
        "__type": 'Button',
        "class_name": 'run_button',
        "button_style": 'success', # 'success', 'info', 'warning', 'danger' or ''
        "description": '生成图片！',
        "tooltip": '单击开始生成图片',
        "icon": 'check'
    },
    
    # Box
    "box_gui": {
        "__type": 'Box',
        "layout": {
            "display": 'block',
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
    
def _mergeViewOptions(defaultOpt,kwargs):
    r = {}
    for k in defaultOpt:
        if k[0] != '_':
            r[k] = defaultOpt[k]
            #抛弃_开头的选项
    
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
    
    
    return r

    
    
def createView(name, value = None, **kwargs):
    if name not in _Views:
        raise Exception(f'未定义的View名称 {name}')
    if '__type' not in _Views[name]:
        raise Exception(f'View {name} 没有定义类型')
    
    # 合并参数
    options = _mergeViewOptions(_Views[name], kwargs)
    if value is not None: options['value'] = value
    
    # 创建实例
    widget = None
    __type = _Views[name]['__type']
    if __type == 'Textarea':
        widget = Textarea(**options)
    elif __type == 'Text':
        widget = Text(**options)
    elif __type == 'IntText':
        widget = IntText(**options)
    elif __type == 'BoundedIntText':
        widget = BoundedIntText(**options)
    elif __type == 'BoundedFloatText':
        widget = BoundedFloatText(**options)
    elif __type == 'Dropdown':
        widget = Dropdown(**options)
    elif __type == 'Combobox':
        widget = Combobox(**options)
    elif __type == 'Button':
        widget = Button(**options)
    elif __type == 'Label':
        widget = Label(**options)
    else:
        if value is not None: options['children'] = value #有没有同时要value和children的部件？
        if __type == 'Box':
            widget = Box(**options)
        elif __type == 'HBox':
            widget = HBox(**options)
        elif __type == 'VBox':
            widget = VBox(**options)
        else:
            raise Exception(f'View {name} 定义的类型{__type}未实现')
    
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
        min-height:10em;
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
            position = 'absolute',
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
            position = 'absolute',
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
    
    return {
        'container': container,
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'style_sheets': style_sheets,
    }
    

def createWidthHeightView(width_value = 512, height_value = 512, step64 = False):
    style_sheets = '''
@media (max-width:576px) {
    {root} .box_width_height {
        flex: 8 8 60% !important;
    }
}
'''
    w_width = BoundedIntText(
        layout=Layout(
            flex = '1 0 2em'
        ),
        value=width_value,
        min=64,
        max=1024,
        step=64,
    )
    w_height = BoundedIntText(
        layout=w_width.layout,
        value=height_value,
        min=64,
        max=1024,
        step=64,
    )
    
    if step64:
        def validate(change):
            num = change.new % 64
            if change.new < 64:
                change.owner.value = 64
            elif num == 0:
                pass
            elif num < 32:
                change.owner.value = change.new - num
            else:
                change.owner.value = change.new - num + 64
        w_width.observe(validate,names = 'value')
        w_height.observe(validate,names = 'value')
        
    container = HBox([
        w_width,
        Label(
            value='X',
            layout = Layout(
                flex='0 0 auto',
                padding='0 1em'
            )
        ),
        w_height,
    ])
    setLayout('col04', container)
    container.add_class('box_width_height')
    
    return {
        'container': container,
        'width': w_width,
        'height': w_height,
        'style_sheets': style_sheets,
    }
    