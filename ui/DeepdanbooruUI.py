#请注意，本模型不对输出结果负责，也不会保存解析用的图片，一切责任由提供图片的用户自负
import ipywidgets as widgets
from ipywidgets import Layout,HBox,VBox,Box 
from .deepdanbooru_use import deepdanbooru_to_get_tags
from IPython.display import clear_output, display
from .deepdanbooru_use import parse_args as deepdanbooru_parse_args
from collections import OrderedDict

def get_widget_extractor(widget_dict):
    # allows accessing after setting, this is to reduce the diff against the argparse code
    class WidgetDict(OrderedDict):
        def __getattr__(self, val):
            x = self.get(val)
            return x.value if x is not None else None
    return WidgetDict(widget_dict)

class DeepdanbooruUI():

    def __init__(self,  **kwargs):
        self.task = 'deepdanbooru'
        self.widget_opt = OrderedDict()
        self.gui = None
        self.run_button = None
        self.run_button_out = widgets.Output()
        
        #self.parse_args = None
        self.parse_args = deepdanbooru_parse_args
        #默认参数覆盖次序：
        #user_config.py > config.py > 当前args > views.py
        args = {  #注意无效Key错误
            "image_path": 'resources/image_Kurisu.png',
        }
        args.update(kwargs)
        widget_opt = self.widget_opt
        
        layoutCol12 = Layout(
            flex = "12 12 90%",
            margin = "0.5em",
            align_items = "center"
        )
        styleDescription = {
            'description_width': "9rem"
        }
        
        widget_opt['image_path'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='需要解析tag的图片路径' ,
            value=args['image_path'],
            disabled=False
        )
        
        self.run_button = widgets.Button(
            description='点击解析图片！',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to run (settings will update automatically)',
            icon='check'
        )
        self.run_button.on_click(self.on_run_button_click)

        self.gui = widgets.Box([
            widget_opt['image_path'],
            self.run_button,
            self.run_button_out
        ], layout = Layout(
            display = "flex",
            flex_flow = "row wrap", #HBox会覆写此属性
            align_items = "center",
            # max_width = '100%',
            margin="0 45px 0 0"
        ))
    

    def run(self, opt):
        args = self.parse_args()
        for k, v in opt.items():
            setattr(args, k, v.value)

        if args.image_path is None:
            raise ValueError("你必须给出一个图片路径")
        deepdanbooru_to_get_tags(args)
        
    def on_run_button_click(self, b):
        with self.run_button_out:
            clear_output()
            self.run(get_widget_extractor(self.widget_opt))