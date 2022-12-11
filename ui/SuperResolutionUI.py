from .ui import StableDiffusionUI

import ipywidgets as widgets
from ipywidgets import Layout,HBox,VBox,Box 

class SuperResolutionUI(StableDiffusionUI):
    def __init__(self, pipeline, **kwargs):
        super().__init__(pipeline = pipeline)
        self.task = 'superres'
        
        #默认参数覆盖次序：
        #user_config.py > config.py > 当前args > views.py
        args = {  #注意无效Key错误
            "image_path": 'resources/image_Kurisu.png',
            "superres_model_name": 'falsr_a',
            "output_dir": 'outputs/highres',
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
            description='需要超分的图片路径' ,
            value=args['image_path'],
            disabled=False
        )

        widget_opt['superres_model_name'] = widgets.Dropdown(
            layout=layoutCol12, style=styleDescription,
            description='超分模型的名字',
            value=args['superres_model_name'],
            options=["falsr_a", "falsr_b", "falsr_c"],
            disabled=False
        )

        widget_opt['output_dir'] = widgets.Text(
            layout=layoutCol12, style=styleDescription,
            description='图片的保存路径',
            value=args['output_dir'],
            disabled=False
        )
        
        self.run_button = widgets.Button(
            description='点击超分图片！',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to run (settings will update automatically)',
            icon='check'
        )
        self.run_button.on_click(self.on_run_button_click)

        self.gui = widgets.Box([
            widget_opt['image_path'],
            widget_opt['superres_model_name'],
            widget_opt['output_dir'],
            self.run_button,
            self.run_button_out
        ], layout = Layout(
            display = "flex",
            flex_flow = "row wrap", #HBox会覆写此属性
            align_items = "center",
            # max_width = '100%',
            margin="0 45px 0 0"
        ))
