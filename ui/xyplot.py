# from time import time
from math import ceil
from PIL import Image, ImageFont, ImageDraw
from traitlets import Bunch
from logging import warning
DEFAULT_FONT = 'wqy-microhei.ttc'

label_format = {
    'num_inference_steps': '步数: {0}',
    'guidance_scale': 'CFG: {0}',
    'sampler': '采样器: {0}',
    'fp16': '算术精度: {0}',
    'seed': '随机种子: {0}',
    'enable_parsing':  '{0}',
    'max_embeddings_multiples': '描述上限倍数: {0}',
    'model_name': '{0}',
}
# 'getbbox',
 # 'getlength',

def further_clip(lines, font, max_length):
    result = []
    if isinstance(lines, str):
        lines = lines.splitlines()
    for line in lines:
        # line_length = drawing.textlength(line, font)
        line_length = font.getlength(line)
        if line_length <= max_length:
            result.append(line)
            continue
        _pos = max_length * len(line) // line_length - 1
        _pos = int(_pos)
        result.append(line[:_pos])
        result.append(line[_pos:])
    return result   
        
    
def count_wrap(lines, font, max_length):
    # 【！】注意不检查lines中单个项是否换行
    if isinstance(lines, str):
        lines = lines.splitlines()
    
    count = 0
    max_width = 0
    for line in lines:
        count += 1
        # line_length = drawing.textlength(line, font)
        line_length = font.getlength(line)
        max_width = max(max_width, line_length)
        if max_length < line_length:
            count += 1
    return count, max_width
    
def wrap(text_or_words, font, max_length):
    words = text_or_words.split() if isinstance(text_or_words, str) else text_or_words
    if not text_or_words or not words: return []
    lines = ['']
    
    # aaaaa|aa
    # bbbbb|bb
    
    index = -1
    total = len(words)
    word = ''
    line = ''
    
    while (index < total):
        if word == '':
            index += 1
            if index >= total: break
            word = words[index].strip()
            
        line = f'{lines[-1]} {word}'.strip()
        # line_length = drawing.textlength(line, font)
        line_length = font.getlength(line)
        
        # 行长度不超过行宽 => 继续当前行并下一个单词
        if line_length <= max_length:
            lines[-1] = line
            word = ''
            continue
            
        # 行长度超过行宽
        # 单词小于行宽 => 下一行并下一个单词
        # word_length = drawing.textlength(word, font)
        word_length = font.getlength(word)
        if word_length <= max_length:
            lines.append(word)
            word = ''
            continue
            
        # 单词长度超过行宽
        # 当前行已满2/3 => 下一行并继续这个单词
        if (line_length - word_length) > max_length * 2 / 3:
            lines.append('')
            # word = word
            continue
           
        # 否 => 截断，下一行，截断新单词
        _pos = max(len(line) * max_length //line_length - 1, 1)
        _pos = int(_pos)
        # assert _pos > 1
        lines[-1] = line[:_pos]
        lines.append('')
        # print (_pos, line[:_pos], line[_pos:], word)
        word = line[_pos:]
    return lines

def key_to_label(key, data, contract_level = 0):
    val = str(data[key])
    if key == 'enable_parsing':
        return val if not val == '无' else '不计算权重'
        
    if contract_level > 0:
        if key == 'max_embeddings_multiples': return f'描述上限: {val}'
        if key == 'model_name': return val
        if key == 'sampler' and contract_level <= 1: return val
    if contract_level > 1:
        if key == 'fp16': return f'精度: {val}'
        if key == 'seed': return f'种子: {val}'
        if key == 'sampler':
            return val.replace('Discrete','').replace('Ancestral', 'A')
    
    return label_format[key].format(val) if key in label_format else f'{key}: val'
        
# --------------------------------------------------
# 检查表头
# --------------------------------------------------
        
class HeaderCheckState(Bunch):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert 'width' in self, '必须指定宽度'
        assert 'height' in self, '必须指定高度'
        
        if 'spacing_factor' not in self:
            self.spacing_factor = 0.5
            
        if 'fontsize' not in self:
            self['fontsize'] = int((self.width + self.height) // 25)
            
        if 'font' not in self:
            self.font = ImageFont.truetype(DEFAULT_FONT, self['fontsize'])
            
        if 'available_height' not in self:
            self.available_height = self.height - self.spacing * 2
            
        if 'available_width' not in self:
            self.available_width = self.width - self.spacing * 2
    
    
    def set_fontsize(self, fontsize):
        fontsize = int(fontsize)
        if self['fontsize'] == fontsize: return
        # print('setfontsize>', self['fontsize'],fontsize)
        self['fontsize'] = fontsize
        self.font = ImageFont.truetype(DEFAULT_FONT, fontsize)
        
    @property
    def spacing(self):
        return self.fontsize * self.spacing_factor // 1
        
    @property
    def max_lines(self):
        spacing = self.spacing
        fontsize = self.fontsize
        available_height = self.available_height
        return max(
            3, 
            (available_height - spacing) // (fontsize + spacing)
        )
    
    def __setattr__(self, key, value):
        if key == 'fontsize': return self.set_fontsize(value)
        return super().__setattr__(key,value)
    
class HeaderDrawState():
    def __init__(self, data, state):
        self.data = data
        self.state = state
        
        self.used_width = 0
        self.used_lines = 0
        
        self.lines = []
        self._data_words = []
        self._contract_level = -1
        
    @property
    def data_words(self):
        data = self.data
        contract_level = self.state.contract_level
        if self._contract_level != contract_level:
            self._contract_level = contract_level
            self._data_words.clear()
            self._data_words.extend(
                key_to_label(x, data, contract_level) \
                    for x in data if x != 'replacement'
            )
        return self._data_words
    
    def try_draw(self):
        if not self.data: return True
        
        data = self.data
        state = self.state
        font = state.font
        # drawing = state.drawing
        
        max_lines = state.max_lines
        available_width = state.available_width
        
        lines = self.lines
        lines.clear()
        self.used_width = 0
        
        # 计算描述词需要多少行
        if 'replacement' in data:
            _text = data['replacement']
            _count, _width = count_wrap(
                _text, font, available_width
            )
            
            if _count == 1:
                lines.append(_text)
                self.used_width = _width
            else:
                lines.extend(wrap(
                    _text, font, available_width
                ))
                self.used_width = available_width
        
        # 其他key
        words = self.data_words
        self.used_lines = len(lines)
        self.need_futher_clip = False
        
        self.count_data_wrapped = 0
        if words:
            # 其他Key都单行书写会不会超出最大行数限制？
            if (len(words) + self.used_lines <= max_lines):
                # 检查单行书写是否溢出换行
                _count, data_width = count_wrap(words, font, available_width)
                
                self.count_data_wrapped = _count - len(words)
                if (self.count_data_wrapped < 1):
                    #未发生换行
                    lines.extend(words)
                    # 最大需要的行数 => 用于后续计算列头高度
                    self.used_lines = len(lines)
                    # 最大使用的宽度 => 用于后续计算行头宽度
                    self.used_width = max(
                        data_width,
                        self.used_width
                    )
                    return True
                elif self.count_data_wrapped > 2:
                    #换行次数太多
                    pass
                elif (_count + self.used_lines <= max_lines):
                    #发生换行，且总行数小于最大行数
                    lines.extend(words)
                    # 最大需要的行数 => 用于后续计算列头高度
                    self.used_lines += _count
                    # TODO lines参数不对
                    self.need_futher_clip = True
                    # 最大使用的宽度 => 用于后续计算行头宽度
                    self.used_width = available_width
                    return True
                    
            # 其他Key换行并检查
            lines.extend(wrap(
                words, font, available_width
            ))
            self.used_lines = len(lines)
            self.used_width = available_width
        
        return self.used_lines <= max_lines
    
    def draw_at(self, drawing, xy, area_height):
        if not self.data: return
        state = self.state
        
        if not self.try_draw():
            warning(f'最终绘制时，超出绘制区域 {self.data}')
        
        if self.need_futher_clip:
            _a = further_clip(self.lines, state.font, state.available_width)
            self.lines.clear()
            self.lines.extend(_a)
            self.used_lines = len(self.lines)
            
        #重新调整行间距
        factor = area_height / self.used_lines / state.fontsize - 1
        spacing = min(0.7, max(0.2, factor)) * state.fontsize
        
        
        drawing.multiline_text(
            xy,
            '\n'.join(self.lines),
            font = state.font, 
            spacing = spacing,
            fill = (0, 0, 0), 
            anchor = "mm", 
            align = "center"
        )
   
def check_col_head(width, height, label_datas, state):
    _max_available_height = state._max_available_height
    head_list = []
    # _start_at = time()
    # _i = -1
    for data in label_datas:
        # print(f'check_col> [{_i}] {time() - _start_at}')
        # _i += 1
        
        head = HeaderDrawState(data, state)
        head_list.append(head)
        
        if head.try_draw(): continue
        
        exceed_lines = head.used_lines - state.max_lines
        
        # 检查是否需要缩短标签
        if state.contract_level >= 2:
            # 最大值，不检查
            pass
        elif head.used_lines > 5:
            # 如果总行数过多 => 缩短标签2
            state.contract_level = 2
        elif head.count_data_wrapped > 0:
            # 超行是data引起的
            state.contract_level += 2 \
                if head.count_data_wrapped > 2 \
                else 1
            if (head.count_data_wrapped >= exceed_lines) \
                and head.try_draw(): continue
        else:
            # 超行和data无关
            pass
            
        #调整行间距
        exceed_lines = head.used_lines - state.max_lines
        _size = state.fontsize
        # _factor = (
                # state.available_height / head.used_lines / _size - 1
            # )
        # if _factor >= 0.2:
            # state.spacing_factor = _factor
        # elif exceed_lines > 5:
            # state.spacing_factor = 0.2
        # elif exceed_lines > 3:
            # state.spacing_factor = 0.3
            
        while not head.try_draw():
            # 如果字号到1了仍然写不下
            if state.fontsize <= 1: break
            
            # 缩减需要的1/4字号
            _size = max(min(
                    _size * (state.max_lines / head.used_lines - 1) // 4,
                    -1
                ) + _size,
                1
            )
            state.fontsize = _size
            
            # 增加需要的行数1/4高度
            _height = state.available_height
            if _height < _max_available_height:
                _height = min(
                    _height * (head.used_lines / state.max_lines - 1) // 4 + _height,
                    _max_available_height,
                )
                state.available_height = _height
    
    # state.max_used_lines = max(x.used_lines for x in head_list)
    # print(f'check_col> [{_i}] {time() - _start_at}')
    return head_list

def check_row_head(width, height, label_datas, state):
    
    head_list = []
    # _start_at = time()
    # _i = -1
    for data in label_datas:
        # print(f'check_row> [{_i}] {time() - _start_at}')
        # _i += 1
        head = HeaderDrawState(data, state)
        head_list.append(head)
        
        if head.try_draw(): continue
        
        # 检查是否需要缩短标签
        if head.count_data_wrapped > 0:
            # 除了model_name也没什么能导致换行的了
            state.contract_level = 1
            
        #调整行间距
        # exceed_lines = head.used_lines - state.max_lines
        _size = state.fontsize
        # _factor = (
                # state.available_height / head.used_lines / _size - 1
            # )
        # if _factor >= 0.2:
            # state.spacing_factor = _factor
        # elif exceed_lines > 5:
            # state.spacing_factor = 0.2
        # elif exceed_lines > 3:
            # state.spacing_factor = 0.3
        
        while not head.try_draw():
            # 如果字号到1了仍然写不下
            if state.fontsize <= 1: break
            
            # 缩减需要的1/4字号
            _size = max(min(
                    _size * (state.max_lines / head.used_lines - 1) // 4,
                    -1
                ) + _size,
                1
            )
            state.fontsize = _size

    # state.max_used_lines = state.max_lines
    # print(f'check_row> [{_i}] {time() - _start_at}')
    return head_list
    
# --------------------------------------------------
# 绘制
# --------------------------------------------------
    
class Layout():
    LAYOUT_NAME = 'Layout'
    
    def __init__(self, **kwargs):
        self.width = kwargs['width']
        self.height = kwargs['height']
        # self.head_width = ceil()
        # self.head_height = ceil(kwargs['head_height'])
        self.padding_space = ceil((kwargs['width'] + kwargs['height']) / 50)
        self.col_num = kwargs['col_num']
        self.row_num = kwargs['row_num']
        self.head_width = 0
        self.head_height = 0
        self.padding_top = 0
        self.padding_left = 0
        
    def set_head_size(self, list_head_width, list_head_height):
        self.head_width = ceil(max(list_head_width))
        self.head_height = ceil(max(list_head_height))
        if self.head_width > 0:
            self.padding_left = self.head_width + self.padding_space * 2
        if self.head_height > 0:
            self.padding_top = self.head_height + self.padding_space * 2
    
    def recomended_head_size(self):
        width, height = self.width, self.height
        row_num, col_num = self.row_num, self.col_num
        available_height = max(
            width // 2,
            height // 2,
            min(width, height),
            0.414 * min(width, height) * row_num,
        )
        available_width = max(
            height // 2,
            min(width, height),
            width * col_num // 4,
        )
        _max_available_height = max(
            height,
            width,
            0.414 * height * row_num,
        )
        return (available_height, available_width, _max_available_height)
        
    def height_col(self, i):
        return self.head_height
        
    def height_row(self, i):
        return self.height
        
    def size(self):
        return (
            self.width * self.col_num + self.padding_left,
            self.height * self.row_num + self.padding_top,
        )
        
    def pos_img_from_i(self, index, horizontal = True):
        if horizontal:
            x = index % self.col_num
            y = index // self.col_num
        else:
            x = index // self.row_num
            y = index % self.row_num
        r = self.pos_img(x, y)
        # print((self.col_num, self.row_num), index, (x,y), r)
        return r
        
    def pos_img(self, x, y):
        assert x < self.col_num
        assert y < self.row_num
        return (
            x * self.width + self.padding_left,
            y * self.height + self.padding_top,
        )
        
    def pos_col(self, index):
        assert index < self.col_num
        x = self.padding_left + self.width * (1 + 2*index) // 2
        y = self.padding_top // 2
        return (x, y)
        
    def pos_row(self, index):
        assert index < self.row_num
        x = self.padding_left // 2
        y = self.padding_top + self.height * (1 + 2*index) // 2
        return (x, y)

class HorizontalWrappedLayout(Layout):
    UP_AND_DOWN = True
    LAYOUT_NAME = 'HorizontalWrappedLayout'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.col_num > 1
        assert self.row_num == 1
        
        self.col_num_actual = (self.col_num + 1) // 2
        # self.row_num_actual = _pos
        self.head_height = 0
        self.head_height2 = 0
        self.padding_top = 0
        self.padding_top2 = 0
    
    def set_head_size(self, list_head_width, list_head_height):
        self.head_width = ceil(max(list_head_width))
        self.head_height = ceil(max(list_head_height[:self.col_num_actual]))
        self.head_height2 = ceil(max(list_head_height[self.col_num_actual:]))
        if self.head_width > 0:
            self.padding_left = self.head_width + self.padding_space * 2
        if self.head_height > 0:
            self.padding_top = self.head_height + self.padding_space * 2
        if self.head_height2 > 0:
            self.padding_top2 = self.head_height2 + self.padding_space * 2
        
    
    def recomended_head_size(self):
        width, height = self.width, self.height
        col_num_actual = self.col_num_actual
        _width = max(
            height // 2,
            min(width, height),
            width * col_num_actual // 4,
        )
        t = super().recomended_head_size()
        return (t[0], _width, t[1])
    
    def height_col(self, i):
        if i < self.col_num_actual: return self.head_height
        return self.head_height2
        
    def size(self):
        return (
            self.width * self.col_num_actual + self.padding_left,
            self.height * 2 + self.padding_top + self.padding_top2,
        )
        
    def pos_img_from_i(self, index, horizontal = True):
        return self.pos_img(index, 0)
        
    def pos_img(self, x, y = 0):
        if x >= self.col_num_actual:
            x -= self.col_num_actual
            y = 1
        
        if y == 0:
            return (
                x * self.width + self.padding_left,
                y * self.height + self.padding_top,
            )
        elif self.UP_AND_DOWN:
            return (
                x * self.width + self.padding_left,
                y * self.height + self.padding_top,
            )
        return (
            x * self.width + self.padding_left,
            y * self.height + self.padding_top + self.padding_top2,
        )
    
    def pos_col(self, index):
        assert index < self.col_num
        x = index
        y = 0
        
        if index < self.col_num_actual:
            x, y = index, 0
        else:
            x, y = index - self.col_num_actual, 1
        
        if (y > 0) and self.UP_AND_DOWN:
            return (
                self.padding_left + self.width * (1 + 2*x) // 2,
                self.padding_top2 // 2 + 2 * self.height + self.padding_top
            )
        return (
            self.padding_left + self.width * (1 + 2*x) // 2,
            self.padding_top // 2 + y * self.height
        )
    
    def pos_row(self, index):
        assert index <= 1
        x = self.padding_left // 2
        y = self.padding_top + self.height * (1 + 2*index) // 2
        
        if (index > 0) and not self.UP_AND_DOWN:
            y += self.padding_top2
        
        return (x, y)
        
    @staticmethod
    def suit_for(**kwargs):
        col_num = kwargs['col_num']
        row_num = kwargs['row_num']
        if row_num > 1 or col_num < 6: return False
        width = kwargs['width']
        height = kwargs['height']
        # head_width = ceil(max(kwargs['iterate_head_width']))
        # head_height = ceil(max(kwargs['iterate_head_height']))
        
        return width * col_num > 4 * (height * row_num)
        
def get_layout(**kwargs):
    if HorizontalWrappedLayout.suit_for(**kwargs):
        return HorizontalWrappedLayout(**kwargs)
    return Layout(**kwargs)

    
def draw_xyplot(images, x_datas, y_datas, horizontal = True):
    
    width, height = images[0].size
    col_num = len(x_datas)
    row_num = len(y_datas)
    
    # 初始化布局
    layout = get_layout(
        width = width,
        height = height,
        col_num = col_num,
        row_num = row_num,
    )
    
    # 计算表头尺寸边界
    available_height, available_width, _max_available_height = layout.recomended_head_size()
    
    # 初始化检查参数
    colState = HeaderCheckState(
        width = width,
        height = height,
        contract_level = 0,
        available_height = available_height,
        _max_available_height = _max_available_height,
        # fontsize 默认值 (width + height) // 25
        # font 默认值
        # available_width 默认值
        spacing_factor = 0.2,
    )
    rowState = HeaderCheckState(
        width = width,
        height = height,
        contract_level = 1,
        available_width = available_width,
        spacing_factor = 0.2,
    )
    
    # 运行检查
    cols = check_col_head(width, height, x_datas, colState)
    rowState.contract_level = min(colState.contract_level, 1)
    rowState.fontsize = (colState.fontsize + rowState.fontsize) // 2
    rows = check_row_head(width, height, y_datas, rowState)
    
    # 输入检查后的表头尺寸
    layout.set_head_size(
        list_head_width = [x.used_width for x in rows],
        list_head_height = [(colState.fontsize + colState.spacing) * x.used_lines for x in cols],
    )
    
    # 绘制图片
    img_plot = Image.new("RGB", layout.size(), "white")
    
    for i in range(len(images)):
        img = images[i]
        if (i >= col_num * row_num): return
        img_plot.paste(img, layout.pos_img_from_i(i, horizontal))
        # print(layout.pos_img_from_i(i, horizontal))
    
    # 绘制表头
    drawing = ImageDraw.Draw(img_plot)
    for i in range(len(cols)):
        cols[i].draw_at(drawing, layout.pos_col(i), layout.height_col(i))
    for i in range(len(rows)):
        rows[i].draw_at(drawing, layout.pos_row(i), layout.height_row(i))
    
    # 水平换行布局需要再画一次行头
    if layout.LAYOUT_NAME == 'HorizontalWrappedLayout':
        rows[0].draw_at(drawing, layout.pos_row(1), layout.height_row(0))
        
    
    return img_plot
    

def generate_test_image(options):
    from .png_info_helper import serialize_to_text
    text = serialize_to_text(options)
    width, height = options['width'], options['height']
    
    fontsize = (width+ height)//50
    font = ImageFont.truetype(DEFAULT_FONT, fontsize)
    
    # _, __, right, bottom = font.getbbox(text)
    max_width, max_height = width - fontsize, height - fontsize
    
    # 计算高度
    count_lines = count_wrap(text.splitlines(), font, max_width)[0]
    need_height = fontsize * 1.2 * count_lines
    
    # 计算缩放
    if need_height >= max_height:
        fontsize = max(int( max_height / need_height * fontsize ), 1)
        font = ImageFont.truetype(DEFAULT_FONT, fontsize)
    
    # 开始绘制
    lines = further_clip(text.splitlines(), font, max_width)
    print('lines =',len(lines),'fontsize',fontsize ,'max_width =', max_width)
        
    img = Image.new("RGB", (width,height), "white")
    drawing = ImageDraw.Draw(img)
    drawing.multiline_text(
        (width//2, height//2),
        '\n'.join(lines),
        font = font, 
        spacing = int(fontsize*0.2),
        fill = (0, 0, 0), 
        anchor = "mm", 
        align = "left"
    )
    return img
    
    
    
def test():
# TODO
# 模型名称只能在两行内书写
# 当delta_size>5时缩短标签名
    x_datas = [
        {
            'model_name': 'ruisi/anything',
        },
        {
            'model_name': 'ruisi/anything',
        },
        {
            'model_name': 'ruisi/anything',
        },
        {
            'model_name': '222',
            'guidance_scale': '20',
        },
        {
            'model_name': '222',
            'guidance_scale': '20',
        },
        {
            'model_name': '222',
            'guidance_scale': '20',
        },
        {
            'model_name': '333',
            'guidance_scale': 30,
            'sampler': 'DDIM_33',
        },
        {
            'model_name': 'naclbit/trinart_stable_diffusion_v2_60k',
            'guidance_scale': 40,
            'sampler': 'KDPM2AncestralDiscrete',
            # 'replacement': '''
            # adadfsd
# 1 girl,(beautiful detailed eyes), detailed face,perfect_body,strappy_heels,(cross-laced legwear),(Ribbon wrapped around legs),
            # ''',
        },
    ]
    y_datas = [
        {
            'guidance_scale': '10',
            'model_name': 'naclbit/trinart_stable_diffusion_v2_60k',
        },
        # {
            # 'guidance_scale': '20',
            # 'sampler': 'EulerDiscrete',
        # },
        # {
            # 'guidance_scale': 30,
            # 'sampler': 'LMSDiscrete',
            # 'max_embeddings_multiples': 33,
            # 'replacement': 'xxxx',
        # },
    ]
    img0 = Image.open('resources/image_Kurisu.png')
    img1 = Image.open('resources/Ring.png')
    images = []
    
    for i in range(len(x_datas) * len(y_datas) // 2):
        images.append(img0)
        images.append(img1)
    
    return draw_xyplot(images, x_datas, y_datas)