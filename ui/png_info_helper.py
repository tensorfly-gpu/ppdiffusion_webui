import os 
import re
import json
from json import JSONDecodeError
from PIL import Image, PngImagePlugin
from enum import Enum #IntFlag?

'''
一共六种图片信息来源的情况
1.  [Paddle] 直接输出到txt
2.  [Paddle] 直接保存到png的info中
3.  [PaddleLikeWebUI] 仿webui输出到txt
4.  [PaddleLikeWebUI] 仿webui输出到png的info[parameters]中
5.  [WebUI] webui图片
6.  [NAIFU] NAIFU图片
'''
class InfoFormat(Enum):
    Paddle = 1
    WebUI = 4
    NAIFU = 8
    
    PaddleLikeWebUI = 5
    Unknown = 255

# 所有Paddle输出的参数
PRAM_NAME_LIST = (
    'prompt',
    'negative_prompt',
    'height',
    'width',
    'num_inference_steps',
    'guidance_scale',
    'num_images_per_prompt',
    'eta',
    'seed',
    'latents',
    'max_embeddings_multiples',
    'no_boseos_middle',
    'skip_parsing',
    'skip_weighting',
    'epoch_time',
    'sampler',
    'superres_model_name',
    'model_name',
    
    # img2img
    'init_image',
)

#情况3/4 [Paddle]=>[PaddleLikeWebUI]
MAP_PARAM_TO_LABEL = {
    'prompt': '',
    'negative_prompt': 'Negative prompt: ', #用于与webui保持一致
    'num_inference_steps': 'Steps: ', #用于与webui保持一致
    'sampler': 'Sampler: ',
    'guidance_scale': 'CFG scale: ',
    'strength': 'Strength: ',
    'seed': 'Seed: ',
    'width':'width: ',
    'height':'height: ',
}
#情况3/4/5 [PaddleLikeWebUI]/[WebUI]=>[Paddle]
MAP_LAEBL_TO_PARAM = {
    'Prompt': 'prompt',
    'Negative prompt': 'negative_prompt',
    'Steps': 'num_inference_steps',
    'Sampler': 'sampler',
    'CFG Scale': 'guidance_scale',
    'CFG scale': 'guidance_scale',
    'Strength': 'strength',
    'Seed': 'seed',
    'Width':'width',
    'Height':'height',
    #webui
    'Eta': 'eta',
    'Model': 'model_name', #注意model_name可能是webui的模型，不能直接使用
    'Model hash': 'model_hash',
}
# [NAIFU]=>[Paddle]
MAP_NAIFU_TAG_TO_PARAM = {
    'steps': 'num_inference_steps',
    'scale': 'guidance_scale',
    'uc': 'negative_prompt',
}

# --------------------------------------------------
# 序列化
# --------------------------------------------------

# 输出[PaddleLikeWebUI]的样式
def serialize_to_text(params):
    """
    将参数序列化为文本，以用于保存图片。格式[PaddleLikeWebUI]
    """
    # Todo 剔除无用信息
    labels = MAP_PARAM_TO_LABEL
    info = ''
    for k in labels:
        if k in params: info += labels[k] + str(params[k]) + '\n'
    
    for k in params:
        if k not in labels: info += k + ': ' + str(params[k]) + '\n'
    
    return info
    
    
def serialize_to_pnginfo(params, existing_info = None, mark_paddle = True):
    """
    将参数序列化到图像信息中。格式[PaddleLikeWebUI]
    参数 existing_info 用于继承。
    """
    text = serialize_to_text(params)
    
    dict = {}
    if existing_info is None:
        pass
    elif 'parameters' in existing_info:
        # dict.update(existing_info)
        dict['original_parameters'] = existing_info.pop('parameters')
    elif 'prompt' in existing_info:
        # 如果是[Paddle]，那么将其转换为[PaddleLikeWebUI]，并舍弃掉[Paddle]信息
        dict['original_parameters'] = serialize_to_text(existing_info)
        for k in existing_info:
            if k not in PRAM_NAME_LIST: dict[k] = existing_info[k]
    else:
        #当做[NAIFU]处理(未知的tag不能保证数据类型是iTXt)
        for k in ('Title', 'Description', 'Software', 'Source', 'Comment'):
            if k in existing_info:
                dict[k] = existing_info[k]
    
    if mark_paddle: dict['Software'] = 'PaddleNLP'
    dict['parameters'] = text

    pnginfo = PngImagePlugin.PngInfo()
    for key, val in dict.items():
        pnginfo.add_text(key, str(val))
        
    return pnginfo

def imageinfo_to_pnginfo(info, update_format = True):
    """
    从[Image.info]生成[PngInfo]，用于继承图片信息。
    仅用于超分辨率（highres）的图片保存。
    不认识的信息会被过滤掉。
    """
    dict = {}
    if ('prompt' in info) and update_format:
        # 如果是[Paddle]，那么将其转换为[PaddleLikeWebUI]，并舍弃掉[Paddle]信息
        dict['parameters'] = serialize_to_text(info)
        for k in info:
            if k not in PRAM_NAME_LIST: dict[k] = info[k]
    # 可信的info
    for k in ('Title', 'Description', 'Software', 'Source', 'Comment', 'parameters'):
        if k in info:
            dict[k] = info[k]

    pnginfo = PngImagePlugin.PngInfo()
    for key, val in dict.items():
        pnginfo.add_text(key, str(val))
        
    return pnginfo
    
# --------------------------------------------------
# 反序列化
# --------------------------------------------------
    
    
def _parse_value(s):
    if s == 'None': return None
    if s == 'False': return False
    if s == 'True': return True
    if re.fullmatch(r'[\d\.]+', s): 
        return int(s) if s.find('.') < 0 else float(s) 
    
    return s
 
# 只支持[Paddle][PaddleLikeWebUI][WebUI]
def _deserialize_from_lines(enumerable, format_presumed = InfoFormat.Unknown):
    dict = {}
    fmt = format_presumed
    
    ln = -1
    name = 'prompt'
    for line in enumerable:
        ln += 1
        line = line.rstrip('\n')
        key, colon, val = line.partition(': ')

        if (ln == 0) and (key == 'prompt'):
            fmt = InfoFormat.Paddle
        
        # 没有冒号分隔
        if colon == '':
            if ln == 0:
                name = 'prompt'
                dict[name] = line
            elif name == 'prompt' or name == 'negative_prompt':
                dict[name] += '\n' + line #追加上一行
            elif line != '':
                #不认识的换行参数
                dict[name] += '\n' + line

        # 有冒号分隔
        elif key in PRAM_NAME_LIST:
            # 1/2原始格式
            name = key
            dict[name] = val
        elif key in MAP_LAEBL_TO_PARAM:
            # 3/4格式
            fmt = InfoFormat.PaddleLikeWebUI
            name = MAP_LAEBL_TO_PARAM[key]
            dict[name] = val
        
        # 发现标签但是不认识
        elif name == 'prompt' or name == 'negative_prompt':
            # prompt下不视为标签
            dict[name] += '\n' + line
        
        # 看着像一个标签
        elif re.fullmatch(r'\w+', name):
            # 当他是个标签
            name = key
            dict[name] = val
        else:
            dict[name] += '\n' + line #追加上一行
        
    
    # 处理webui格式（[WebUI]=>[Paddle]
    if ('num_inference_steps' in dict) and (dict['num_inference_steps'].find(', ') > -1):
        webui_text = dict['num_inference_steps']
        fmt = InfoFormat.WebUI
        webui_text = 'num_inference_steps: '+webui_text
        for pair in webui_text.split(', '):
            key, colon, val = pair.partition(': ')
            key = key if key not in MAP_LAEBL_TO_PARAM else MAP_LAEBL_TO_PARAM[key]
            dict[key] = val
            
        # 处理Size: 768x512
        if ('Size' in dict):
            size = re.split(r'\D',dict.pop('Size'))
            dict['width'] = size[0]
            dict['height'] = size[1]
    
    for k in dict:
        dict[k] = _parse_value(dict[k])
    return (dict,fmt)
    
def deserialize_from_txt(text, format_presumed = InfoFormat.Unknown):
    """ 从一段文本提取参数信息。支持格式[Paddle][PaddleLikeWebUI][WebUI] """
    return _deserialize_from_lines(text.splitlines(), format_presumed)
    
# 直接从图片Info中收集信息[Paddle]
def _collect_from_pnginfo(info):
    dict = {}
    for key in info:
        if key in PRAM_NAME_LIST:
            dict[key] = _parse_value(info[key])
    fmt = InfoFormat.Paddle if 'prompt' in dict else InfoFormat.Unknown
    return (dict,fmt)

# 从Naifu中提取信息 [NAIFU]
def _collect_from_pnginfo_naifu(info):
    if ('Description' not in info) \
    or ('Comment' not in info) \
    or ('Software' not in info):
        return ({}, InfoFormat.Unknown)
    
    try:
        data = json.loads(info['Comment'])
        data['prompt'] = info['Description']
        for key in MAP_NAIFU_TAG_TO_PARAM:
            if key in data:
                data[MAP_NAIFU_TAG_TO_PARAM[key]] = data.pop(key)
        return (data, InfoFormat.NAIFU)
    except JSONDecodeError:
        return ({}, InfoFormat.Unknown)

def deserialize_from_image(image):
    """ 从图片获取参数信息。参数为Image或文件地址。"""
    if isinstance(image, str):
        assert os.path.isfile(image), f'{image}不是可读文件'
        image = Image.open(image)
        
    if 'parameters' in image.info:  #是情况4/5 [PaddleLikeWebUI]
        return deserialize_from_txt(image.info['parameters'], InfoFormat.PaddleLikeWebUI)
    
    # [NAIFU]
    dict, fmt = _collect_from_pnginfo_naifu(image.info)
    if fmt is InfoFormat.NAIFU: return (dict, fmt)
    
    # [Paddle]
    return _collect_from_pnginfo(image.info)
    
def deserialize_from_filename(filename):
    """ 从文本文件或图像文件获取参数信息，优先从其对应的文本文件中提取。参数为文件地址。"""
    txt_path, dot, ext = filename.rpartition('.')
    txt_path += '.txt'
    if os.path.isfile(txt_path):
        with open(txt_path, 'r') as f:
            (dict,fmt) = _deserialize_from_lines(f)
            if (fmt is not InfoFormat.Unknown) or ext.lower() == 'txt':
                return (dict, fmt)
    
    return deserialize_from_image(filename)
    
    