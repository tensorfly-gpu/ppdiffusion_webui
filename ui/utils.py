#pt加载功能基于群内@作者版本修改 
import os 
import time
from contextlib import nullcontext, contextmanager
from IPython.display import clear_output, display
from pathlib import Path
from PIL import Image
from .png_info_helper import serialize_to_text, serialize_to_pnginfo
import paddle

_VAE_SIZE_THRESHOLD_ = 300000000       # vae should not be smaller than this
_MODEL_SIZE_THRESHOLD_ = 3000000000    # model should not be smaller than this

def compute_gpu_memory():
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return round(meminfo.total / 1024 / 1024 / 1024, 2)

def empty_cache():
    """Empty CUDA cache. Essential in stable diffusion pipeline."""
    import gc
    gc.collect()

def check_is_model_complete(path = None, check_vae_size=_VAE_SIZE_THRESHOLD_):
    """Auto check whether a model is complete by checking the size of vae > check_vae_size.
    The vae of the model should be named by model_state.pdparams."""
    path = path or os.path.join('./',os.path.basename(model_get_default())).rstrip('.zip')
    return os.path.exists(os.path.join(path,'vae/model_state.pdparams')) and\
         os.path.getsize(os.path.join(path,'vae/model_state.pdparams')) > check_vae_size

def model_get_default(base_path = '/home/aistudio/data'):
    """Return an absolute path of model zip file in the `base_path`."""
    available_models = []
    for folder in os.walk(base_path):
        for filename_ in folder[2]:
            filename = os.path.join(folder[0], filename_)
            if filename.endswith('.zip') and os.path.isfile(filename) and os.path.getsize(filename) > _MODEL_SIZE_THRESHOLD_:
                available_models.append((os.path.getsize(filename), filename, filename_))
    available_models.sort()
    # use the model with smallest size to save computation
    return available_models[0][1]

def model_vae_get_default(base_path = 'data'):
    """Return an absolute path of extra vae if there is any."""
    for folder in os.walk(base_path):
        for filename_ in folder[2]:
            filename = os.path.join(folder[0], filename_)
            if filename.endswith('vae.pdparams'):
                return filename
    return None

def model_unzip(abs_path = None, name = None, dest_path = './', verbose = True):
    """Unzip a model from `abs_path`, `name` is the model name after unzipping."""
    if abs_path is None:
        abs_path = model_get_default()
    if name is None:
        name = os.path.basename(abs_path)

    from zipfile import ZipFile
    dest = os.path.join(dest_path, name).rstrip('.zip')
    if not check_is_model_complete(dest):
        if os.path.exists(dest):
            # clear the incomplete zipfile
            if verbose: print('检测到模型文件破损, 正在删除......')
            import shutil
            shutil.rmtree(dest)
        
        if verbose: print('正在解压模型......')
        with ZipFile(abs_path, 'r') as f:
            f.extractall(dest_path)
    else:
        print('模型已存在')

def package_install(verbose = True):
    try:
        from ppdiffusers.utils import image_grid
        from paddlenlp.transformers.clip.feature_extraction import CLIPFeatureExtractor
        from paddlenlp.transformers import FeatureExtractionMixin
        
    except (ModuleNotFoundError, ImportError, AttributeError):
        if verbose: print('检测到库不完整, 正在安装库')
        os.system("pip install -U pip  -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install -U ppdiffusers paddlenlp OmegaConf --user")
        clear_output()

def diffusers_auto_update(verbose = True):
    package_install(verbose=verbose)

def try_get_catched_model(model_name):
    path = os.path.join('./models/', model_name)
    if check_is_model_complete(path):
        return path
    path = os.path.join('./', model_name)
    if check_is_model_complete(path):
        return path
    return model_name
    
@contextmanager
def context_nologging():
    import logging
    logging.disable(100)
    try:
        yield
    finally:
        logging.disable(30)
 
def save_image_info(image, path = './outputs/', existing_info = None):
    """Save image to a path with arguments."""
    os.makedirs(path, exist_ok=True)
    seed = image.argument['seed']
    filename = time.strftime(f'%Y-%m-%d_%H-%M-%S_SEED_{seed}')
    
    pnginfo_data = serialize_to_pnginfo(image.argument, existing_info)
    info_text = serialize_to_text(image.argument)
    
    with open(os.path.join(path, filename + '.txt'), 'w') as f:
        f.write('Prompt: '+ info_text)
    image_path = os.path.join(path, filename + '.png')
    image.save(image_path, 
        quality=100,
        pnginfo=pnginfo_data
    )
    return image_path
    
def ReadImage(image, height = None, width = None):
    """
    Read an image and resize it to (height,width) if given.
    If (height,width) = (-1,-1), resize it so that 
    it has w,h being multiples of 64 and in medium size.
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    # clever auto inference of image size
    w, h = image.size
    if height == -1 or width == -1:
        if w > h:
            width = 768
            height = max(64, round(width / w * h / 64) * 64)
        else: # w < h
            height = 768
            width = max(64, round(height / h * w / 64) * 64)
        if width > 576 and height > 576:
            width = 576
            height = 576
    if (height is not None) and (width is not None) and (w != width or h != height):
        image = image.resize((width, height), Image.ANTIALIAS)
    return image
    
def convert_pt_to_pdparams(path, dim = 768, save = True):
    """Unsafe .pt embedding to .pdparams."""
    path = str(path)
    assert path.endswith('.pt'), 'Only support conversion of .pt files.'

    import struct
    with open(path, 'rb') as f:
        data = f.read()
    data = ''.join(chr(i) for i in data)

    # locate the tensor in the file
    tensors = []
    for chunk in data.split('ZZZZ'):
        chunk = chunk.strip('Z').split('PK')
        if len(chunk) == 0:
            continue

        tensor = ''
        for i in range(len(chunk)):
            # extract the string with length 768 * (4k)
            tensor += 'PK' + chunk[i]
            if len(tensor) > 2 and len(tensor) % (dim * 4) == 2:
                # remove the leading 'PK'
                tensors.append(tensor[2:])

    tensor = max(tensors, key = lambda x: len(x))

    # convert back to binary representation
    tensor = tensor.encode('latin')

    # every four chars represent a float32
    tensor = [struct.unpack('f', tensor[i:i+4])[0] for i in range(0, len(tensor), 4)]
    tensor = paddle.to_tensor(tensor).reshape((-1, dim))
    if tensor.shape[0] == 1:
        tensor = tensor.flatten()

    if save:
        # locate the name of embedding
        name = ''.join(filter(lambda x: ord(x) > 20, data.split('nameq\x12X')[1].split('q\x13X')[0]))
        paddle.save({name: tensor}, path[:-3] + '.pdparams')

    return tensor

def get_multiple_tokens(token, num = 1, ret_list = True):
    """Parse a single token to multiple tokens."""
    tokens = ['%s_EMB_TOKEN_%d'%(token, i) for i in range(num)]
    if ret_list:
        return tokens
    return ' '.join(tokens)


def collect_local_module_names(base_paths = None):
    """从指定位置检索可用的模型名称，以用于UI选择模型"""
    base_paths = (
            './', 
            './models', 
            os.path.join(os.environ['PPNLP_HOME'], 'models')
        ) if base_paths is None \
        else (base_paths,) if isinstance(base_paths, str) \
        else base_paths
    
    is_model = lambda base, name: (os.path.isfile(
            os.path.join(base, name,'model_index.json')
        )) and (os.path.isfile(
            os.path.join(base, name,'vae', 'config.json')
        )) and (os.path.isfile(
            os.path.join(base, name,'unet', 'model_state.pdparams')
        ))
        
    models = []
    for base_path in base_paths:
        if not os.path.isdir(base_path): continue
        for name in os.listdir(base_path):
            if name.startswith('.'): continue
            
            path = os.path.join(base_path, name)
            
            if path in base_paths: continue
            if not os.path.isdir(path): continue
            
            if is_model(base_path, name):
                models.append(name)
                continue
            
            for name2 in os.listdir(path):
                if name.startswith('.'): continue
                path2 = os.path.join(path, name2)
                if os.path.isdir(path2) and is_model(path, name2):
                    models.append(f'{name}/{name2}')
                    continue
    
    sorted(models)
    return models

    
class StableDiffusionFriendlyPipeline():
    def __init__(self, model_name = "runwayml/stable-diffusion-v1-5", superres_pipeline = None):
        self.pipe = None

        # model
        self.model = model_name
        # vae
        self.vae = None

        # schedulers
        self.available_schedulers = {}

        # super-resolution
        self.superres_pipeline = superres_pipeline

        self.added_tokens = []
                
    def from_pretrained(self, verbose = True, force = False, model_name=None):
        if model_name is not None:
            if len(model_name.strip()) == 0:
                print("!!!!!检测出模型名称为空，我们将默认使用 MoososCap/NOVEL-MODEL")
                model_name = "MoososCap/NOVEL-MODEL"
            if model_name != self.model.strip():
                print(f"!!!!!正在切换新模型, {model_name}")
                self.model = model_name.strip()
                force=True

        model = self.model

        if (not force) and self.pipe is not None:
            return

        if verbose: print('!!!!!正在加载模型, 请耐心等待, 如果出现两行红字是正常的, 不要惊慌!!!!!')
        _ = paddle.zeros((1,)) # activate the paddle on CUDA

        with context_nologging():
            from .pipeline_stable_diffusion_all_in_one import StableDiffusionPipelineAllinOne
            self.pipe = StableDiffusionPipelineAllinOne.from_pretrained(model, safety_checker = None, requires_safety_checker=False)

        # update scheduler
        scheduler = self.pipe.scheduler
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            from ppdiffusers.configuration_utils import FrozenDict
            scheduler._internal_dict = FrozenDict(new_config)
            self.pipe.register_modules(scheduler=scheduler)
        self.available_schedulers = {}
        self.available_schedulers['default'] = scheduler
        # schedulers
        from ppdiffusers import KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler, PNDMScheduler, DDIMScheduler, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler
        self.available_schedulers.update({
            "DPMSolver": DPMSolverMultistepScheduler.from_pretrained(
                model_name,
                subfolder="scheduler",
                thresholding=False,
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                lower_order_final=True,
            ),
            "EulerDiscrete": EulerDiscreteScheduler.from_config(model_name, subfolder="scheduler"), 
            'EulerAncestralDiscrete': EulerAncestralDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler"), 
            'PNDM': PNDMScheduler.from_pretrained(model_name, subfolder="scheduler"), 
            'DDIM': DDIMScheduler.from_pretrained(model_name, subfolder="scheduler", clip_sample=False), 
            'LMSDiscrete' : LMSDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler"), 
            'HeunDiscrete': HeunDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler"), 
            'KDPM2AncestralDiscrete': KDPM2AncestralDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler"), 
            'KDPM2Discrete': KDPM2DiscreteScheduler.from_pretrained(model_name, subfolder="scheduler"), 
        })

        if verbose: print('成功加载完毕, 若默认设置无法生成, 请停止项目等待保存完毕选择GPU重新进入')

    def load_concepts(self, opt):
        added_tokens = []
        is_exist_concepts_library_dir = False
        original_dtype = None
        has_updated = False

        if opt.concepts_library_dir is not None:
            file_paths = None

            path = Path(opt.concepts_library_dir)
            if path.exists():
                #file_paths = path.glob("*.pdparams")
                file_paths = [p for p in path.glob("*.pdparams")]

                # conversion of .pt -> .pdparams embedding
                pt_files = path.glob("*.pt")
                for pt_file in pt_files:
                    try:
                        convert_pt_to_pdparams(pt_file, dim = 768, save = True)
                    except:
                        pass
            
            if opt.concepts_library_dir.endswith('.pdparams') and os.path.exists(opt.concepts_library_dir): 
                # load single file
                file_paths = [opt.concepts_library_dir]                

            if file_paths is not None and len(file_paths)>0:
                is_exist_concepts_library_dir = True
                
                # load the token safely in float32
                original_dtype = self.pipe.text_encoder.dtype
                self.pipe.text_encoder = self.pipe.text_encoder.to(dtype = 'float32')
                self.added_tokens = []
                for p in file_paths:
                    for token, embeds in paddle.load(str(p)).items():
                        added_tokens.append(token)
                        if embeds.dim() == 1:
                            embeds = embeds.reshape((1, -1))
                        tokens = get_multiple_tokens(token, embeds.shape[0], ret_list = True)
                        self.added_tokens.append((token, ' '.join(tokens)))

                        for token, embed in zip(tokens, embeds):
                            self.pipe.tokenizer.add_tokens(token)
                            self.pipe.text_encoder.resize_token_embeddings(len(self.pipe.tokenizer))
                            token_id = self.pipe.tokenizer.convert_tokens_to_ids(token)
                            with paddle.no_grad():
                                if paddle.max(paddle.abs(self.pipe.text_encoder.get_input_embeddings().weight[token_id] - embed)) > 1e-4:
                                    # only add / update new token if it has changed
                                    has_updated = True
                                    self.pipe.text_encoder.get_input_embeddings().weight[token_id] = embed
                            

        if is_exist_concepts_library_dir:
            if has_updated and len(added_tokens):
                str_added_tokens = ", ".join(added_tokens)
                print(f"[导入训练文件] 成功加载了这些新词: {str_added_tokens} ")
        else:
            print(f"[导入训练文件] {opt.concepts_library_dir} 文件夹下没有发现任何文件，跳过加载！")
        
        if self.added_tokens:
            #self_str_added_tokens = ", ".join((self.added_tokens))
            str_added_tokens = ", ".join(added_tokens)
            print(f"[支持的'风格'或'人物'单词]: {str_added_tokens} ")
        if original_dtype is not None:
            self.pipe.text_encoder = self.pipe.text_encoder.to(dtype = original_dtype)
    
    def run(self, opt, task = 'txt2img', on_image_generated = None):
        model_name = try_get_catched_model(opt.model_name)
        self.from_pretrained(model_name=model_name)
        self.load_concepts(opt)

        seed = None if opt.seed == -1 else opt.seed

        # switch scheduler
        self.pipe.scheduler = self.available_schedulers[opt.sampler]
        task_func = None

        # process prompts
        enable_parsing = False
        prompt = opt.prompt
        negative_prompt = opt.negative_prompt
        if '{}' in opt.enable_parsing:
            enable_parsing = True
            # convert {} to ()
            prompt = prompt.translate({40:123, 41:125, 123:40, 125:41})
            negative_prompt = negative_prompt.translate({40:123, 41:125, 123:40, 125:41})
        elif '()' in opt.enable_parsing:
            enable_parsing = True
        for token in self.added_tokens:
            prompt = prompt.replace(token[0], token[1])
            negative_prompt = negative_prompt.replace(token[0], token[1])
        
        
        init_image = None
        mask_image = None
        if task == 'txt2img':
            def task_func():
                return self.pipe.text2image(
                                    prompt, seed=seed, 
                                    width=opt.width, 
                                    height=opt.height, 
                                    guidance_scale=opt.guidance_scale, 
                                    num_inference_steps=opt.num_inference_steps, 
                                    negative_prompt=negative_prompt,
                                    max_embeddings_multiples=int(opt.max_embeddings_multiples),
                                    skip_parsing=(not enable_parsing)
                                ).images[0]
        elif task == 'img2img':
            init_image = ReadImage(opt.image_path, height=opt.height, width=opt.width)
            def task_func():
                return self.pipe.img2img(
                                    prompt, seed=seed, 
                                    image=init_image, 
                                    num_inference_steps=opt.num_inference_steps, 
                                    strength=opt.strength, 
                                    guidance_scale=opt.guidance_scale, 
                                    negative_prompt=negative_prompt,
                                    max_embeddings_multiples=int(opt.max_embeddings_multiples),
                                    skip_parsing=(not enable_parsing)
                                )[0][0]
        elif task == 'inpaint':
            init_image = ReadImage(opt.image_path, height=opt.height, width=opt.width)
            mask_image = ReadImage(opt.mask_path, height=opt.height, width=opt.width)
            def task_func():
                return self.pipe.inpaint(
                                    prompt, seed=seed, 
                                    image=init_image, 
                                    mask_image=mask_image, 
                                    num_inference_steps=opt.num_inference_steps, 
                                    strength=opt.strength, 
                                    guidance_scale=opt.guidance_scale, 
                                    negative_prompt=negative_prompt,
                                    max_embeddings_multiples=int(opt.max_embeddings_multiples),
                                    skip_parsing=(not enable_parsing)
                                )[0][0]
            
        if opt.fp16 == 'float16' and opt.sampler != "LMSDiscrete":
            context = paddle.amp.auto_cast(True, level = 'O2') # level = 'O2' # seems to have BUG if enable O2
        else:
            context = nullcontext()

        image_info = init_image.info if init_image is not None else None
        with context:
            for i in range(opt.num_return_images):
                empty_cache()
                image = task_func()
                image.argument['sampler'] = opt.sampler
                
                # super resolution
                if (self.superres_pipeline is not None):
                    argument = image.argument
                    argument['superres_model_name'] = opt.superres_model_name
                    
                    image = self.superres_pipeline.run(opt, image = image, end_to_end = False)
                    image.argument = argument

                if task == 'img2img':
                    image.argument['init_image'] = opt.image_path
                elif task == 'inpaint':
                    image.argument['init_image'] = opt.image_path
                    image.argument['mask_path'] = opt.mask_path
                        

                image.argument['model_name'] = opt.model_name
                
                if on_image_generated is not None:
                    on_image_generated(
                        image = image,
                        options = opt,
                        count = i,
                        total = opt.num_return_images,
                        image_info = image_info,
                    )
                    continue
                
                save_image_info(image, opt.output_dir,image_info)
                
                if i % 50 == 0:
                    clear_output()
                
                display(image)
                
                print('Seed = ', image.argument['seed'], 
                    '    (%d / %d ... %.2f%%)'%(i + 1, opt.num_return_images, (i + 1.) / opt.num_return_images * 100))

class SuperResolutionPipeline():
    def __init__(self):
        self.model = None
        self.model_name = ''
    
    def run(self, opt, 
                image = None, 
                task = 'superres', 
                end_to_end = True,
                force_empty_cache = True,
                on_image_generated = None,
            ):
        """
        end_to_end: return PIL image if False, display in the notebook and autosave otherwise
        empty_cache: force clear the GPU cache by deleting the model
        """
        if opt.superres_model_name is None or opt.superres_model_name in ('','无'):
            return image

        import numpy as np
        if image is None:
            image = ReadImage(opt.image_path, height=None, width=None) # avoid resizing
        image_info = image.info
        
        image = np.array(image)
        image = image[:,:,[2,1,0]]  # RGB -> BGR

        empty_cache()
        if self.model_name != opt.superres_model_name:
            if self.model is not None:
                del self.model 

            with context_nologging():
                # [ WARNING] - The _initialize method in HubModule will soon be deprecated, you can use the __init__() to handle the initialization of the object
                import paddlehub as hub
                # print('正在加载超分模型! 如果出现两三行红字是正常的, 不要担心哦!')
                self.model = hub.Module(name = opt.superres_model_name)
            
        self.model_name = opt.superres_model_name

        # time.sleep(.1) # wait until the warning prints
        # print('正在超分......请耐心等待')
    
        try:
            image = self.model.reconstruct([image], use_gpu = (paddle.device.get_device() != 'cpu'))[0]['data']
        except:
            print('图片尺寸过大, 超分时超过显存限制')
            self.empty_cache(force_empty_cache)
            paddle.disable_static()
            return

        image = image[:,:,[2,1,0]] # BGR -> RGB
        image = Image.fromarray(image)
        
        self.empty_cache(force_empty_cache)
        paddle.disable_static()
        
        if on_image_generated is not None:
            on_image_generated(
                image = image,
                options = opt,
                count = 0,
                total = 1,
                image_info = image_info,
            )
            return
        if end_to_end:
            cur_time = time.time()
            os.makedirs(opt.output_dir, exist_ok = True)
            image.save(os.path.join(opt.output_dir,f'Highres_{cur_time}.png'), quality=100)
            clear_output()
            display(image)
            return
        return image
    
    def empty_cache(self, force = True):
        # NOTE: it seems that ordinary method cannot clear the cache
        # so we have to delete the model (?)
        if not force:
            return
        del self.model
        self.model = None
        self.model_name = ''
