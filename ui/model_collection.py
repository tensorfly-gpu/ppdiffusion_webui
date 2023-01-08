import os 
from traitlets import HasTraits,Tuple


default_model_list = [
    "MoososCap/NOVEL-MODEL", 
    "ruisi/anything",
    "Linaqruf/anything-v3.0",
    "Baitian/momocha",
    "Baitian/momoco",
    "hequanshaguo/monoko-e",
    "hakurei/waifu-diffusion", 
    "hakurei/waifu-diffusion-v1-3", 
    "CompVis/stable-diffusion-v1-4", 
    "runwayml/stable-diffusion-v1-5", 
    "stabilityai/stable-diffusion-2",
    "stabilityai/stable-diffusion-2-base",
    "naclbit/trinart_stable_diffusion_v2_60k", 
    "naclbit/trinart_stable_diffusion_v2_95k", 
    "naclbit/trinart_stable_diffusion_v2_115k", 
    "ringhyacinth/nail-set-diffuser",
    "Deltaadams/Hentai-Diffusion",
    "BAAI/AltDiffusion",
    "BAAI/AltDiffusion-m9",
    "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1",
    "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1",
    "huawei-noah/Wukong-Huahua"
]
local_model_list = []

cache_paths = [
    './',
]

def is_avaliable_model(path):
    return (os.path.isfile(
            os.path.join(path,'model_index.json')
        )) and (os.path.isfile(
            os.path.join(path,'vae', 'model_state.pdparams')
        )) and (os.path.isfile(
            os.path.join(path,'unet', 'model_state.pdparams')
        ))

    
def collect_local_model_names(base_paths = None):
    """从指定位置检索可用的模型名称，以用于UI选择模型"""
    base_paths = (
            './', 
            os.path.join(os.environ['PPNLP_HOME'], 'models')
        ) if base_paths is None \
        else (base_paths,) if isinstance(base_paths, str) \
        else base_paths
    
    models = []
    for base_path in base_paths:
        if not os.path.isdir(base_path): continue
        for name in os.listdir(base_path):
            if name.startswith('.'): continue
            
            path = os.path.join(base_path, name)
            
            if path in base_paths: continue
            if not os.path.isdir(path): continue
            
            if is_avaliable_model(path):
                models.append(name)
                continue
            
            for name2 in os.listdir(path):
                if name.startswith('.'): continue
                path2 = os.path.join(path, name2)
                if os.path.isdir(path2) and is_avaliable_model(path2):
                    models.append(f'{name}/{name2}')
                    continue
    
    sorted(models)
    return models
    


class ObservableModelCollection(HasTraits):
    models = Tuple()
    
    def __init__(self):
        super().__init__()
        self._cached_paths = list(cache_paths)
        self.models = tuple()
        
        self._has_loaded = False
    
    def load_locals(self):
        if not self._has_loaded:
            self.refresh_locals()
    
    def refresh_locals(self):
        if not self._has_loaded:
            #懒加载，因为environ['PPNLP_HOME']尚未初始化
            self._cached_paths.append(
                os.path.join(os.environ['PPNLP_HOME'], 'models')
            )
            
        _local_models = collect_local_model_names(self._cached_paths)
        _last_models = []
        for name in local_model_list:
            if (name not in _local_models) \
                and (name not in default_model_list) \
                and (self.is_avaliable_name(name)):
                _last_models.append(name)
        local_model_list.clear()
        local_model_list.extend(_last_models)
        local_model_list.extend(_local_models)
        self._has_loaded = True
        self._update_list()
    
    def is_avaliable_name(self, model_name):
        if model_name in default_model_list: return True
        if is_avaliable_model(model_name): return True
        
        for dir in self._cached_paths:
            path = os.path.join(dir, model_name)
            if is_avaliable_model(path):
                return True
        return False
        
    def record_model_name(self, model_name):
        if (model_name not in self.models):
            local_model_list.append(model_name)
            self._update_list()
    
    def _update_list(self):
        sorted(local_model_list)
        new_models = tuple(default_model_list) + tuple(local_model_list)
        if new_models != self.models:
            self.models = new_models

model_collection = ObservableModelCollection()