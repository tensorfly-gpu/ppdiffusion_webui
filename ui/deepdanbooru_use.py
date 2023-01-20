from PIL import Image
import numpy as np
import paddle
import tqdm
from .deep_danbooru_model import DeepDanbooruModel
import argparse

#model = deepdanbooru_paddle.deep_danbooru_model.DeepDanbooruModel()
model = DeepDanbooruModel()
model.set_state_dict(paddle.load('/home/aistudio/ui/deepdanbooru.pdparams'))
model.eval()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_path", default=None, type=str, help="Path to the picture to use."
    )
    args = parser.parse_known_args()[0]
    return args

def deepdanbooru_to_get_tags(args):
    pic = Image.open(args.image_path).convert("RGB").resize((512, 512))
    a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

    with paddle.no_grad(), paddle.amp.auto_cast():
        x = paddle.to_tensor(a)

     # first run
        y = model(x)[0].detach().cpu().numpy()

    # measure performance
        for n in tqdm.tqdm(range(1)):
            model(x)


    for i, p in enumerate(y):
        if p >= 0.5:
            print(model.tags[i], p)
        