import json
import ast
from PIL import Image


def load_cls_list():
    with open("./imagenet_class_index.json") as json_file:
        class_idx = json.load(json_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    return idx2label


def text2list(data):
    return [int(x) for x in ast.literal_eval(data)]


def load_image(image_file):
    img = Image.open(image_file)
    return img


if __name__ == "__main__":
    a = load_cls_list()
    print(a)
