import json


def load_cls_list():
    with open("./imagenet_class_index.json") as json_file:
        class_idx = json.load(json_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    return idx2label


if __name__ == "__main__":
    a = load_cls_list()
    print(a)
