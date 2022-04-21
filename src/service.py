import bentoml
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from bentoml.io import Image, JSON
import torch
import bentoml
from bentoml.io import NumpyNdarray, JSON


iris_cls_runner = bentoml.sklearn.load_runner("iris_cls:latest")
imagenet_runner = bentoml.pytorch.load_runner("imagenet_cls:latest")

cls_service = bentoml.Service("iris_imagenet_service", runners=[iris_cls_runner, imagenet_runner])

image_pre_pro = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@cls_service.api(input=NumpyNdarray(), output=JSON())
def classifier_iris(input_data):
    result = iris_cls_runner.run(input_data)
    return result


@cls_service.api(input=Image(), output=JSON())
def classifier_imagenet(input_img):
    input_tensor = image_pre_pro(input_img)
    logit = imagenet_runner.run(input_tensor)
    porb = torch.softmax(logit, dim=0)
    sort_porb = porb.sort(descending=True)
    top_k_dict = dict()
    for val, idx in zip(sort_porb[0][:5], sort_porb[1][:5]):
        top_k_dict[idx.item()] = val.item()

    return top_k_dict
