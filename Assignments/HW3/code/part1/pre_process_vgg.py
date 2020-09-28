import torch
from torchvision import transforms
from PIL import Image


def prepare_image(image, device=torch.device("cpu")):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    image = preprocess(image).unsqueeze(0).to(device)
    return image


def predict(image, model, labels=None):
    _, index = model(image).data[0].max(0)
    if labels is not None:
        return labels[index.item()]
    else:
        return str(index.item())


def deprocess(image, device=torch.device("cpu")):
    return image * torch.tensor([0.229, 0.224, 0.225]).to(device) + torch.tensor([0.485, 0.456, 0.406]).to(device)


def load_image(path):
    image = Image.open(path)
    return image


def transform_from_cv2(im, size=224):
    tr_pipeline_list = transforms.Compose([transforms.Resize(size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    im_pil = Image.fromarray(im)
    im = tr_pipeline_list(im_pil)
    im = im.unsqueeze(0)
    im = torch.autograd.Variable(im)
    im = im.float()
    return im