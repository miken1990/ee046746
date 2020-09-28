import torch
from torchvision import transforms
from PIL import Image
import os


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


def get_fc_vector(image, model, layer_to_output):
    # conv layers
    modulelist_features = list(model.features.modules())
    for count, layer in enumerate(modulelist_features[1:]):
        image = layer(image)

    # avg layer
    modulelist_avgpool = list(model.avgpool.modules())
    fc_vec = modulelist_avgpool[0](image)
    # print(f'fc vec shape: {fc_vec.shape}')
    fc_vec = fc_vec.view(fc_vec.size(0), -1)
    # print(f'shape after view: {fc_vec.shape}')

    # fc layers
    modulelist_clf = list(model.classifier.modules())
    for c, l in enumerate(modulelist_clf[1:]):
        # print(f'count: {c}, layer: {l}')
        fc_vec = l(fc_vec)
        if c == layer_to_output:
            return fc_vec


def save_feature_vec(dirs, model, num_fc_layer, dir_to_label, save_name):
    final_X = []
    final_Y = []
    for directory in dirs:
        print(f'calculating feature vectors for {directory}')
        for i, im_name in enumerate(sorted(os.listdir(directory))):
            full_im_path = os.path.join(directory, im_name)
            im = load_image(full_im_path)
            norm_im = prepare_image(im, device='cpu')
            feature_vec = get_fc_vector(norm_im, model, num_fc_layer).detach().numpy().squeeze(0)
            final_X.append(feature_vec)
            final_Y.append(dir_to_label[directory])

    state = {
        'X': final_X,
        'y': final_Y
    }
    save_name = os.path.join('.', save_name)
    print(f'saving X and y for svm')
    torch.save(state, save_name)


def load_feature_vec(directory):

    for i, im_name in enumerate(os.listdir(directory)):
        full_im_path = os.path.join(directory, im_name)
        print(f'loading feature vector for {im_name}')
        feature_vec = torch.load(full_im_path)['f_vec'].numpy()
        print(feature_vec.shape)
