import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import json
import argparse

# Initiate variables with default values
checkpoint = 'checkpoint.pth'
filepath = 'cat_to_name.json'
image_path = 'flowers/test/100/image_07896.jpg'
topk = 5

# Set up parameters for entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', action='store', type=str, help='Name of trained model to be loaded and used for predictions.')
parser.add_argument('-i', '--image_path', action='store', type=str, help='Location of image to predict e.g. flowers/test/class/image')
parser.add_argument('-k', '--topk', action='store', type=int, help='Select the number of classes you wish to see in descending order.')
parser.add_argument('-j', '--json', action='store', type=str, help='Define the name of the json file holding class names.')
parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()

# Select parameters entered in the command line
if args.checkpoint:
    checkpoint = args.checkpoint
if args.image_path:
    image_path = args.image_path
if args.topk:
    topk = args.topk
if args.json:
    filepath = args.json
if args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

with open(filepath, 'r') as f:
    cat_to_name = json.load(f)


def load_model(checkpoint_path):
    '''
    Load model from a checkpoint
    '''
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = 9216
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = 1024
    else:
        raise ValueError('Sorry, base architecture not recognized')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(in_features, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(p=0.15),
        nn.Linear(checkpoint['hidden_units'], 512),
        nn.ReLU(),
        nn.Dropout(p=0.15),
        nn.Linear(512, 102),
        nn.LogSoftmax(dim=1)
    )

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model.to(device)


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    size = 256, 256
    crop_size = 224

    im = Image.open(image_path)
    im.thumbnail(size)

    left = (size[0] - crop_size) / 2
    top = (size[1] - crop_size) / 2
    right = (left + crop_size)
    bottom = (top + crop_size)

    im = im.crop((left, top, right, bottom))

    np_image = transforms.ToTensor()(im)
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    pytorch_np_image = np_image.numpy()

    return pytorch_np_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Use process_image function to create numpy image tensor
    pytorch_np_image = process_image(image_path)

    # Changing from numpy to pytorch tensor
    pytorch_tensor = torch.tensor(pytorch_np_image)
    pytorch_tensor = pytorch_tensor.float()

    # Removing RunTimeError for missing batch size - add batch size of 1
    pytorch_tensor = pytorch_tensor.unsqueeze(0)

    # Run model in evaluation mode to make predictions
    model.eval()
    with torch.no_grad():
        LogSoftmax_predictions = model.forward(pytorch_tensor.to(device))
    predictions = torch.exp(LogSoftmax_predictions)

    # Identify top predictions and top labels
    top_preds, top_labs = predictions.topk(topk)

    top_preds = top_preds.cpu().numpy().squeeze()
    top_labs = top_labs.cpu().numpy().squeeze()

    labels = pd.DataFrame({'class': pd.Series(model.class_to_idx), 'flower_name': pd.Series(cat_to_name)})
    labels = labels.set_index('class')
    labels = labels.iloc[top_labs]
    labels['predictions'] = top_preds

    return labels


model = load_model(checkpoint)

print('-' * 40)

print(model)
print('The model being used for the prediction is above.')
input("When you are ready - press Enter to continue to the prediction.")
labels = predict(image_path, model, topk)
print('-' * 40)
print(labels)
print('-' * 40)