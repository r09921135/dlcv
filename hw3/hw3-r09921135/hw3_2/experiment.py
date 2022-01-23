import torch

from matplotlib import pyplot as plt
import cv2
import math
from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco
from configuration import Config
import os

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--img_path', type=str, help='path to image', default='../hw3_data/p2_data/images')
parser.add_argument('--out_path', type=str, help='path to output image', default='./output')
parser.add_argument('--v', type=str, help='version', default='v3')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
args = parser.parse_args()
image_path = args.img_path
output_path = args.out_path
version = args.v
checkpoint_path = args.checkpoint

config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
      raise NotImplementedError('No model to chose from!')
    else:
      if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
      print("Found checkpoint! Loading!")
      model,_ = caption.build_model(config)
      print("Loading Checkpoint...")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      model.load_state_dict(checkpoint['model'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)


# register forward hook
def prepare_hook(model, feat_list, attn_map_list):
    hooks = [
        model.backbone.register_forward_hook(
            lambda self, input, output: feat_list.append(output)
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: attn_map_list.append(output)
        ),
    ]
    return hooks


@torch.no_grad()
def evaluate(image):
    model.eval()
    feat_list = []
    attn_map_list = []
    for i in range(config.max_position_embeddings - 1):
        hooks = prepare_hook(model, feat_list, attn_map_list)

        predictions = model(image, caption, cap_mask)  # (B, 128, 30522)

        for hook in hooks:
            hook.remove()

        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption, feat_list, attn_map_list

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption, feat_list, attn_map_list


# visulize cross-attention
def viz_attn(image_fn, output, feat_list, attn_map_list):
    img = Image.open(os.path.join(image_path, image_fn))
    img_h, img_w = img.size[0], img.size[1]

    feat = feat_list[0][0][-1].decompose()[0]
    h, w = feat.shape[2], feat.shape[3]
    n_word = len(attn_map_list)
    scale = 0.25

    fig_size_x = math.ceil(n_word/2) * w * scale
    fig_size_y = 2 * h * scale
    fig = plt.figure(figsize=(fig_size_x, fig_size_y))
    ax = fig.add_subplot(2, math.ceil(n_word/2), 1)
    # show original image
    ax.imshow(img)
    ax.set_title('<start>', fontsize=20)

    for i in range(n_word-1):  # attn_map_list include EOS, need to subtract 1
        attn_map = attn_map_list[i][1]  # (B, 128, h*w)
        attn = attn_map[0, i, :].reshape((h, w)).numpy()

        ax = fig.add_subplot(2, math.ceil(n_word/2), i+2)
        ax.imshow(img)
        attn = cv2.resize(attn, (img_h, img_w))
        # show cross-attention map
        ax.imshow(attn, alpha=0.5, cmap='jet')

        if i == n_word-2:
            ax.set_title("<end>", fontsize=20)
        else:
            ax.set_title(tokenizer.decode(output[0, i+1].item()), fontsize=20)  # output include BOS, neet to plus 1

    save_path = os.path.join(output_path, (image_fn.split('.'))[0] + '.png')
    plt.savefig(save_path)



print('Start evaluating!')
filenames = [file for file in os.listdir(image_path)]
for image_fn in filenames:
    image = Image.open(os.path.join(image_path, image_fn))
    image = coco.val_transform(image)
    image = image.unsqueeze(0)

    output, feat_list, attn_map_list = evaluate(image)

    viz_attn(image_fn, output, feat_list, attn_map_list)

print('Done!')
