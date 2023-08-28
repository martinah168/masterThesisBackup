import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image

plt.switch_backend('agg')


def get_confmat_image(confmat: torch.Tensor, mode: str) -> torch.Tensor:
    ''' Visualize confusion matrix as matplotlib figure.'''
    confmat = confmat.detach().cpu().numpy()

    df_cm = pd.DataFrame(confmat,
                         index=np.arange(confmat.shape[0]),
                         columns=np.arange(confmat.shape[0]))

    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, fmt='', annot=True)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'{mode.capitalize()} Confusion matrix')
    plt.tight_layout()

    # for debugging save the figure
    # plt.savefig('test_img/confmat.png')

    pil_image = fig2img(plt.gcf())
    plt.close()

    np_image = np.array(pil_image)
    torch_image = torch.from_numpy(np_image).float().div(255).permute(2, 0, 1)
    return torch_image


def fig2img(fig: plt.Figure) -> Image.Image:
    '''Convert a Matplotlib figure to a PIL Image and return it'''
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
