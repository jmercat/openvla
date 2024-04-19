import tqdm
import wandb

from prismatic.vla.datasets.datasets import EpisodicRLDSDataset
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES

MIX_NAME = "oxe_magic_soup_plus"
DATA_ROOT = "gs://rail-orca-central2/resize_256_256"
N_VIDEOS = 3  # number of videos per dataset

assert MIX_NAME in OXE_NAMED_MIXTURES
wandb.init(entity="stanford-voltron", project="vla_data_vis", name=f"VIS_{MIX_NAME}")
for dataset_info in tqdm.tqdm(OXE_NAMED_MIXTURES[MIX_NAME]):
    dataset_name, weight = dataset_info
    dataset = EpisodicRLDSDataset(
        DATA_ROOT,
        dataset_name,
        batch_transform=None,
        resize_resolution=(256, 256),
        shuffle_buffer_size=10,
    )
    for i, episode in enumerate(dataset.dataset):
        imgs = episode["observation"]["image_primary"].numpy()
        wandb.log({dataset_name: wandb.Video(imgs, fps=5)})
        if i == N_VIDEOS - 1:
            break
