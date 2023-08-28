from enum import Enum


class TrainMode(str, Enum):
    # manipulate mode = training the classifier
    manipulate = "manipulate"
    # default training mode!
    diffusion = "diffusion"
    # default latent training mode!
    # fitting the a DDPM to a given latent
    latent_diffusion = "latentdiffusion"
    simsiam = "simsiam"
    simclr = "simclr"
    # supervised base line
    supervised = "supervised"

    def is_manipulate(self):
        return self in [TrainMode.manipulate, TrainMode.supervised]

    def is_diffusion(self):
        return self in [TrainMode.diffusion, TrainMode.latent_diffusion]

    def is_autoenc(self):
        # the network possibly does autoencoding
        return self in [TrainMode.diffusion]

    def is_latent_diffusion(self):
        return self in [TrainMode.latent_diffusion]

    def use_latent_net(self):
        return self.is_latent_diffusion()

    def require_dataset_infer(self):
        """
        whether training in this mode requires the latent variables to be available?
        """
        # this will precalculate all the latents before hand
        # and the dataset will be all the predicted latents
        return self in [TrainMode.latent_diffusion, TrainMode.manipulate]
