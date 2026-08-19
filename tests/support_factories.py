from distill_codec.models.mock import MockStudentEncoder


def create_encoder(channels=16):
    return MockStudentEncoder(latent_channels=channels)

