import unittest

from microvqvae.model import MicroVQVAEModel


class MicroVQVAEModelApiTests(unittest.TestCase):
    def test_public_model_preserves_full_dvae_api(self):
        model = MicroVQVAEModel(embed_dim=1280, d_model=320, nhead=4, num_enc_layers=6, num_dec_layers=6, codebook_size=8192, code_dim=64)
        for attr in ["forward", "masked_mse", "masked_cosine_loss", "decode_tokens", "lookup_codebook", "codebook_diversity_loss"]:
            self.assertTrue(hasattr(model, attr), f"Missing expected model API: {attr}")


if __name__ == "__main__":
    unittest.main()
