import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import microvqvae.pipeline as pipeline


class FakeEmbedder:
    @classmethod
    def from_pretrained(cls, model_name_or_path, device='auto', dtype='auto'):
        return cls()

    def embed_records(self, records, batch_size=32, max_length=1024):
        values = torch.arange(len(records) * 1280, dtype=torch.float32)
        return values.reshape(len(records), 1280)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode_tokens(self, x, mask):
        return torch.arange(x.size(1), device=x.device).unsqueeze(0)

    def lookup_codebook(self, indices):
        return torch.stack([indices.float(), indices.float() + 10.0], dim=-1)


def fake_load_checkpoint(checkpoint, device='auto'):
    return FakeModel(), {'embed_dim': 1280, 'code_dim': 2}


class PipelineSmokeTests(unittest.TestCase):
    def test_tokenize_protein_fasta_runs_with_synthetic_embeddings(self):
        original_embedder = pipeline.PairESMEmbedder
        original_loader = pipeline.load_microvqvae_checkpoint
        pipeline.PairESMEmbedder = FakeEmbedder
        pipeline.load_microvqvae_checkpoint = fake_load_checkpoint
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                fasta_path = tmp_path / 'proteins.faa'
                output_dir = tmp_path / 'out'
                fasta_path.write_text(
                    '>protein_a first protein\nMAAA\n'
                    '>protein_b second protein\nMBBB\n'
                    '>protein_c third protein\nMCCC\n'
                )

                metadata = pipeline.tokenize_protein_fasta(
                    input_fasta=str(fasta_path),
                    checkpoint='fake.ckpt',
                    output_dir=str(output_dir),
                    pair_esm_model='fake-pair-esm',
                    batch_size=2,
                    window_size=2,
                    device='cpu',
                )

                tokens = np.load(output_dir / 'tokens.npy')
                codebook = np.load(output_dir / 'codebook_embeddings.npy')
                saved_metadata = json.loads((output_dir / 'metadata.json').read_text())
                tsv_lines = (output_dir / 'tokens.tsv').read_text().strip().splitlines()

                np.testing.assert_array_equal(tokens, np.array([0, 1, 0]))
                np.testing.assert_array_equal(
                    codebook,
                    np.array([[0.0, 10.0], [1.0, 11.0], [0.0, 10.0]], dtype=np.float32),
                )
                self.assertEqual(metadata['num_proteins'], 3)
                self.assertEqual(saved_metadata['code_dim'], 2)
                self.assertEqual(tsv_lines[0], 'index\tsequence_id\tdescription\tlength_aa\ttoken_id')
                self.assertEqual(tsv_lines[1].split('\t')[1], 'protein_a')
        finally:
            pipeline.PairESMEmbedder = original_embedder
            pipeline.load_microvqvae_checkpoint = original_loader


if __name__ == '__main__':
    unittest.main()
