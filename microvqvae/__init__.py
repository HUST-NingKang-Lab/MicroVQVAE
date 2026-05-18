__all__ = ["tokenize_protein_fasta"]


def tokenize_protein_fasta(*args, **kwargs):
    from .pipeline import tokenize_protein_fasta as _tokenize_protein_fasta

    return _tokenize_protein_fasta(*args, **kwargs)
