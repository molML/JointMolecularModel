

import re
import torch
import torch.nn.functional as F
from constants import VOCAB


def smiles_tokenizer(smiles: str, extra_patterns: list[str] = None) -> list[str]:
    """ Tokenize a SMILES. By default, we use the base SMILES grammar tokens and the reactive nonmetals H, C, N, O, F,
    P, S, Cl, Se, Br, I:

    '(\\[|\\]|Cl|Se|se|Br|H|C|c|N|n|O|o|F|P|p|S|s|I|\\(|\\)|\\.|=|#|-|\\+|\\\\|\\/|:|~|@|\\?|>|\\*|\\$|\\%\\d{2}|\\d)'

    :param smiles: SMILES string
    :param extra_patterns: extra tokens to consider (default = None)
        e.g. metalloids: ['Si', 'As', 'Te', 'te', 'B', 'b']  (in ChEMBL33: B+b=0.23%, Si=0.13%, As=0.01%, Te+te=0.01%).
        Mind you that the order matters. If you place 'C' before 'Cl', all Cl tokens will actually be tokenized as C,
        meaning that subsets should always come after superset strings, aka, place two letter elements first in the list
    :return: list of tokens extracted from the smiles string in their original order
    """
    base_smiles_patterns = "(\[|\]|insert_here|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\d)"
    # reactive_nonmetals = ['Cl', 'Si', 'si', 'Se', 'se', 'Br', 'B', 'H', 'C', 'c', 'N', 'n', 'O', 'o', 'F', 'P', 'p', 'S', 's', 'I']
    reactive_nonmetals = ['Cl', 'Br', 'H', 'C', 'c', 'N', 'n', 'O', 'o', 'F', 'P', 'p', 'S', 's', 'I']

    # Add all allowed elements to the base SMILES tokens
    extra_patterns = reactive_nonmetals if extra_patterns is None else extra_patterns + reactive_nonmetals
    pattern = base_smiles_patterns.replace('insert_here', "|".join(extra_patterns))

    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smiles)]

    return tokens


def smiles_to_encoding(smi: str) -> torch.Tensor:
    """Converts a SMILES string into a list of token indices using a predefined vocabulary """

    encoding = [VOCAB['start_idx']] + [VOCAB['token_indices'][i] for i in smiles_tokenizer(smi)] + [VOCAB['end_idx']]
    encoding.extend([VOCAB['pad_idx']] * (VOCAB['max_len'] - len(encoding)))

    return torch.tensor(encoding)


def encode_smiles(smiles: list[str]):
    return torch.stack([smiles_to_encoding(smi) for smi in smiles])


def one_hot_encode(encodings):
    return F.one_hot(encodings, VOCAB['vocab_size'])


def probs_to_encoding(x: torch.Tensor) -> torch.Tensor:
    """ Gets the most probable token for every entry in a sequence

    :param x: Tensor in shape (batch x seq_length x vocab)
    :return: x: Tensor in shape (batch x seq_length)
    """

    assert x.dim() == 3
    return x.argmax(dim=2)


def encoding_to_smiles(encoding: torch.Tensor) -> list[str]:
    """ Convert a tensor of token indices into a list of character strings

    :param encoding: Tensor in shape (batch x seq_length x vocab) containing ints
    :return: list of SMILES strings (with utility tokens)
    """

    assert encoding.dim() == 2, f"Encodings should be shape (batch_size x seq_length), not {encoding.shape}"
    return [''.join([VOCAB['indices_token'][t_i.item()] for t_i in enc]) for enc in encoding]


def clean_smiles(smiles: list[str]) -> list[str]:
    """ Strips the start and end character from a list of SMILES strings: >xxxxxx;____ -> xxxxxx

    :param smiles: list of 'uncleaned' SMILES
    :return: list of SMILES strings
    """

    return [smi.split(VOCAB['start_char'])[-1].split(VOCAB['end_char'])[0] for smi in smiles]
