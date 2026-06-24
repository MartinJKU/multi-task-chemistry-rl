from __future__ import annotations

import re


_BRACKET_ELEMENT = re.compile(r"^\[(?:\d+)?([A-Z][a-z]?|se|as|[bcnops]|\*)")
_AROMATIC_TWO_CHAR = {"as", "se"}
_ORGANIC_TWO_CHAR = {"Br", "Cl"}
_ORGANIC_ONE_CHAR = set("BCNOPSFIbcnops*")


def smiles_atom_tokens(smiles: str) -> list[str]:
    """Return atom tokens in the 0-based order used by SMILES.

    Bonds, parentheses, stereochemistry markers, charges, and ring-closure
    labels do not create atom indices. Bracket expressions create exactly one
    atom token, including isotope-labelled hydrogens such as ``[2H]``.
    """
    atoms: list[str] = []
    i = 0
    while i < len(smiles):
        char = smiles[i]
        if char == "[":
            end = smiles.find("]", i + 1)
            if end < 0:
                break
            bracket = smiles[i : end + 1]
            match = _BRACKET_ELEMENT.match(bracket)
            if match:
                atoms.append(match.group(1))
            i = end + 1
            continue

        pair = smiles[i : i + 2]
        if pair in _ORGANIC_TWO_CHAR or pair in _AROMATIC_TWO_CHAR:
            atoms.append(pair)
            i += 2
            continue
        if char in _ORGANIC_ONE_CHAR:
            atoms.append(char)
        i += 1
    return atoms


def format_atom_map(smiles: str) -> str:
    """Format the complete SMILES atom map as ``0:C; 1:O; ...``."""
    return "; ".join(
        f"{index}:{token}" for index, token in enumerate(smiles_atom_tokens(smiles))
    )
