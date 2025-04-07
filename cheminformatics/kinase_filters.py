""" I compiled a set of kinase inhibitor 'rules' to filter out most junk in the screening libraries. These rules are
driven by generally accepted medicinal chemistry knowledge that is general to all kinase inhibitors. I validated these
rules with PKIDB database [1]: 98.6% of all kinase inhibitors that ever went into clinical trials pass these filters.
Out of 241,623 molecules in the cleaned Specs library we use for screening, 189,025 pass all four filters, with the
majority (45,283) failing because of a lacking ATP-mimetic core.


1. A molecule must have an ATP mimetic core.
    Must contain at least one heteroatom in a ring, or a fused carbocyclic system. This captures the ATP-mimetic core
    requirement while allowing some room for e.g., non-aromatic fused systems like fluorene or indane.

2. A molecule must have a polar anchor.
    Ensures that there’s some polar group to avoid greaseball molecules, help with solubility, and possibly interact
    with solvent-front residues.

3. A molecule must have some hydrophobic bulk.
    Ensures that there’s enough hydrophobic mass or planarity to fill the kinase ATP pocket (e.g., gatekeeper region).

4. A molecule must have a directional H-bond donor/acceptor.
    Ensures potential for hinge interaction via a directional H-bond donor/acceptor, ideally in a semi-rigid context.

5. A molecule must have 2 or more rings.
    Almost all kinase inhibitors, with very few exceptions, ontain a core + substituent ring (e.g., hinge binder +
    solvent front tail). This rule removes fragment-like or ultra-simple molecules, matches kinase inhibitor-like
    properties, and brings us closer to a lead-like space.


[1] Carles, F., Bourg, S., Meyer, C., and Bonnet, P. (2018). PKIDB: A Curated, Annotated and Updated Database of
    Kinase Inhibitors in Clinical Trials. Molecules 23, 908. DOI:10.3390/molecules23040908

Derek van Tilborg
Eindhoven University of Technology
April 2025
"""

from rdkit import Chem
from rdkit.Chem import Descriptors


def _has_fused_ring(mol):
    # Any atom that is in a ring system, and in more than one ring
    smarts = "[R&!R1]"

    return mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))


def _has_heterocycle(mol):
    # Any atom thats in a ring and is not a carbon
    smarts = "[r;!#6]"

    return mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))


def has_two_rings(mol):
    # Pretty much every kinase inhibitor has two or more rings
    nrings = Descriptors.RingCount(mol)

    return True if nrings > 1 else False


def has_ATP_mimetic_core(mol):
    # Ensures flat, structured ATP mimic

    return _has_fused_ring(mol) or _has_heterocycle(mol)


def has_polarity_anchor(mol):
    # Ensures solubility or solvent interaction. Checks if there’s some polar group to avoid greaseball molecules,
    # help with solubility, possibly interact with solvent-front residues

    polar_anchor_smarts = [
        "[NX3;H2,H1;!$(NC=O)]",  # Primary/secondary amine (not in amide)
        "[OX2H]",  # Alcohol
        "[SX4](=O)(=O)[#7]",  # Sulfonamide
        "C(=O)[NX3]",  # Amide
        "C(=O)O",  # Carboxylic acid

        # We relax these rules with ring nitrogens, as these still interact with water and solvent-exposed pockets
        "[n]",  # Aromatic nitrogen in ring
        "[N;R]"  # Any ring nitrogen
    ]

    return any(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in polar_anchor_smarts)


def has_hydrophobic_bulk(mol):
    # Ensure there’s enough hydrophobic mass or planarity to fill the kinase ATP pocket (e.g., gatekeeper region)

    hydrophobic_smarts = [
        "c1ccccc1",  # Benzene
        "C1CCCCC1",  # Cyclohexyl
        "[#6]-[#6]-[#6]",  # Short chain
        "C=C",  # Alkene
        "n1cccn1",  # Pyrrole-like 5-membered ring
        "c1ncnc2ncccc12",  # Triazolopyrimidine (general purine mimic)
        "[R;!#6]",  # Ring atom that's not carbon
    ]

    return any(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in hydrophobic_smarts)


def has_directional_motif(mol):
    # Ensure potential for hinge interaction via a directional H-bond donor/acceptor,
    # ideally in a semi-rigid context.

    directional_smarts = [
        "[CX3]=[OX1]",  # Carbonyl (e.g., ketone, amide)
        "[NX3;H2,H1]",  # Primary or secondary amine
        "[OX2H]",  # Alcohol
        "[n]",  # Aromatic nitrogen (heterocycle)
        "[NX2]=[CX3]"  # Imine (directional acceptor)
    ]

    return any(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in directional_smarts)


def find_kinase_violations(smiles: str):

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "SMILES issues"

    found_reasons = []

    if not has_ATP_mimetic_core(mol):
        found_reasons.append('No ATP mimetic')

    if not has_two_rings(mol):
        found_reasons.append('Too few rings')

    if not has_polarity_anchor(mol):
        found_reasons.append('No polarity anchor')

    if not has_directional_motif(mol):
        found_reasons.append('No directional motif')

    if not has_hydrophobic_bulk(mol):
        found_reasons.append('No hydrophobic bulk')

    # Passes all filters
    return ", ".join(found_reasons) if found_reasons else 'Passed'
