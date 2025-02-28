"""

Python implementation of the CATS molecular descriptor, taken directly from Alex Müller
https://github.com/alexarnimueller/cats-descriptor

From the paper:

Reutlinger, M., Koch, C. P., Reker, D., Todoroff, N., Schneider, P., Rodrigues, T., & Schneider, G. (2013).
Chemically Advanced Template Search (CATS) for Scaffold-Hopping and Prospective Target Prediction for 'Orphan'
Molecules. Molecular informatics, 32(2), 133–138. https://doi.org/10.1002/minf.201200141

The multiprocessing functions were giving me errors, so I reimplemented the wrapper function in
``cheminformatics.descriptors.mols_to_cats``

Derek van Tilborg
Eindhoven University of Technology
July 2024
"""


import numpy as np
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.Pharm2D import SigFactory, Generate


def get_cats_factory(features: str = "cats", names: bool = False) -> SigFactory.SigFactory:
    """Get the feature combinations paired to all possible distances

    :param features: {str} which pharmacophore features to consider; available: ["cats", "rdkit"]
    :param names: {bool} whether to return an array describing the bits with names of features and distances
    :return: RDKit signature factory to be used for 2D pharmacophore fingerprint calculation

    This is what's in the CATS factory: (in bins of 10)

    Acceptor Acceptor
    Acceptor Aromatic
    Acceptor Donor
    Acceptor Hydrophobe
    Acceptor NegIonizable
    Acceptor PosIonizable
    Aromatic Aromatic
    Aromatic Donor
    Aromatic Hydrophobe
    Aromatic NegIonizable
    Aromatic PosIonizable
    Donor Donor
    Donor Hydrophobe
    Donor NegIonizable
    Donor PosIonizable
    Hydrophobe Hydrophobe
    Hydrophobe NegIonizable
    Hydrophobe PosIonizable
    NegIonizable NegIonizable
    NegIonizable PosIonizable
    PosIonizable PosIonizable

    """
    if features == "cats":
        fdef = fdef_cats
    else:
        fdef = fdef_rdkit
    factory = ChemicalFeatures.BuildFeatureFactoryFromString(fdef)
    sigfactory = SigFactory.SigFactory(factory, useCounts=True, minPointCount=2, maxPointCount=2)
    sigfactory.SetBins([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)])
    sigfactory.Init()
    if names:
        descs = [sigfactory.GetBitDescription(i) for i in range(sigfactory.GetSigSize())]
        return sigfactory, descs
    else:
        return sigfactory


def cats(mol: Mol) -> np.array:
    """Function to calculate the CATS pharmacophore descriptor for one molecule.
    Descriptions of the individual features can be obtained from the function ``get_cats_sigfactory``.

    :param mol: {RDKit molecule} molecule to calculate the descriptor for
    :return: {numpy.ndarray} calculated descriptor vector
    """
    factory = get_cats_factory()
    arr = np.zeros((1,))
    ConvertToNumpyArray(Generate.Gen2DFingerprint(mol, factory), arr)
    scale = np.array([10 * [sum(arr[i: i + 10])] for i in range(0, 210, 10)]).flatten()
    return np.divide(arr, scale, out=np.zeros_like(arr), where=scale != 0).astype("float32")



fdef_cats = """
AtomType Hydroxylgroup [O;H1;+0]
AtomType OxygenAtom [#8]
AtomType PosCharge [+,++,+++,++++,++++]
AtomType NegCharge [-,--,---,----]
AtomType Carbon_AttachedOther [#6;$([#6]~[#7,#8,#9,#15,#16,#17,#35,#53,#14,#5,#34])]
AtomType CarbonLipophilic [#6;+0;!{Carbon_AttachedOther}]
AtomType ClBrI [#17,#35,#53]
AtomType SC2 [#16;X2]([#6])[#6]
AtomType NH_NH2_NH3 [#7;H1,H2,H3;+0]
AtomType NH0 [#7;H0;+0]
AtomType FlCl [#9,#17]
AtomType NH2 [#7;H2]
AtomType CSPOOH [C,S,P](=O)-[O;H1]

DefineFeature SingleAtomDonor [{Hydroxylgroup},{NH_NH2_NH3}]
  Family Donor
  Weights 1.0
EndFeature

DefineFeature SingleAtomAcceptor [{OxygenAtom},{NH0},{FlCl}]
  Family Acceptor
  Weights 1.0
EndFeature

DefineFeature SingleAtomPositive [{PosCharge},{NH2}]
  Family PosIonizable
  Weights 1.0
EndFeature

DefineFeature SingleAtomNegative [{NegCharge},{CSPOOH}]
  Family NegIonizable
  Weights 1.0
EndFeature

DefineFeature SingleAtomLipophilic [!a;{CarbonLipophilic},{ClBrI},{SC2}]
  Family Hydrophobe
  Weights 1.0
EndFeature

DefineFeature Arom4 [a]1[a][a][a]1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0
EndFeature

DefineFeature Arom5 [a]1[a][a][a][a]1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0
EndFeature

DefineFeature Arom6 [a]1[a][a][a][a][a]1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0
EndFeature

DefineFeature Arom7 [a]1[a][a][a][a][a][a]1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0,1.0
EndFeature

DefineFeature Arom8 [a]1[a][a][a][a][a][a][a]1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
EndFeature
"""

fdef_rdkit = """
# $Id$
#
# RDKit base fdef file.
# Created by Greg Landrum
#

AtomType NDonor [N&!H0&v3,N&!H0&+1&v4,n&H1&+0]
AtomType AmideN [$(N-C(=O))]
AtomType SulfonamideN [$([N;H0]S(=O)(=O))]
AtomType NDonor [$([Nv3](-C)(-C)-C)]

AtomType NDonor [$(n[n;H1]),$(nc[n;H1])]

AtomType ChalcDonor [O,S;H1;+0]
DefineFeature SingleAtomDonor [{NDonor},{ChalcDonor}]
  Family Donor
  Weights 1
EndFeature

# aromatic N, but not indole or pyrole or fusing two rings
AtomType NAcceptor [n;+0;!X3;!$([n;H1](cc)cc)]
AtomType NAcceptor [$([N;H0]#[C&v4])]
# tertiary nitrogen adjacent to aromatic carbon
AtomType NAcceptor [N&v3;H0;$(Nc)]

# removes thioether and nitro oxygen
AtomType ChalcAcceptor [O;H0;v2;!$(O=N-*)]
Atomtype ChalcAcceptor [O;-;!$(*-N=O)]

# Removed aromatic sulfur from ChalcAcceptor definition
Atomtype ChalcAcceptor [o;+0]

# Hydroxyls and acids
AtomType Hydroxyl [O;H1;v2]

# F is an acceptor so long as the C has no other halogen neighbors. This is maybe
# a bit too general, but the idea is to eliminate things like CF3
AtomType HalogenAcceptor [F;$(F-[#6]);!$(FC[F,Cl,Br,I])]

DefineFeature SingleAtomAcceptor [{Hydroxyl},{ChalcAcceptor},{NAcceptor},{HalogenAcceptor}]
  Family Acceptor
  Weights 1
EndFeature

# this one is delightfully easy:
DefineFeature AcidicGroup [C,S](=[O,S,P])-[O;H1,H0&-1]
  Family NegIonizable
  Weights 1.0,1.0,1.0
EndFeature

AtomType Carbon_NotDouble [C;!$(C=*)]
AtomType BasicNH2 [$([N;H2&+0][{Carbon_NotDouble}])]
AtomType BasicNH1 [$([N;H1&+0]([{Carbon_NotDouble}])[{Carbon_NotDouble}])]
AtomType PosNH3 [$([N;H3&+1][{Carbon_NotDouble}])]
AtomType PosNH2 [$([N;H2&+1]([{Carbon_NotDouble}])[{Carbon_NotDouble}])]
AtomType PosNH1 [$([N;H1&+1]([{Carbon_NotDouble}])([{Carbon_NotDouble}])[{Carbon_NotDouble}])]
AtomType BasicNH0 [$([N;H0&+0]([{Carbon_NotDouble}])([{Carbon_NotDouble}])[{Carbon_NotDouble}])]
AtomType QuatN [$([N;H0&+1]([{Carbon_NotDouble}])([{Carbon_NotDouble}])([{Carbon_NotDouble}])[{Carbon_NotDouble}])]


DefineFeature BasicGroup [{BasicNH2},{BasicNH1},{BasicNH0};!$(N[a])]
  Family PosIonizable
  Weights 1.0
EndFeature

# 14.11.2007 (GL): add !$([N+]-[O-]) constraint so we don't match
# nitro (or similar) groups
DefineFeature PosN [#7;+;!$([N+]-[O-])]
 Family PosIonizable
 Weights 1.0
EndFeature

# imidazole group can be positively charged (too promiscuous?)
DefineFeature Imidazole c1ncnc1
  Family PosIonizable
  Weights 1.0,1.0,1.0,1.0,1.0
EndFeature
# guanidine group is positively charged (too promiscuous?)
DefineFeature Guanidine NC(=N)N
  Family PosIonizable
  Weights 1.0,1.0,1.0,1.0
EndFeature

# the LigZn binder features were adapted from combichem.fdl
DefineFeature ZnBinder1 [S;D1]-[#6]
  Family ZnBinder
  Weights 1,0
EndFeature
DefineFeature ZnBinder2 [#6]-C(=O)-C-[S;D1]
  Family ZnBinder
  Weights 0,0,1,0,1
EndFeature
DefineFeature ZnBinder3 [#6]-C(=O)-C-C-[S;D1]
  Family ZnBinder
  Weights 0,0,1,0,0,1
EndFeature

DefineFeature ZnBinder4 [#6]-C(=O)-N-[O;D1]
  Family ZnBinder
  Weights 0,0,1,0,1
EndFeature
DefineFeature ZnBinder5 [#6]-C(=O)-[O;D1]
  Family ZnBinder
  Weights 0,0,1,1
EndFeature
DefineFeature ZnBinder6 [#6]-P(=O)(-O)-[C,O,N]-[C,H]
  Family ZnBinder
  Weights 0,0,1,1,0,0
EndFeature


# aromatic rings of various sizes:
#
# Note that with the aromatics, it's important to include the ring-size queries along with
# the aromaticity query for two reasons:
#   1) Much of the current feature-location code assumes that the feature point is
#      equidistant from the atoms defining it. Larger definitions like: a1aaaaaaaa1 will actually
#      match things like 'o1c2cccc2ccc1', which have an aromatic unit spread across multiple simple
#      rings and so don't fit that requirement.
#   2) It's *way* faster.
#

#
# 21.1.2008 (GL): update ring membership tests to reflect corrected meaning of
# "r" in SMARTS parser
#
AtomType AromR4 [a;r4,!R1&r3]
DefineFeature Arom4 [{AromR4}]1:[{AromR4}]:[{AromR4}]:[{AromR4}]:1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0
EndFeature
AtomType AromR5 [a;r5,!R1&r4,!R1&r3]
DefineFeature Arom5 [{AromR5}]1:[{AromR5}]:[{AromR5}]:[{AromR5}]:[{AromR5}]:1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0
EndFeature
AtomType AromR6 [a;r6,!R1&r5,!R1&r4,!R1&r3]
DefineFeature Arom6 [{AromR6}]1:[{AromR6}]:[{AromR6}]:[{AromR6}]:[{AromR6}]:[{AromR6}]:1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0
EndFeature
AtomType AromR7 [a;r7,!R1&r6,!R1&r5,!R1&r4,!R1&r3]
DefineFeature Arom7 [{AromR7}]1:[{AromR7}]:[{AromR7}]:[{AromR7}]:[{AromR7}]:[{AromR7}]:[{AromR7}]:1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0,1.0
EndFeature
AtomType AromR8 [a;r8,!R1&r7,!R1&r6,!R1&r5,!R1&r4,!R1&r3]
DefineFeature Arom8 [{AromR8}]1:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
EndFeature

# hydrophobic features
# any carbon that is not bonded to a polar atom is considered a hydrophobe
#
# 23.11.2007 (GL): match any bond (not just single bonds); add #6 at
#  beginning to make it more efficient
AtomType Carbon_Polar [#6;$([#6]~[#7,#8,#9])]
# 23.11.2007 (GL): don't match charged carbon
AtomType Carbon_NonPolar [#6;+0;!{Carbon_Polar}]

DefineFeature ThreeWayAttach [D3,D4;{Carbon_NonPolar}]
  Family Hydrophobe
  Weights 1.0
EndFeature

DefineFeature ChainTwoWayAttach [R0;D2;{Carbon_NonPolar}]
  Family Hydrophobe
  Weights 1.0
EndFeature

# hydrophobic atom
AtomType Hphobe [c,s,S&H0&v2,Br,I,{Carbon_NonPolar}]
AtomType RingHphobe [R;{Hphobe}]

# nitro groups in the RD code are always: *-[N+](=O)[O-]
DefineFeature Nitro2 [N;D3;+](=O)[O-]
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0
EndFeature

#
# 21.1.2008 (GL): update ring membership tests to reflect corrected meaning of
# "r" in SMARTS parser
#
AtomType Ring6 [r6,!R1&r5,!R1&r4,!R1&r3]
DefineFeature RH6_6 [{Ring6};{RingHphobe}]1[{Ring6};{RingHphobe}][{Ring6};{RingHphobe}][{Ring6};{RingHphobe}][{Ring6};{RingHphobe}][{Ring6};{RingHphobe}]1
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0,1.0,1.0,1.0
EndFeature

AtomType Ring5 [r5,!R1&r4,!R1&r3]
DefineFeature RH5_5 [{Ring5};{RingHphobe}]1[{Ring5};{RingHphobe}][{Ring5};{RingHphobe}][{Ring5};{RingHphobe}][{Ring5};{RingHphobe}]1
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0,1.0,1.0
EndFeature

AtomType Ring4 [r4,!R1&r3]
DefineFeature RH4_4 [{Ring4};{RingHphobe}]1[{Ring4};{RingHphobe}][{Ring4};{RingHphobe}][{Ring4};{RingHphobe}]1
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0,1.0
EndFeature

AtomType Ring3 [r3]
DefineFeature RH3_3 [{Ring3};{RingHphobe}]1[{Ring3};{RingHphobe}][{Ring3};{RingHphobe}]1
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0
EndFeature

DefineFeature tButyl [C;!R](-[CH3])(-[CH3])-[CH3]
  Family LumpedHydrophobe
  Weights 1.0,0.0,0.0,0.0
EndFeature

DefineFeature iPropyl [CH;!R](-[CH3])-[CH3]
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0
EndFeature
"""

