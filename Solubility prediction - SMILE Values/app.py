import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors

## Calculate molecular descriptors
def AromaticProportion(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  AromaticAtom = sum(aa_count)
  HeavyAtom = Descriptors.HeavyAtomCount(m)
  AR = AromaticAtom/HeavyAtom
  return AR

def generate(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData= np.arange(1,1)
    i=0
    for mol in moldata:

        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)

        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1

    columnNames=["Molecular LogP","Molecular Weight","Number of Rotatable Bonds","Aromatic Proportion"]
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors

image = Image.open('molecule.jpg')

st.image(image, use_column_width=True)

st.write("""
# Molecular Solubility Prediction
***
""")

# Input molecules

st.sidebar.header('Input the value of SMILES (Simplified Molecular-Input Line-Entry System)')
## Read SMILES input
SMILES_input = "NCCCC\nCCC\nCN"
SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
SMILES = "C\n" + SMILES 
SMILES = SMILES.split('\n')
st.header('Input SMILES value')
SMILES[1:] 
## Calculate molecular descriptors
st.header('Computed molecular descriptor values')
X = generate(SMILES)
X[1:] 

# Reads in saved model
load_model = pickle.load(open('solubility_model.pkl', 'rb'))

prediction = load_model.predict(X)

st.header('Predicted LogS values')
prediction[1:] 