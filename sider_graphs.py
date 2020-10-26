import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import (InMemoryDataset, Data)
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import matplotlib.pyplot as plt

atoms_list = {'C': 0, 'O': 1, 'N': 2, 'S': 3, 'Cl': 4, 'P': 5, 'F': 6, 'Br': 7, 'I': 8,
              'Si': 9, 'B': 10, 'Na': 11, 'Se': 12, 'Fe': 13, 'As': 14, 'rare': 15}
bonds_list = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


class sider_graphs(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(sider_graphs, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['sider_mols.txt']

    @property
    def processed_file_names(self):
        return 'graphs.pt'

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            lines = f.readlines()
        data_list = []
        for line in lines:
            line = line.split()
            # print(line)
            # weight = torch.FloatTensor([float(line[2])])
            mol = Chem.MolFromSmiles(line[0])
            type_idx = []
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol not in atoms_list.keys():
                    symbol = 'rare'

                type_idx.append(atoms_list[symbol])
            x = F.one_hot(torch.tensor(type_idx),
                          num_classes=len(atoms_list)).to(torch.float)

            row, col, bond_idx = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_idx += 2 * [bonds_list[bond.GetBondType()]]
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = F.one_hot(torch.tensor(bond_idx),
                                  num_classes=len(bonds_list)).to(torch.float)
            N = mol.GetNumAtoms()
            edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

            data_list.append(Data(x=x, edge_attr=edge_attr, edge_index=edge_index))
        torch.save(self.collate(data_list), self.processed_paths[0])
