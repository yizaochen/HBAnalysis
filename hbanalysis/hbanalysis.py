from os import path, mkdir
from MDAnalysis import Universe
import numpy as np
import h5py
from hbanalysis.na_seq import sequences

atomname_map = {'A': {'type1': 'N6', 'type2': 'N1', 'type3': 'C2'}, 
                'T': {'type1': 'O4', 'type2': 'N3', 'type3': 'O2'},
                'C': {'type1': 'N4', 'type2': 'N3', 'type3': 'O2'},
                'G': {'type1': 'O6', 'type2': 'N1', 'type3': 'N2'}}
typelist = ['type1', 'type2', 'type3']
complement = {'A': 'T', 'C': 'G', 'T': 'A', 'G': 'C'}

class HBAgent:
    type_na = 'bdna+bdna'
    n_bp = 21

    def __init__(self, host, rootfolder):
        self.host = host
        self.rootfolder = rootfolder
        self.hostfolder = path.join(rootfolder, host)
        self.nafolder = path.join(self.hostfolder, self.type_na)
        self.inputfolder = path.join(self.nafolder, 'input')
        self.aafolder = path.join(self.inputfolder, 'allatoms')
        self.hbdatafolder = path.join(self.nafolder, 'hbdata')

        self.seq_guide = sequences[host]['guide']

        self.pdb = path.join(self.aafolder, f'{self.type_na}.npt4.all.pdb')
        self.xtc = path.join(self.aafolder, f'{self.type_na}.all.xtc')
        self.u = Universe(self.pdb, self.xtc)
        self.n_frames = len(self.u.trajectory)
        self.bp_atoms = self.__get_bp_pairs()

        self.hdf5_file = path.join(self.hbdatafolder, 'hb_distance_timeseries.h5')
        self.hb_data_hf = None

        self.__check_folder()

    def __check_folder(self):
        for folder in [self.hbdatafolder]:
            check_dir_exist_and_make(folder)

    def make_hdf5(self):
        hf = h5py.File(self.hdf5_file, 'w')
        for bpid in range(1, self.n_bp+1):
            for hbtype in typelist:
                key = f'{bpid}-{hbtype}'
                data = self.__get_hb_timeseries(bpid, hbtype)
                hf.create_dataset(key, data=data)
        frame_array, time_array = self.__get_frame_time_array()
        hf.create_dataset('Frame', data=frame_array)
        hf.create_dataset('Time', data=time_array)
        hf.close()
        print(f'Write HB-Distance Time series to {self.hdf5_file}')

    def read_hdf5(self):
        self.hb_data_hf = h5py.File(self.hdf5_file, 'r')
        print(f'Read HB-Distance Time series from {self.hdf5_file}')

    def get_frame_array(self):
        key = 'Frame'
        return np.array(self.hb_data_hf.get(key))

    def get_time_array(self):
        key = 'Time'
        return np.array(self.hb_data_hf.get(key))

    def get_hb_timeseries_from_hdf5(self, bpid, hbtype):
        key = f'{bpid}-{hbtype}'
        return np.array(self.hb_data_hf.get(key))

    def __get_bp_pairs(self):
        d_result = dict()
        for resid_i in range(1, self.n_bp+1):
            resname_i = self.seq_guide[resid_i-1]
            resname_j = complement[resname_i]
            resid_j = (2 * self.n_bp + 1) - resid_i
            for hbtype in typelist:
                key = f'{resid_i}-{hbtype}'
                atomname_i = atomname_map[resname_i][hbtype]
                atomname_j = atomname_map[resname_j][hbtype]
                atom_i = self.u.select_atoms(f'resid {resid_i} and name {atomname_i}')
                atom_j = self.u.select_atoms(f'resid {resid_j} and name {atomname_j}')
                if atom_i.n_atoms != 1:
                    raise InputException(f'Something wrong with {resid_i}-{atomname_i} in the {self.pdb}')
                if atom_j.n_atoms != 1:
                    raise InputException(f'Something wrong with {resid_j}-{atomname_j} in the {self.pdb}')
                d_result[key] = (atom_i, atom_j)
        return d_result

    def __get_hb_timeseries(self, bpid, hbtype):
        key = f'{bpid}-{hbtype}'
        data = np.zeros(self.n_frames)
        for idx, ts in enumerate(self.u.trajectory):
            if ts.frame == 0:
                print(f'Start {key} distance extraction.')
            atom_i, atom_j = self.bp_atoms[key]
            pos_i = atom_i.positions[0]
            pos_j = atom_j.positions[0]
            data[idx] = np.linalg.norm(pos_i - pos_j)
        return data

    def __get_frame_time_array(self):
        frame_array = np.zeros(self.n_frames)
        time_array = np.zeros(self.n_frames)
        for idx, ts in enumerate(self.u.trajectory):
            frame_array[idx] = ts.frame
            time_array[idx] = self.u.trajectory.time
        return frame_array, time_array


class InputException(Exception):
    pass

def check_dir_exist_and_make(file_path):
    if path.exists(file_path):
        print("{0} exists".format(file_path))
    else:
        print("mkdir {0}".format(file_path))
        mkdir(file_path)