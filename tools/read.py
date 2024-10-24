"""
Collection of data objects that contain Athena output.
"""

import os
import numpy as np
from scipy.optimize import minimize
from glob import glob


class DataHST:
    """
    Contains data from the Athena history (hst) file
    volume averaged quantities over time
    """

    version = '3.0'
    
    def __init__(self, filename, silent=False):
        """
        : filename : the hst file name
        : silent   : print basic info or not
        """
        
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"'{filename}' does not exist.")
        
        self.data = {}
        
        with open(filename, 'r') as f:
            
            eof = f.seek(0,2)
            f.seek(0,0)
            
            f.readline() # skip line
            
            line = f.readline()
            self.names = [x.split('=')[-1] for x in line[:-1].split('  ')[1:] if x != '']
            
            f.readline() # skip line
            
            for name in self.names:
                
                self.data[name] = []
                
            while f.tell() != eof:
                
                line = f.readline()
                                
                for i, name in enumerate(self.names):

                    self.data[name] += [np.array(line.split(), float)[i]]
            
        if not silent:
            print("horizontally averaged quantities:\n", self.names)


class DataVTK:
    """
    Contains data and metadata from Athena VTK files
    """
    
    version = '3.3'
    
    def __init__(self, filename, silent=True):
        """
        : filename : the VTK file name
        : silent   : print basic info or not
        """
        
        if not os.path.exists(filename):
            raise NameError(f"File {filename} does not exist")
        
        with open(filename, 'rb') as f:
            
            eof = f.seek(0,2) # mark the end of the file
            f.seek(0,0)       # and go back to the beginning
                        
            # read the header metadata
            
            self._metadata(f)
            
            self.box_size = self.dx * self.Nx
            self._set_cell_centers()
            
            # now handle the scalar and vector data
            
            self.data    = {} # data to be in {'name': np array} dictionary
            self.svtypes = [] # scalar or vector
            self.names   = [] # name of quantity
            self.dtypes  = [] # data type
            
            shape3d = np.hstack([np.flipud(self.Nx[:self.dim]), 3])
            
            while f.tell() != eof:
                
                line = f.readline().decode('utf-8').split()
                if line == []:
                    line = f.readline().decode('utf-8').split()
                    if f.tell() >= eof:
                        # sometimes there is an extra byte in dpar.vtk files. Why?
                        break
                
                self.svtypes.append(line[0])
                self.names.append(line[1])
                self.dtypes.append(line[2])
                
                if line[0] == "SCALARS":
                    f.readline() # skip line
                    
                    data = np.fromfile(f, np.dtype('f'), self.size)
                    self.data[line[1]] = data.byteswap().reshape(np.flipud(self.Nx[:self.dim]))
                
                if line[0] == "VECTORS":
                    
                    data = np.fromfile(f, np.dtype('f'), 3*self.size) # 3D vector fields even for 2D simulations
                    self.data[line[1]] = data.byteswap().reshape(shape3d)
            
            f.close()
                        
        if not silent:
            print("Scalars:", [name for i, name in enumerate(self.names) if self.svtypes[i]=="SCALARS"])
            print("Vectors:", [name for i, name in enumerate(self.names) if self.svtypes[i]=="VECTORS"])
    
    @property
    def tn(self, omega=1.0):
        """
        Computes the nearest periodic point to current time 't'
        tn = n*Ly/(q*omega*Lx)
        assumes q = 1.5 (Keplerian disk)
        """
        
        Lx, Ly, _ = self.box_size

        def func(n, t):
            tn = n*Ly/(1.5*omega*Lx)
            return np.abs(tn - t)

        sol = minimize(func, 0, args=self.t)

        n = round(*sol.x)

        return func(n, 0)
    
    def _metadata(self, f):
        
        f.readline() # skip line
            
        line = f.readline().decode('utf-8')
        self.t      = float(line[line.find("time")+6:line.find(", level")])
        self.level  = int(line[line.find("level")+7:line.find(", domain")])
        self.domain = int(line[line.find("domain")+8:])

        line = f.readline().decode('utf-8')[:-1]
        assert line == "BINARY", f"VTK file does not contain binary data, contains {line}"

        line = f.readline().decode('utf-8')[:-1]
        assert line == "DATASET STRUCTURED_POINTS", f"{line} is not supported"

        line = f.readline().decode('utf-8')
        self.Nx = np.array(line.split()[1:], int) - 1
        if self.Nx[2] == 0:
            self.dim = 2
        else:
            self.dim = 3

        line = f.readline().decode('utf-8')
        assert line[:6] == "ORIGIN", f"no ORIGIN, {line}"
        self.origin = np.array(line.split()[1:], float)

        line = f.readline().decode('utf-8')
        assert line[:7] == "SPACING", f"no SPACING, {line}"
        self.dx = np.array(line.split()[1:], float)

        line = f.readline().decode('utf-8')
        assert line[:9] == "CELL_DATA", f"no CELL_DATA, {line}"
        self.size = int(line[10:])
            
    def _set_cell_centers(self):
        # need to handle cases when 2D data is xz or yz

        self.ccx = np.linspace(self.origin[0] + 0.5*self.dx[0],
                               self.origin[0] + self.box_size[0] - 0.5*self.dx[0],
                               self.Nx[0])
        self.ccy = np.linspace(self.origin[1] + 0.5*self.dx[1],
                               self.origin[1] + self.box_size[1] - 0.5*self.dx[1],
                               self.Nx[1])
        self.ccz = np.linspace(self.origin[2] + 0.5*self.dx[2],
                               self.origin[2] + self.box_size[2] - 0.5*self.dx[2],
                               self.Nx[2])

class Data1D:
    """
    Contains data from (custom) Athena 1d files
    horizontally (xy) averaged quantities
    """

    version = '3.0'
    
    def __init__(self, filename, silent=False):
        """
        : filename : the file name
        : silent   : print basic info or not
        """
        
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"'{filename}' does not exist.")
        
        self.data = {}
        
        with open(filename, 'r') as f:
            
            eof = f.seek(0,2)
            f.seek(0,0)
            
            self.names = f.readline().split()[1:]
            
            for name in self.names:
                
                self.data[name] = []
                
            while f.tell() != eof:
                
                line = f.readline()
                                
                for i, name in enumerate(self.names):

                    self.data[name] += [np.array(line.split(), float)[i]]
            
        if not silent:
            print("horizontally averaged quantities:\n", self.names, "\n")
            
            
class SpaceTimeData:
    """
    Creates and stores space-time data for quantities from a directory of Athena 1d files
    """
    
    version = '1.1'
        
    def __init__(self, path, dt=1.0, fmt='1d', silent=False):
      """
      path : directory containing the series of 1d files
      dt   : timestep between 1d file outputs
      """
        
        if path[-1] not in ['\\', '/']:
            path += '/'
        
        if not os.path.exists(path):
            raise NameError("path does not exist!")
            
        self.data = {}
        
        self.file_count = len([name for name in os.listdir(path) if os.path.isfile(path+name) and f".{fmt}" in name])
        if not self.file_count > 0:
            raise FileNotFoundError(f"No files with extension '{fmt}' were found in '{path}'")
            
        print(path)
        
        for i, f in enumerate(sorted(glob(path+f"*.{fmt}"))):
            
            if i == 0: # first loop
                
                do = Data1D(f)
                
                self.progress(0, self.file_count)
                
                self.names = do.names
                
                for name in self.names:
                    self.data[name] = do.data[name]
                    
                if 'x3' in self.data:
                    self.z = do.data['x3']
                elif 'x1' in self.data:
                    self.z = do.data['x1']
                self.t = np.arange(self.file_count)*dt
                self.t_orbit = self.t/(2*np.pi)
                
                self.Z, self.T = np.meshgrid(self.z, self.t)
                self.T_orbit = self.T/(2*np.pi)
                    
                continue # end first loop
                    
            self.progress(i+1, self.file_count)
            
            do = Data1D(f, silent=True)

            for name in self.names:
                self.data[name] = np.append(self.data[name], do.data[name])
                
        # reshape data
        for name in self.names:
            self.data[name] = self.data[name].reshape(self.Z.shape)

        print("\n")
            
    def progress(self, it, total):
    
        fill = "█"
        length = 50

        fraction = it/total
        filledLength = int(length*fraction)
        
        bar = fill*filledLength + "-"*(length - filledLength)    

        print(f"\rProgress |{bar}| {100*fraction:.1f}% Complete", end='\r')
