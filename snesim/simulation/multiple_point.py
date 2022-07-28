"""
Usage

```
from snesim.simulation.multiple_point import Snesim

ssim = Snesim(ti_path="bin/ti_strebelle.out",
              par_file_path="bin/parameter.par",
              snesim_path="bin/snesim.exe")

ssim.sim_parameters("bin/samples50.out", "results_pg.out", 1)
ssim.create_parameter_file()

ssim.run()
```

"""

import os
import numpy
import pandas
import pygeostat as gs
from snesim.plots import *

SEED: int = 303258


class Snesim:
    """ Class to do SNESIM simulations. """

    def __init__(self, ti_path: str, par_file_path: str, snesim_path: str):
        super(Snesim, self).__init__()
        self.parameter = None
        self.ti_path: str = ti_path
        self.par_path: str = par_file_path
        self.snesim_path: str = snesim_path
        self.snesim_exe = None

        self.samples_path: str
        self.output_path: str
        self.realizations: int

    def __set_parameters(self, samples_path: str, output_path: str,
                         realizations: int) -> None:
        self.samples_path = samples_path
        self.output_path = output_path
        self.realizations = realizations

    def sim_parameters(self, samples_path: str,
                       output_path: str, realizations: int) -> None:
        self.__set_parameters(samples_path, output_path, realizations)

    def _create_snesim_executable(self) -> object:
        self.snesim_exe = gs.Program(self.snesim_path, parfile=self.parameter)

    def create_parameter_file(self) -> None:
        self.parameter: str = f"""
Parameters for SNESIM
***********************
START OF PARAMETERS:
{self.samples_path}           - file with original data
1  2  3  4                    - columns for x, y, z, variable
2                             - number of categories
0   1                         - category codes
0.7  0.3                      - (target) global pdf
0                             - use (target) vertical proportions (0=no, 1=yes)
vertprop.dat                  - file with target vertical proportions
0                             - servosystem parameter (0=no correction)
-1                            - debugging level: 0,1,2,3
snesim.dbg                    - debugging file
{self.output_path}            - file for simulation output
{self.realizations}           - number of realizations to generate
150    0.5    1.0              - nx,xmn,xsiz
150    0.5    1.0              - ny,ymn,ysiz
1     0.5    1.0              - nz,zmn,zsiz
303258                        - random number seed
30                            - max number of conditioning primary data
10                            - min. replicates number
0	0                       - condtion to LP (0=no, 1=yes), flag for iauto
1.0	1.0                     - two weighting factors to combine P(A|B) and P(A|C)
localprop.dat                 - file for local proportions
0                             - condition to rotation and affinity (0=no, 1=yes)
rotangle.dat                  - file for rotation and affinity
3                             - number of affinity categories
1.0  1.0  1.0                 - affinity factors (X,Y,Z)     
1.0  0.6  1.0                 - affinity factors             
1.0  2.0  1.0                 - affinity factors             
6                             - number of multiple grids
ti_strebelle.out              - file for training image
250  250  1                   - training image dimensions: nxtr, nytr, nztr
1                             - column for training variable
10.0   10.0   5.0             - maximum search radii (hmax,hmin,vert)
0.0    0.0   0.0              - angles for search ellipsoid
"""

        if self.snesim_exe is None:
            self._create_snesim_executable()

        self.snesim_exe.writepar(self.parameter, self.par_path)

    def run(self):
        self.snesim_exe.run(parstr=self.parameter, parfile=self.par_path)

    def _read_simulation_output(self, nanval: int = -997799) -> object:
        debug_level = 1
        if not (os.path.isfile(self.output_path)):
            print("Filename:'%s', does not exist" % self.output_path)

        file = open(self.output_path, "r")
        if debug_level > 0:
            print("eas: file ->%20s" % self.output_path)

        eas = {'title': (file.readline()).strip('\n')}

        if debug_level > 0:
            print("eas: title->%20s" % eas['title'])

        dim_arr = eas['title'].split()
        if len(dim_arr) == 3:
            eas['dim'] = {}
            eas['dim']['nx'] = int(dim_arr[0])
            eas['dim']['ny'] = int(dim_arr[1])
            eas['dim']['nz'] = int(dim_arr[2])

        eas['n_cols'] = int(file.readline())

        eas['header'] = []
        for i in range(0, eas['n_cols']):
            # print (i)
            h_val = (file.readline()).strip('\n')
            eas['header'].append(h_val)

            if debug_level > 1:
                print("eas: header(%2d)-> %s" % (i, eas['header'][i]))

        file.close()

        try:
            eas['D'] = numpy.genfromtxt(self.output_path, skip_header=2 + eas['n_cols'])
            if debug_level > 1:
                print("eas: Read data from %s" % self.output_path)
        except:
            print("eas: COULD NOT READ DATA FROM %s" % self.output_path)

        # add NaN values
        try:
            eas['D'][eas['D'] == nanval] = numpy.nan
        except:
            print("eas: FAILED TO HANDLE NaN VALUES (%d(" % nanval)

        # If dimensions are given in title, then convert to 2D/3D array
        if "dim" in eas:
            if eas['dim']['nz'] == 1:
                eas['Dmat'] = eas['D'].reshape((eas['dim']['ny'], eas['dim']['nx']))
            elif eas['dim']['nx'] == 1:
                eas['Dmat'] = numpy.transpose(eas['D'].reshape((eas['dim']['nz'], eas['dim']['ny'])))
            elif eas['dim']['ny'] == 1:
                eas['Dmat'] = eas['D'].reshape((eas['dim']['nz'], eas['dim']['nx']))
            else:
                eas['Dmat'] = eas['D'].reshape((eas['dim']['nz'], eas['dim']['ny'], eas['dim']['nx']))

            eas['Dmat'] = eas['D'].reshape((eas['dim']['nx'], eas['dim']['ny'], eas['dim']['nz']))
            eas['Dmat'] = eas['D'].reshape((eas['dim']['nz'], eas['dim']['ny'], eas['dim']['nx']))

            eas['Dmat'] = numpy.transpose(eas['Dmat'], (2, 1, 0))

            if debug_level > 0:
                print("EAS: Converted data in matrixes (Dmat)")

        eas['filename'] = self.output_path

        return eas

    def get_realizations(self) -> object:
        # Gets only first column of data
        realizations = self._read_simulation_output()["D"][:, 0]

        # Returns realizations based on n_real, nx, ny
        return realizations.reshape(100, 150, 150)


