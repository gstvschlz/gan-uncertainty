import functools
import os
import operator
import numpy as np
import pygeostat as gs
import matplotlib.pyplot as plt

os.makedirs("data/temp/selected/", exist_ok=True)

if not os.path.exists("bin/gan_results_header.out"):
    numpy_array = np.loadtxt("bin/gan_results.out")
    header_name = [f'ti_{idx}' for idx in range(numpy_array.shape[1])]
    header_name = f'\n'.join(header_name)

    # Save again with header
    np.savetxt(fname="bin/gan_results_header.out",
               X=numpy_array,
               header=f"gan_results\n{numpy_array.shape[1]}\n{header_name}", comments="")

conditional_samples = "data/conditioning_data/samples50"
generated_images = "bin/gan_results_header.out"
output = "bin/result.out"

ti_selection_p = gs.Program("bin/ti-selector.exe", parfile='param.par')

parstr = """                                    Parameters
                                   ************

 START OF PARAMETERS:
 {conditional_samples}                                       - samples datafile name
 50                                              - sample number
 150 0.5 1                                         - nx, xmn, xsiz; in sample migration grid
 150 0.5 1                                         - ny, ymn, ysiz; in sample migration grid
 1   0.5 1                                         - nz, zmn, zsiz; in sample migration grid
 {generated_images}                                        - training images datafile name
 150 0.5 1                                         - nx, xmn, xsiz; in training images
 150 0.5 1                                         - ny, ymn, ysiz; in training images
 1   0.5 1                                         - nz, zmn, zsiz; in training images
 25 25 0                                           - search radii in x,y and z directions
 1                                                - minimum event order to analyze
 10                                                - maximum event order to analyze
 0                                                 - compatibility mode: 0 for relative, 1 for absolute
 1                                                 - column (for absolute mode)
 0.5                                                 - tolerance parameter
 0                                                 - variable type
 0.1                                               - acceptance threshold (continuous variables)
 0.45                                                 - training image fraction to check
 12345                                             - random seed
 {output}                                        - output file name
"""

pars = dict(conditional_samples=conditional_samples,
            generated_images=generated_images, output=output)

ti_selection_p.run(parstr=parstr.format(**pars))

# Get similarities values
with open("bin/result.out", 'r') as file:
    # Convert to list of string
    lines = file.readlines()
    lines = [l.replace('\n', '').strip() for l in lines]
    idx_before: int = lines.index("Relative compatibility with training images (direct sampling):") + 1
    idx_after: int = lines.index("Occurrences of conditioning events in each training image:")
    sim = lines[idx_before:idx_after]

    sim_array = functools.reduce(operator.iconcat,
                                 [x.split() for x in sim], [])

    sim_array = list(
        map(float, sim_array))

similarity = [[idx, float(x)] for idx, x in enumerate(sim_array)]
sorted_similarity = sorted(similarity, key=operator.itemgetter(1), reverse=True)[:10]

print("Sorted similarity:", sorted_similarity)

for idx, similarity in sorted_similarity:
    nome = "data/temp/selected/teste_" + str(idx) + "_sim" + str(round(similarity, 4)) + ".png"

    plt.imshow(numpy_array[:, idx].reshape(150, 150), cmap="gray")
    plt.axis('off')
    plt.grid('off')
    plt.savefig(nome, dpi=300, bbox_inches='tight', transparent="True", pad_inches=0)
