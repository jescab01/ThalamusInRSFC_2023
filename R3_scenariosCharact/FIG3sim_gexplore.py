
## Simulate and check
import time
import pickle
import warnings
warnings.filterwarnings('ignore')  # For a clean output: omitting "overflow encountered in exp" warning.
from functions import simulate

# Simulate :: NEMOS_035 - may take a while
output = []
g_sel = [[4, 0.022], [36, 0.022], [4, 2.2e-8], [36, 2.2e-8]]
for g, sth in g_sel:
    output.append(simulate("NEMOS_035", "jr", g=g, pth=0.09, sigmath=sth, pcx=0.09, sigmacx=2.2e-8, th='pTh', t=60, mode="FIG"))

## Save simulations results using pickle
open_file = open("PAPER2/R3_scenariosCharact/data/g_explore-FIG3_N35-initialConditions_v2-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + ".pkl", "wb")
pickle.dump([g_sel, output], open_file)
open_file.close()

