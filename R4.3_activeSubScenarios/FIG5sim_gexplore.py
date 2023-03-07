
## Simulate and check
import time
import pickle
import warnings
warnings.filterwarnings('ignore')  # For a clean output: omitting "overflow encountered in exp" warning.
from functions import simulate

# Simulate :: NEMOS_035 - may take a while
output = []
g_sel = [[2, 0.15, 0.022], [2, 0.12, 0.022], [2, 0.15, 0.09], [2, 0.15, 0.5]]
for g, pth, sth in g_sel:
    output.append(simulate("NEMOS_035", "jr", g=g, pth=pth, sigmath=sth, pcx=0.09, sigmacx=2.2e-8, th='pTh', t=60, mode="FIG"))

## Save simulations results using pickle
open_file = open("PAPER2/R5_activeSubScenarios/data/g_explore-FIG5_N35-initialConditions_v2-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + ".pkl", "wb")
pickle.dump([g_sel, output], open_file)
open_file.close()

