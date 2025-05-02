
from astropy.table import Table
import numpy as np
import pandas as pd


def append_output(dataset, pwd, catalog_name, ens_modes, l1s, u1s):

	dataset['phz_inact'] = ens_modes
	dataset['lower_1sig_inact'] = l1s
	dataset['upper_1sig_inact'] = u1s

	tab = Table.from_pandas(dataset)
	tab.write(pwd + catalog_name + 'PICZL_photoz_inact.fits', overwrite=True)
