
from astropy.table import Table
import numpy as np
import pandas as pd


def append_output(dataset, pwd, catalog_name, ens_modes, l1s, u1s, areas):

	dataset['phz_act'] = ens_modes
	dataset['lower_1sig_act'] = l1s
	dataset['upper_1sig_act'] = u1s
	dataset['pdf_in_inverval'] = areas

	tab = Table.from_pandas(dataset)
	tab.write(pwd + catalog_name + 'PICZL_photoz_act.fits', overwrite=True)
