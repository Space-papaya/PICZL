
from astropy.table import Table
import numpy as np
import pandas as pd


def append_output(dataset, pwd, catalog_name):

	dataset['phz'] = ens_modes
	dataset['lower_1sig'] = lower_1sig
	dataset['upper_1sig'] = upper_1sig
	dataset['lower_3sig'] = lower_3sig
	dataset['upper_3sig'] = upper_3sig

	tab = Table.from_pandas(dataset)
	tab.write(pwd + catalog_name + 'PICZL_photoz.fits', overwrite=True)
