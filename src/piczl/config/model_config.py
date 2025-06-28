
ACTIVE_CONFIG = {
	"model_dir": "/home/wroster/learning-photoz/PICZL_OZ/models/inactive",

	"model_files": ['crps_model_G=11_B=512_lr=0.0002.h5',
	'crps_model_G=4_B=256_lr=0.0002.h5',
	'crps_model_G=5_B=256_lr=0.00035.h5',
	'crps_model_G=7_B=256_lr=0.0005.h5',
	'nll_model_G=5_B=512_lr=0.0005.h5',
	'nll_model_G=4_B=512_lr=0.0005.h5',
	'nll_model_G=5_B=256_lr=0.0005.h5',
	'nll_model_G=3_B=512_lr=0.00035.h5'
	],

	"model_weights": [0.08227451309829409, 0.17245762536202783, 0.02718044442985964, 0.11249663791786871,
				0.2592408249176275, 0.1408404956224673, 0.02917528832932573, 0.1763341703225292]
}

INACTIVE_CONFIG = {
	"model_dir": "/home/wroster/learning-photoz/PICZL_OZ/models/active",

	"model_files": ['crps_model_G=11_B=512_lr=0.0002.h5',
	'crps_model_G=4_B=256_lr=0.0002.h5',
	'crps_model_G=5_B=256_lr=0.00035.h5',
	'crps_model_G=7_B=256_lr=0.0005.h5',
	'nll_model_G=5_B=512_lr=0.0005.h5',
	'nll_model_G=4_B=512_lr=0.0005.h5',
	'nll_model_G=5_B=256_lr=0.0005.h5',
	'nll_model_G=3_B=512_lr=0.00035.h5'
	],

	"model_weights": [0.08227451309829409, 0.17245762536202783, 0.02718044442985964, 0.11249663791786871,
				0.2592408249176275, 0.1408404956224673, 0.02917528832932573, 0.1763341703225292]}


CONFIGS = {
    "active": ACTIVE_CONFIG,
    "inactive": INACTIVE_CONFIG
}
