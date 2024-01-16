from desilike.base import BaseCalculator
from cosmosis.runtime.pipeline import LikelihoodPipeline
import os



class DESY3Observable(BaseCalculator):


	def initialize(self, ini_file_dir=None, ini_file_name=None, cosmosis_dir=None, theory=None):
		os.environ['COSMOSIS_STD_DIR'] = cosmosis_dir
		os.environ['INI_FILE_DIR'] = ini_file_dir

		ini_file = ini_file_dir + '/' + ini_file_name
		cosmosis_pipe = LikelihoodPipeline(ini_file)
		cosmosis_data = cosmosis_pipe.build_starting_block([])
		print('Running Cosmosis to extract data and covariance')
		cosmosis_pipe.run(cosmosis_data)

		self.flatdata = cosmosis_data['data_vector','2pt_data']
		self.covariance = cosmosis_data['data_vector','2pt_covariance']

		self.theory = theory


	def calculate(self):
		self.flattheory = self.theory.theory_vector


