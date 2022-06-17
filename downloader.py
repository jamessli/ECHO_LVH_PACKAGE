import os
import zipfile 
import logging
import requests
from tqdm import tqdm


MODEL_TO_URL = {
	'models': 'https://drive.google.com/file/d/1hWOCpw2gzhrBd0E_8sLFzGDvOQoGAzlM/view?usp=sharing',
}

def download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params={ 'id' : id }, stream=True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params=params, stream=True)

	save_response_content(response, destination)

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

def download_model(destination, name = 'models'):
	project_dir = os.path.dirname(os.path.abspath(__file__))

	file_destination = destination
	file_id = MODEL_TO_URL[name].split('id=')[-1]
	logging.info(f'Downloading {name} model (~1000MB tar.xz archive)')
	download_file_from_google_drive(file_id, file_destination)

	logging.info('Extracting model from archive (~1300MB folder)')
	with zipfile.ZipFile(file_destination, 'r') as zip_ref:
		zip_ref.extractall(path=os.path.dirname(file_destination))

	logging.info('Removing archive')
	os.remove(file_destination)
	logging.info('Done.')

def downloader(model_path):

	download_model(model_path)