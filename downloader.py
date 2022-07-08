import os
import zipfile 
import requests
from tqdm import tqdm
from pathlib import Path

def download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params={ 'id' : id }, stream=True)
	token = get_confirm_token(response)

	params = { 'id' : id, 'confirm' : 1 }
	response = session.get(URL, params=params, stream=True)

	save_response_content(response, destination)

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768
	print(response)
	with open(destination + '/models.zip', "wb") as f:
		for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

def download_model(destination, name = 'models'):

	download_file_from_google_drive('1CtV3i2rPDO9wKVEEPcIlw0cy67o7KIbb', destination)

	with zipfile.ZipFile(destination + '/models.zip', 'r') as zip_ref:
		zip_ref.extractall(destination)

	os.remove(destination + '/models.zip')

def downloader(destination):

	download_model(destination + '/models/')