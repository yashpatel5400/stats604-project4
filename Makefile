.PHONY: clean rawdata data predictions

data: 
	python data/process_noaa.py
	python data/process_wunderground.py

rawdata:
	rm -rf data/raw_noaa
	rm -rf data/raw_wunderground
	python raw_data/noaa_download.py
	python raw_data/wunderground_download.py

clean:
	rm -rf data/processed_noaa
	rm -rf data/processed_wunderground

predictions:
	python predictor/main.py
