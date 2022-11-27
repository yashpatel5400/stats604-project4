make clean:
	rm -rf data/processed_noaa && rm -rf data/processed_wunderground

make rawdata:
	rm -rf data/raw_noaa && rm -rf data/raw_wunderground && python raw_data/noaa_download.py && python raw_data/wunderground_download.py

make data: data/raw_noaa data/raw_wunderground
	python data/process_noaa.py && python data/process_wunderground.py

make predictions:
	python predictor/main.py
