make clean:
	rm -rf raw_data/noaa && rm -rf raw_data/wunderground && rm -rf data/wunderground && rm -rf data/noaa

make: data

make predictions:
	python predictor/main.py

make rawdata:
	cd raw_data && python noaa_download.py && python wunderground_download.py

make data: rawdata
	cd data && python convert_raw.py