make clean:
	rm -rf raw_data/noaa && rm -rf raw_data/wunderground

make:

make predictions:
	python predictor/main.py

make rawdata:
	cd raw_data && python noaa_download.py && python wunderground_download.py

make report: