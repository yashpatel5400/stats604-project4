make clean:
	rm -rf raw_data

make:

make predictions:
	python predictor/main.py

make rawdata:
	cd raw_data && python noaa_download.py && python wunderground_download.py

make report: