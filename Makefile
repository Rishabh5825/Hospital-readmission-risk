.PHONY: data train test serve dashboard docker

data:
	python src/data_prep.py

train: data
	python src/train.py

test:
	pytest tests/ -v

serve: train
	uvicorn api.app:app --reload

dashboard:
	streamlit run dashboard/streamlit_app.py

docker:
	docker build -t readmission-api .
	docker run -p 8000:8000 readmission-api
