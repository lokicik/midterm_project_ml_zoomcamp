FROM python:3.8.12-slim

RUN pip install pipenv waitress flask pandas numpy scikit-learn==1.2.2 lightgbm xgboost

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y libgomp1
# hello
WORKDIR /app
COPY ["predict.py", "drinking.bin","smoking.bin","./"]





EXPOSE 7860

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:7860", "predict:app"]
