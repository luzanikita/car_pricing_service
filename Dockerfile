FROM python:3.8

RUN pip install --upgrade pip

RUN git clone https://github.com/nigi4/car_pricing_service.git
WORKDIR car_pricing_service

COPY . .

RUN pip install -r requirements.txt

CMD python ./server.py
