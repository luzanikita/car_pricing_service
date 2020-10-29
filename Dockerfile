FROM python:3.8

RUN git clone https://github.com/nigi4/car_pricing_service.git
WORKDIR car_pricing_service

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD python ./server.py
