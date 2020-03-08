FROM python:3.6

COPY dripline-python /dripline-python
COPY dragonfly /dragonfly
RUN pip install -e /dripline-python &&\
    pip install -e /dragonfly[colorlog,database,slack]

RUN pip install supervisor

COPY etc.supervisord /etc/supervisor
COPY config /config

EXPOSE 9001

CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
