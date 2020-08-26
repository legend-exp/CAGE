FROM python:2.7

# deps
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y sqlite3 && \
    pip install flask

# config
RUN cd /opt && git clone -b 1.0.x https://github.com/gamegos/cesi
RUN cd /opt/cesi && sqlite3 userinfo.db < userinfo.sql
COPY cesi.conf /etc/cesi.conf

CMD python /opt/cesi/cesi/web.py

