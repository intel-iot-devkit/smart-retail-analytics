FROM grafana/grafana:5.3.1

USER root
ENV http_proxy 'http://proxy-iind.intel.com:911'
ENV https_proxy 'http://proxy-iind.intel.com:911'
RUN apt-get update && apt-get install -y curl gettext-base && rm -rf /var/lib/apt/lists/*

WORKDIR /etc/grafana
COPY datasources ./datasources

WORKDIR /app
COPY entrypoint.sh ./
RUN chmod u+x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
