FROM grafana/grafana:8.2.0-ubuntu

USER root
ENV http_proxy 'http://proxy-dmz.intel.com:911'
ENV https_proxy 'http://proxy-dmz.intel.com:911'
ENV no_proxy 'no_proxy=127.0.0.1,localhost,127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.svc,.svc.cluster.local,intel.com,10.2.127.101,.intel.com'
RUN apt-get update && apt-get install -y curl gettext-base && rm -rf /var/lib/apt/lists/*

WORKDIR /etc/grafana
COPY datasources ./datasources

WORKDIR /app
COPY entrypoint.sh ./
RUN chmod u+x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
