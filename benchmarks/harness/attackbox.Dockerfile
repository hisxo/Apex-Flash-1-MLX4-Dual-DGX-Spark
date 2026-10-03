FROM pwntools/pwntools@sha256:243d643d77ab8366e716158183c624d5f1d4493397ec7cb906e340d2025437ac

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        binutils \
        ca-certificates \
        curl \
        file \
        gdb \
        git \
        jq \
        ltrace \
        netcat-openbsd \
        strace \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
CMD ["tail", "-f", "/dev/null"]
