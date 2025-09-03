# Dockerfile for security scanner sandbox
FROM python:3.13-slim

# Install essential tools for security testing
RUN apt-get update && apt-get install -y \
  curl \
  wget \
  git \
  vim \
  nano \
  netcat-traditional \
  dnsutils \
  iputils-ping \
  traceroute \
  nmap \
  sqlmap \
  dirb \
  perl \
  libnet-ssleay-perl \
  libio-socket-ssl-perl \
  libwww-perl \
  ca-certificates \
  unzip \
  golang-go \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install nikto (from upstream)
RUN git clone --depth 1 https://github.com/sullo/nikto.git /opt/nikto && \
  ln -s /opt/nikto/program/nikto.pl /usr/local/bin/nikto && \
  chmod +x /opt/nikto/program/nikto.pl

# Install nuclei (build from source with Go for portability)
RUN set -eux; \
  export GO111MODULE=on; \
  export GOPATH=/go; \
  mkdir -p "$GOPATH"; \
  go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest; \
  install "$GOPATH/bin/nuclei" /usr/local/bin/nuclei; \
  rm -rf "$GOPATH"

# Install gobuster and ffuf (build from source with Go for portability)
RUN set -eux; \
  export GO111MODULE=on; \
  export GOPATH=/go; \
  mkdir -p "$GOPATH"; \
  go install github.com/OJ/gobuster/v3@latest; \
  go install github.com/ffuf/ffuf/v2@latest; \
  install "$GOPATH/bin/gobuster" /usr/local/bin/gobuster; \
  install "$GOPATH/bin/ffuf" /usr/local/bin/ffuf; \
  rm -rf "$GOPATH"

# Create non-root user for safer execution
RUN useradd -m -s /bin/bash user && \
  mkdir -p /home/user && \
  chown -R user:user /home/user

# Set working directory
WORKDIR /home/user

# Switch to non-root user
USER user

# Default command to keep container running
CMD ["sleep", "infinity"]
