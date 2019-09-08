FROM continuumio/miniconda3:latest

USER root

# Install extra packages if required
RUN apt-get update && \
    apt-get install -y redis-server && \
    rm -rf /var/lib/apt/lists/*

# Add the user that will run the app (no need to run as root)
#RUN groupadd -r ubuntu && useradd -r -g ubuntu ubuntu

RUN mkdir -p /app
WORKDIR /app

# Install app requirements
COPY environment-cpu.yml /app/environment.yml
RUN conda config --add channels conda-forge && \
    conda env create -f environment.yml && \
    rm -rf /opt/conda/pkgs/*

# Install app
COPY . /app/
#RUN chown -R myuser:myuser /app/*

# activate the myapp environment
#ENV PATH /opt/conda/envs/myapp/bin:$PATH

EXPOSE 8000

ENTRYPOINT [ ]
CMD [ "/app/start-all.sh" ]