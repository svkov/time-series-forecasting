FROM continuumio/miniconda:latest

WORKDIR /home/docker_conda_template

COPY environment.yml ./
RUN conda env create -f environment.yml

COPY app.py ./
COPY boot.sh ./

RUN echo "source activate data_loader" > ~/.bashrc
ENV PATH /opt/conda/envs/data_loader/bin:$PATH

RUN chmod +x boot.sh


EXPOSE 5000

ENTRYPOINT ["sh", "boot.sh"]
