FROM continuumio/miniconda:latest

WORKDIR /home/docker_conda_template

COPY environment.yml ./
#RUN conda config --add channels conda-forge
#RUN conda config --append channels ranaroussi
RUN conda env create -f environment.yml

COPY app.py ./
COPY boot.sh ./
RUN chmod +x boot.sh
RUN echo "source activate data_loader" > ~/.bashrc
ENV PATH /opt/conda/envs/data_loader/bin:$PATH
EXPOSE 5000
ENTRYPOINT ["sh", "boot.sh"]
