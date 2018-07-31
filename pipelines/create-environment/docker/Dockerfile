# Use an official Python runtime as a parent image
FROM continuumio/anaconda3

ENV JUPYTER_PASSWORD sha1:7c4613b10073:01d7dcef0bdae99ced22d7aebaa61d931f446b27
# Obtain the SHA hash for your chosen password and copy it above. To do this, use:
# from notebook.auth import passwd
# passwd()
# You'll be asked to put in your password twice. The SHA value will display.
# Copy it (without the single quotes) and paste it above. This will be the password you use to login to
# the Jupyter server.
RUN apt-get install -y sox
RUN apt-get install -y ffmpeg
ADD requirements.txt /
RUN pip install -r /requirements.txt
RUN git clone https://github.com/pseeth/scaper && \
    cd scaper && \
    git checkout source_separation \
    && pip install -e .
RUN conda install pytorch torchvision cuda92 -c pytorch
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.password='${JUPYTER_PASSWORD}'" >> /root/.jupyter/jupyter_notebook_config.py
RUN jupyter nbextension enable --py widgetsnbextension

ENTRYPOINT ["python"]
