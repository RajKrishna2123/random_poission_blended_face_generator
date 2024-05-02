FROM python:3.11

WORKDIR /app

RUN apt update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgtk-3-dev 

RUN pip install pandas \
    matplotlib \
    scikit-image \
    scikit-learn \
    opencv-python \
    django \
    scipy 

EXPOSE 8000

CMD ["/bin/bash"]
