# set base image (host OS) 
FROM python:3.8 
# Replicate the host user UID and GID to the image 
ARG UNAME=audituser 
ARG UID=1000 
ARG GID=1000 
RUN groupadd -g $GID -o $UNAME 
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME 
USER $UNAME 
# Identify the maintainer of an image 
LABEL maintainer="b.r.d.v.berlo@tue.nl" 
# Set the working directory in the container 
WORKDIR /project 
# Copy the content of the local directory to the working directory 
COPY ./Code . 
# Install dependencies 
RUN pip3 install -r Code/requirements.txt 
# Make run script executable 
CMD chmod +x Code/run.sh 
# In case you are not using a script to execute your project (run.sh is not present), 
you must comment the previous file.
