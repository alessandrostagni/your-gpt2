FROM python:latest
WORKDIR /usr/app/src
COPY nlp_bot.py ./

RUN python3 -m pip install -U discord.py

CMD [ "python", "./nlp_bot.py"]