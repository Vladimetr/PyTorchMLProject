FROM python:3.7

# pip install

# Добавить пользователя non-root
RUN useradd -m vova --uid=1018 -U
# -m create home dir /home/vova
# -U create same groupID and set user to this group

# -> cat /etc/passwd
# vova:x:1018:1018::/home/vova:/bin/sh

USER vova:1018
WORKDIR /home/vova

CMD ["/bin/bash"]