FROM rocker/shiny:latest
LABEL maintainer="Никита <Krivolapovnikita9@gmail.com>"
RUN apt-get update && apt-get install -y libcurl4-gnutls-dev libssl-dev
RUN R -e "install.packages(c('shiny', 'shinyWidgets', 'digest', 'magrittr', 'htmltools'), repos='https://cran.rstudio.com/')"
COPY shiny-server.conf /etc/shiny-server/shiny-server.conf
COPY Rprofile.site /usr/local/lib/R/etc/
COPY . /srv/shiny-server/memory-hex/
EXPOSE 3838
