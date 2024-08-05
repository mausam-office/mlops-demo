# variables
SRC_DIR := src
PROJECT_NAME := ml_project
REQ_FILE := requirements.txt
TOUCHFILE := $(SRC_DIR)/.venv/touchfile

# OS based
# use git bash in windows to create venv
ifeq ($(OS), Windows_NT)
	VENV_DIR = Scripts
else
	VENV_DIR = bin
endif

ACTIVATE := . $(SRC_DIR)/.venv/$(VENV_DIR)/activate
TOUCH := touch $(TOUCHFILE)


venv: $(SRC_DIR)/.venv/touchfile

$(SRC_DIR)/.venv/touchfile: $(REQ_FILE)
	python -m venv $(SRC_DIR)/.venv
	$(ACTIVATE) && pip install -Ur $(REQ_FILE)
	$(TOUCH)

run: venv
	$(ACTIVATE) && python main.py

test: venv
	$(ACTIVATE) && pytest


