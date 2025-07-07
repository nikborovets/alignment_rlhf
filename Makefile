.PHONY: all sft rm reinforce eval clean

all: sft rm reinforce eval

sft:
	python scripts/download_sft.py

rm:
	PYTHONPATH=$$(pwd) python scripts/train_rm_manual.py

reinforce:
	PYTHONPATH=$$(pwd) python scripts/train_reinforce.py

eval:
	PYTHONPATH=$$(pwd) python scripts/eval.py

clean:
	rm -rf models/*
	rm -rf reports/*.csv
