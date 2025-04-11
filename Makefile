# Makefile for managing the project

# Run commands with:
# make <target>
# Example: make run

run:
	python src/organize_media.py

clear:
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	mv cfg/processed_files_hash.json cfg/processed_files_hash.json.bak