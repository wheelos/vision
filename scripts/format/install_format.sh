#!/bin/bash

# Install clang-format, shfmt
apt-get -y update &&
	apt-get -y install \
		clang-format

# Install autopep8
pip3 install -q autopep8
