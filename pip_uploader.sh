#!/bin/bash

# Clean first dist and build
rm -rf dist
rm -rf build
python setup.py sdist bdist_wheel
twine check dist/*

if [[ $1 == "publish" ]]
then
    echo "Publishing to pip"
    twine upload dist/*
else
    echo "Publishing to pip test"
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
fi
