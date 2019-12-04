#!/bin/bash

# This script grows the dog dataset.
cd Stanford_Dogs/n02088094-Afghan_hound
cp `ls | tail -100` ../../data  

cd ../n02085936-Maltese_dog       
cp `ls | tail -100` ../../data   

