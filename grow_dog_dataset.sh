#!/bin/bash

# This script grows the dog dataset.
(cd Images/n02088094-Afghan_hound && cp `ls | tail -100` ../../data)  

(cd Images/n02085936-Maltese_dog && cp `ls | tail -100` ../../data)  

