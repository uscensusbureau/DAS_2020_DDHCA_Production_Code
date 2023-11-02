#!/bin/bash

# Create the virtual env.
echo "--------------------------------"
echo "Creating a virtual environment..."
echo "--------------------------------"
python3 -m venv .venv
printf "\nVirtual environment created."
printf "\n\n\n"

# Activate the virtual env.
echo "--------------------------------"
echo "Activating the virtual environment..."
echo "--------------------------------"
source .venv/bin/activate
printf "\nVirtual environment activated."
printf "\n\n\n"

# Download and install dependencies.
echo "--------------------------------"
echo "Making the local package..."
echo "--------------------------------"
make package
printf "\nPackage made."
printf "\n\n\n"

# Deactivate the virtual env
echo "--------------------------------"
echo "Deactivating the virtual environment..."
echo "--------------------------------"
deactivate
printf "\nVirtual environment deactivated."
printf "\n\n\n"

# Run make venv.
echo "--------------------------------"
echo "Running make venv..."
echo "--------------------------------"
make venv
printf "\nVenv made."
printf "\n\n\n"

# Run safetab_h_cef_run_tests.sh.
echo "--------------------------------"
echo "Running safetab_h_cef_run_tests.sh..."
echo "Does the following path to run the test look correct?"
echo "VENV=${PWD}/virtualenv tests/safetab_h_cef_run_tests.sh"
echo "--------------------------------"
read -p "Say y/n to continue: " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
  printf "\n"
  VENV=${PWD}/virtualenv tests/safetab_h_cef_run_tests.sh
fi
printf "\nsafetab_h_cef_run_tests.sh finished."
printf "\n\n\n"

# Run safetab_p_cef_run_tests.sh.
echo "--------------------------------"
echo "Running safetab_p_cef_run_tests.sh..."
echo "Does the following path to run the test look correct?"
echo "VENV=${PWD}/virtualenv tests/safetab_p_cef_run_tests.sh"
echo "--------------------------------"
read -p "Say y/n to continue: " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
  printf "\n"
  VENV=${PWD}/virtualenv tests/safetab_p_cef_run_tests.sh
fi
printf "\nsafetab_p_cef_run_tests.sh finished."
printf "\n\n\n"

# Checkpoint
read -r -p "Press enter to exit."
