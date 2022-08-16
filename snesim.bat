cd bin/
snesim.exe | echo "snesim.par" 

cd ../

jupyter nbconvert --execute --to notebook --inplace SNESIM.ipynb