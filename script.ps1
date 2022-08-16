cd .\gan_bin\

for($i = 0; $i -lt 100; $i++){
     echo "___ SIMULATION $i ___"
     echo "snesim_gan.par" | .\snesim.exe
     [int]$sum = [int]$i+1
     ((Get-Content -path snesim_gan.par -Raw) -replace "snesim_$i.out","snesim_$sum.out") | Set-Content -Path snesim_gan.par
     ((Get-Content -path snesim_gan.par -Raw) -replace "ti_$i.out", "ti_$sum.out") | Set-Content -Path snesim_gan.par
}

cd .\..