wget 'https://satdat.ngdc.noaa.gov/sem/goes/data/sat_locations/goes13/dn_goes-l2-orb1m_g13_y2013_v0_0.nc'

for i in $(seq -w 1 31); do

    wget -r -l 1 --no-directories "https://satdat.ngdc.noaa.gov/sem/goes/data/science/mag/goes13/magn-l2-hires/2013/10/dn_magn-l2-hires_g13_d201310${i}_v0_0_2.nc"
    
done
    