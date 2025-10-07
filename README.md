Program to calculate gradient of scalar field given by field values in points. 
The gradient is calculated using a linear interpolation with least square method. 
The ranges for varying the parameters are calculated at the preliminary stage with primitve algorithm with probability of artifacts.
To prevent the influence of artifacts on the results, the exclusion of erroneous results due to exceeding the 2 standart deviation is used.
Also during the least square method results that results that deviate too much are also excluded but by 3 standart deviation.
Program was writen on C and later translated on C+.
