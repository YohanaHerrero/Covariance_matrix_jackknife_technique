# Covariance_matrix_jackknife_technique
Following the Jackknife technique (spliting parent sample into independent subsamples), the covariance matrix of a set of measurements is computed.

## **What is this repository for?**

When data points are correlated, one can account for this correlation using the jackknife resampling technique and computing the covariance matrix. The jackknife approach requires a division of the sample area into several independent regions, each of which must be large enough to cover the full range of scales under consideration. By excluding each of the divisions (one at a time), measurements are perfomed on each subsample. By carrying out as many measurements as sample divisions, the covariance matrix is calculated, from which error bars are afterwards inferred.

I use polygons to split the data sample, plot the jackknife regions, and the covariance matrix with imshow.

## **Installing Covariance_matrix_jackknife_technique**

No installation is needed. Simply cloning the GitHub repository and importing the script is enough. Hence, 

```
    git clone https://github.com/YohanaHerrero/Covariance_matrix_jackknife_technique.git
```

The code is written in Python and uses a range of default packages included in standard installations of Python:

### **Standard packages:**

- numpy  
- matplotlib
- math
- shapely

### **Special packages:**

- astropy 

After adding the Covariance_matrix_jackknife_technique directory to the PYTHONPATH or changing location to the Covariance_matrix_jackknife_technique directory, the repository can be imported in python with:

```
    import Covariance_matrix_jackknife_technique
```

Besides the python packages, you will also need a fits table containing the following columns:

- RA in degrees
- DEC in degrees
- Z 

Instead, the three parameters above can be replaced by various kind of data, according to specific needs.
