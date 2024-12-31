--- COMPILATION ---

A Makefile is provided. 
The EIGEN library must be installed in your system.

LINUX DISTRIBUTIONS:
To compile both versions, type 'make' into the command line. 
If you only want to compile one of the versions, type 'make kmeanspp' for the standard k-means++ algorithm, 'make kmeanspp_tie' for the accelerated version with using only the TIE or 'make kmeanspp_tie_norm' for the accelerated version using both the TIE and Norm filters.


--- PARAMETERS ---

-input [file]: Path to the source .csv file, where each row represents a point. [MANDATORY]

-n_clusters [int]: Desired number of clusters (k). Default: 4096.

seed [int]: Seed value for random number generation (s). Default: 0.


--- OUTPUT ---

The algorithm outputs:
- Time: Execution time (in seconds).
- Score: The Sum of Squared Distances (SSD) for the obtained clustering.
- A list of all selected centers, printed sequentially.


Command line example:
./kmeanspp_tie_norm -input [path] -n_clusters 32 -seed 10
