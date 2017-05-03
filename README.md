# vimpy
Python package for assessing variable importance nonparametrically. This Python code is based on theoretical results, and accompanies an R package and paper. 

The functionality of the Python package is more limited than that of the R package. Here I require you to estimate the conditional means $E_P(Y \mid X = x)$ and $E_P(Y \mid X_{(-j)} = x_{(-j)})$ --- where $X_{(-j)}$ is the vector $X$ with the $j$th component(s) removed --- prior to using the variable importance function. You can also only estimate variable importance for one feature (or group of features) at a time using this function.

However, the basic code is the same as the R package. I just leave you to do a bit more work!
