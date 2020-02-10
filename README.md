# High_Dimensional_Logistic_Regression

Goal: to correct the bias in the coefficients (betas) produced from MLE in high dimensions.  See this paper: https://www.pnas.org/content/116/29/14516.short

This is a work in progress. Distributed under the 3-Clause BSD license.

Package Outline:

1. **.help**  
   a. Should have some examples, with the provided simulated datasets.  
   b. Should explain each function.
2. **.data**  
   a. Simulated datasets, of varying n, p. Stores the known signals as well.  
3. **logistic.params** (similar/built on top of sklearn)  
   a. Should check for existance of MLE  
   b. Should correct the bias with formula from paper  
4. **.diagnostics**  
   a. Any out of the box plots?  

