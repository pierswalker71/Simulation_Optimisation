# Simulation_Optimisation

**Introduction**<br>
This programme has been created to demonstrate the use of Machine Learning to optimise the output of a simulator.

**Business problem being solved: The optimisation of a simulation tool** <br>
A company uses a simulation tool to model business procesess, but would like to optimise the components of the model to produce a desired output (e.g. number of widgets manufactured) at the lowest cost.

This notebook demonstrates that a simulator may be treated as a black box whose input-output relationship is learnt by machine-learning regression models.
The regression model can then be searched - significantly much faster than a typical simulation can be run - to determine the best input parameters to achieve a given target output for the lowest cost. This approach may also be applied to optimise an practical experiment which yields a single measured value.

In this example the value of 5 integer inputs are optimised (e.g. corresponding to the number of manufacturing machines in 5 depots) to produce a target outcome of 80 (e.g. 80 widgets/day throughput).

**Techniques employed:**
1. Generation of complex multi-dimensional function.
2. Machine Learning regression using Sklearn; including train-test-split.
3. Use of ML Pipelines; including data-scaling and polynomial transformations.
3. Hyper-parameter optimisation using deterministic or stochastic grid-search.
4. Global optimisation using Scipy.
5. Data manipulation using Pandas and Numpy.

**Programme Process Steps:**
1. Input Parameters
Constants such as the target simulation output value and the costs associated with the input parameters is set by the user.
2. Run Simulator To Conduct Exploratory Runs
Runs the simulator with a wide range of random values for each input parameter to obtain and record the associated simulator output.
The purpose is to generate sufficient data to construct a reliable Machine Learning regression model.
Note: Currently a simulator is emulated; replaced by a function that generates a value for a set of inputs.
3. Build and Test Regression Models
A set of different SKLearn regression models are defined within a pipeline, along with example hyper-parameters that could be optimised. Simple linear models such as LinearRegression or Lasso are preceeded by a PolynomialFeatures approach to pre-trasform the data.
The set of regression models are first optimised via cross-validation on the trainng data employing either GridSearchCV or RandomizedSearchCV.
Next the k-fold cross-validation is performed on the training data to generate an 'r-squared' cross_validation score to allow the set of regressors to be ranked.
The top performing regression model is selected and it's performance against the testset is displayed via a true vs predicted value scatter plot.
4. Derive Predicted Optimal Input Parameters
The regresion model has now learnt the interaction between the Simulators' inputs and outputs, which is treated as a black-box. This solution-landscape is navigated using the differential_evolution global optimiser from Scipy. Each parameter value may be considered as a resource with an associated cost proportional to the quantity of that parameter - the sum of each parameter cost is summed to generate a total cost associated with the set of inputs. The Optimiser is used to recommend a predicted optimal set of input parameter values to meet a given output target value, for the lowerest cost.
5. Run Simulator With Concentrated Optimal Runs
The simulator is run using the predicted optimal set of input parameter values, together with a range of runs in which the values are perturbed by a small random amount to perform a concentrated search around this solution.
6. Determine Best Compliant Result
All the resulting runs are reviewed for their closeness to the target Simulator output (e.g. 60), within a given tolerance (+-0.5). The 'compliant' runs with the lowest costs are presented.
7. Process Repetition:
The group of exploratory and optimal simulation runs are then run repeatedly within a loop to provide continued improvement.

**Comments:**
At the end of each loop the best compliance solutions are presented as a heatmap.

As the loops progress - and more data is fed into the ML model - the better its performance. This may be seen by the closeness of the predicted output of the optimal array in Step 4, and the actual output determined in Step 5. It is also shown by the 'Regression Model Performance Against Unseen Test Data' scatter plot.

Also presented at the end of each loop is a chart demonstrating the reduction in the cost of the best compliant solution as a function of loop number.
